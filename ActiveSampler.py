# pylint: disable=no-member

import os
import uuid
from abc import ABC, abstractmethod
from itertools import islice

import numpy as np
import utils.lib.utils as lib_utils
from keras import ops
from PIL import Image
from utils.lib import log
from utils.lib.config import load_config_from_yaml

from datasets import get_datasets
from utils import mri
from load_model import load_model
from measurements import prepare_measurement
from utils.keras_utils import (
    load_img_as_tensor,
    normalize,
    postprocess_image,
    translate,
)


def get_active_diffusion_sampler_class(data_domain):
    """Get the active diffusion sampler class based on the data domain."""
    if data_domain == "image":
        return ImageActiveSampler
    elif data_domain == "kspace":
        return KSpaceActiveSampler
    else:
        raise ValueError(f"data domain `{data_domain}` was not recognised.")


class ActiveSampler(ABC):
    """
    Core functions and state variables for active sampling using diffusion.
    """

    def __init__(
        self,
        image_shape,
        diffusion_model_run_dir,
        target_img_path,
        selection_strategy,
        hard_consistency=False,
        initial_measurement=False,
        sigma=1,
        pixel_region_radius=None,
        data_root=None,
    ):
        """
        hard_consistency (bool): whether or not true measurements should be
            combined with final reconstruction.
        initial_measurement (bool): whether or not a pre-defined measurement
            should be included in the initial mask.
        dataset_variance_map (tensor of shape [1, H, W, 1] where H, W are the data shape): map of variance
            across samples in training set, used for setting predefined mask
        sigma (float): hyperparameter for entropy based sampling methods
        pixel_region_radius (int): radius around selected pixel for windowed sampling. Defaults to None.
            Only works for pixel_variance selection strategy currently.
        data_root (str): the directory containing your train and val datasets as specified
            by the ddim config in diffusion_model_run_dir.
        """
        self.selection_strategy = selection_strategy
        self.image_shape = image_shape
        self.pixel_region_radius = pixel_region_radius
        self.data_root = data_root

        if self.pixel_region_radius is not None:
            assert (
                self.selection_strategy == "pixel_variance"
            ), "pixel_region_radius is only supported for pixel_variance selection strategy."

        if str(diffusion_model_run_dir) == "stable_diffusion":
            self.diffusion_config = load_config_from_yaml(
                "configs/stable_diffusion/config.yaml"
            )
            log.info(
                "Loading StableDiffusion model with parameters from "
                f"{log.yellow('configs/stable_diffusion/config.yaml')}."
            )
        else:
            self.diffusion_config = load_config_from_yaml(
                diffusion_model_run_dir / "config.yaml"
            )

        self.diffusion_model = load_model(
            diffusion_model_run_dir / "checkpoints",
            stable_diffusion_kwargs={
                "img_height": image_shape[0],
                "img_width": image_shape[1],
            },
            image_shape=image_shape,
        )

        self.target_img = self.load_target_img(target_img_path)

        datestring = lib_utils.get_date_string()

        self.save_dir = (
            diffusion_model_run_dir / "inference" / (datestring + "_active_sampling")
        )
        self.save_dir.mkdir(parents=True)
        self.initial_measurement = initial_measurement
        self.operator, self.measurement = self.initialise_operator()
        self.hard_consistency = hard_consistency

        if selection_strategy == "pixel_random":
            self.measurement_selection_fn = None
        elif selection_strategy == "pixel_data_variance":
            self.measurement_selection_fn = None
        elif selection_strategy == "pixel_variance":
            self.measurement_selection_fn = self.select_pixel_variance
        elif selection_strategy == "column_variance":
            self.measurement_selection_fn = self.select_column_variance
        elif selection_strategy == "column_random":
            self.measurement_selection_fn = None
        elif selection_strategy == "column_data_variance":
            self.measurement_selection_fn = None
        elif selection_strategy == "pixel_entropy":
            self.measurement_selection_fn = lambda x: self.select_pixel_entropy(
                x, sigma=sigma
            )
        elif selection_strategy == "column_entropy":
            self.measurement_selection_fn = lambda x: self.select_column_entropy(
                x, sigma=sigma
            )
        elif selection_strategy == "column_equispaced":
            self.measurement_selection_fn = None
        elif selection_strategy == "fastmri_baseline":
            self.measurement_selection_fn = None
        else:
            raise ValueError(
                f"selection_strategy={selection_strategy} is not a valid option."
            )

    @abstractmethod
    def initialise_operator(self):
        """
        This should be implemented for each individual sampler
        """
        return

    def select_column_variance(self, posterior_variance):
        """
        Returns the indices for the column with the highest poserior variance.

        Params:
            posterior_variance (tensor of shape [1, H, W, C])

        Returns:
            array indices for pixels in the column with highest variance.
            the array indices are a tuple with a stack of indices per dim, as in:
            https://stackoverflow.com/questions/41900738/assign-values-to-different-index-positions-in-numpy-array
        """
        max_var_column = ops.argmax(ops.sum(posterior_variance, axis=(0, 1, 3)))

        # Create array of indices corresponding with the chosen column
        mask_shape = self.operator.mask.shape
        max_var_column_indices = np.zeros(
            (mask_shape[1], len(mask_shape) - 1), dtype=np.uint
        )  # `len(mask_shape) - 1` --> don't include channel dim --> mask is broadcast across channels
        max_var_column_indices[:, 1] = np.indices((mask_shape[1],))
        max_var_column_indices[:, 2] = max_var_column

        return tuple(max_var_column_indices.T)

    def select_pixel_variance(self, posterior_variance):
        """
        Returns the index for the pixel with the highest poserior variance.

        Params:
            posterior_variance (tensor of shape [1, H, W, C])
        Returns:
            an array of length 4 (one int for each dim) indicating the index of the max variance pixel.
        """

        maximum_variance_pixel_index = np.unravel_index(
            ops.argmax(posterior_variance), posterior_variance.shape
        )
        return maximum_variance_pixel_index

    @staticmethod
    def select_window_around_pixels(image_shape, selected_pixel, radius=1):
        """
        Selects a window of pixels around the given selected pixels within the specified radius.

        Args:
            image_shape (tuple): The shape of the image.
            selected_pixel (list): A 4D tuple with coordinates of the selected pixel.
            radius (int, optional): The radius around the selected pixels. Defaults to 1.

        Returns:
            tuple: The selected window of pixels. The window is a tuple of arrays, one for each dimension.

        """
        region = np.indices((radius + 1, radius + 1)).reshape(2, -1).T
        # center the kernel
        region = region - radius // 2
        region += np.array(selected_pixel)[1:-1]
        region = np.clip(region, 0, image_shape[1:3])
        region = np.insert(region, (0, 2), 0, axis=1)
        region = tuple(region.T)
        return region

    def select_pixel_entropy(self, measurement_particles, sigma=1):
        error_matrices = ops.convert_to_tensor(
            [
                ops.convert_to_tensor(
                    [(particle_i - particle_j) for particle_i in measurement_particles]
                )
                for particle_j in measurement_particles
            ]
        )
        # sum across channels to get l2 per pixel
        squared_l2_per_pixel_i_j = ops.sum(error_matrices**2, axis=[-1])
        gaussian_error_per_pixel_i_j = ops.exp(
            (squared_l2_per_pixel_i_j) / (2 * sigma**2)
        )
        entropy_per_pixel_i = ops.sum(gaussian_error_per_pixel_i_j, axis=1)
        entropy_per_pixel = ops.sum(ops.log(entropy_per_pixel_i), axis=0)
        # set entropy for already-measured lines to 0
        # TODO: this is now hardcoded to check for MRI data, but should be generic
        if self.operator.mask.shape[-1] == 2:
            entropy_per_pixel = (
                entropy_per_pixel * ops.logical_not(self.operator.mask)[0, ..., 0]
            )
        else:
            entropy_per_pixel = (
                entropy_per_pixel * ops.logical_not(self.operator.mask).squeeze()
            )
        return np.unravel_index(
            ops.argmax(entropy_per_pixel), shape=self.operator.mask.shape
        )

    def select_column_entropy(self, measurement_particles, sigma=1):
        error_matrices = ops.convert_to_tensor(
            [
                ops.convert_to_tensor(
                    [(particle_i - particle_j) for particle_i in measurement_particles]
                )
                for particle_j in measurement_particles
            ]
        )
        # sum across rows and complex channels to get l2 per line
        squared_l2_per_line_i_j = ops.sum(error_matrices**2, axis=[-3, -1])
        gaussian_error_per_line_i_j = ops.exp(
            (squared_l2_per_line_i_j) / (2 * sigma**2)
        )
        entropy_per_line_i = ops.sum(gaussian_error_per_line_i_j, axis=1)
        entropy_per_line = ops.sum(ops.log(entropy_per_line_i), axis=0)
        taken_lines = ops.sum(self.operator.mask, axis=[0, 1, 3])
        # set entropy for already-measured lines to 0
        entropy_per_line = entropy_per_line * ops.logical_not(taken_lines)
        return ops.argmax(entropy_per_line)

    def preprocess(self, x):
        """
        Pre-process function is identity by default
        """
        return x

    def postprocess(self, x):
        """
        Post-process function is identity by default
        """
        return x

    def sample_and_reconstruct(
        self,
        num_samples_to_take,
        sampling_window,
        posterior_shape,
        num_diffusion_steps,
        guidance_kwargs=None,
        guidance_method="dps",
        verbose=True,
        plot_callback=None,
        plotting_interval=None,
    ):
        """
        Run diffuion_model.active_sampling with the current agent state

        Params:
            num_samples_to_take (int): the number of measurements to take
            sampling_window (list of 2 integers): the diffusion steps at which
                to start and stop sampling
            posterior_shape (tuple): the shape of the diffusion model output
            num_diffusion_steps: the number of diffusion steps for the diffusion model
                to take
            guidance_kwargs: dict of kwargs specific to guided diffusion algorithm
            verbose: whether or not to print progbar and plot outputs
            plot_callback (PlotActiveInference): instance of active inference plotter

        Returns:
            poserior_samples (tensor): images reconstructed through active sampling
            measurements (tensor): total set of measurements taken by the sampler.
                this should equal active_sampler.operator.forward(active_sampler.target_img)
        """
        posterior_samples, measurements, _ = self.diffusion_model.active_sampling(
            self.target_img,
            self.operator,
            self.update_operator,
            num_samples_to_take=num_samples_to_take,
            sampling_window=sampling_window,
            image_shape=posterior_shape,
            diffusion_steps=num_diffusion_steps,
            guidance_method=guidance_method,
            guidance_kwargs=guidance_kwargs,
            verbose=verbose,
            plot_callback=plot_callback,
            plotting_interval=plotting_interval,
        )
        if verbose:
            mae = ops.mean(
                ops.abs(self.target_img - ops.mean(posterior_samples, axis=0))
            )
            log.info(f"MAE: {mae:.4f}")

        return posterior_samples, measurements

    def save_result(self, posterior_samples, measurements):
        """
        Save the results of the active sampling run to the save directory.
        """
        posterior_mean = ops.mean(posterior_samples, axis=0)
        posterior_mean = self.postprocess(posterior_mean)

        posterior_samples = self.postprocess(posterior_samples)
        measurements = self.postprocess(measurements)
        mask = self.operator.mask
        mask = mask * 255
        # # make mask binary
        # mask = ops.where(mask > 0.5, 1, -1)
        # mask = self.postprocess(mask)
        target = self.postprocess(self.target_img)

        images = {
            "posterior_mean": posterior_mean,
            "target": target,
            "mask": mask,
            "measurement": measurements,
            **{
                f"posterior_sample_{i}": posterior_sample
                for i, posterior_sample in enumerate(posterior_samples)
            },
        }

        for key, image in images.items():
            path = (self.save_dir / key).with_suffix(".png")
            image = ops.convert_to_numpy(image)
            image = np.squeeze(image).astype("uint8")
            Image.fromarray(image).save(path)
            log.info(f"Saved {key} to {path}")
        return images


class ImageActiveSampler(ActiveSampler):
    """
    Active sampling for the image domain.

    Includes data loading, pre/post-processing, and measurement operator for
    active sampling images.
    """

    def load_target_img(self, target_img_path):
        """
        Loads and pre-processes an image from a given path.

        Params:
            target_img_path (str): path to target image
        Returns:
            preprocessed image loaded into a tensor
        """
        if target_img_path.startswith("validation_dataset_"):
            raise UserWarning(
                "Loading validation set images is currently only implemented for fastMRI. Please specify an absolute path for other datasets."
            )
        if target_img_path == "benchmark":
            # TODO: improve data loading for benchmarking
            return ops.zeros(self.image_shape)[None, ...]
        else:
            target_img = load_img_as_tensor(
                str(target_img_path),
                image_shape=self.image_shape[:2],
                grayscale=bool(self.image_shape[-1] == 1),
            )
            target_img = ops.expand_dims(target_img, axis=0)
            log.info(f"Loaded target image from {log.yellow(target_img_path)}")
            return self.preprocess(target_img)

    def initialise_operator(self):
        """
        Initialises operator and measurement for image subsampling.
        """
        if self.initial_measurement is True:
            raise NotImplementedError(
                "Smart initial measurements have not yet ben implemented for image domain. Please set self.initial_measurement=False"
            )
        initial_mask = ops.zeros(self.target_img.shape)
        operator, measurement = prepare_measurement(
            "inpainting",
            ops.convert_to_tensor(
                self.target_img
            ),  # do we need to convert to tensor here?
            mask=initial_mask,
        )
        return operator, measurement

    def update_operator(self, pred_images):
        """
        Adds a new pixel or column to the subsampling mask, as per the
        sampling strategy.
        """
        if self.selection_strategy in [
            "pixel_entropy",
            "pixel_random",
            "pixel_data_variance",
        ]:
            particles_in_measurement_space = pred_images
            selected_pixel = self.measurement_selection_fn(
                particles_in_measurement_space
            )
            self.operator.mask = self.operator.mask.at[selected_pixel].set(1)
            return self.operator
        elif self.selection_strategy in [
            "column_entropy",
            "column_random",
            "column_data_variance",
        ]:
            particles_in_measurement_space = pred_images
            selected_column = self.measurement_selection_fn(
                particles_in_measurement_space
            )
            self.operator.mask = self.operator.mask.at[:, :, selected_column, :].set(1)
            return self.operator
        else:
            posterior_variance = ops.var(pred_images, axis=0)[None, ...]
            # prevent re-sampling (NOTE: ideally the model would know not to do this)
            taken_measurements = self.operator.mask
            posterior_variance = posterior_variance * ops.logical_not(
                taken_measurements
            )
            selected_indices = self.measurement_selection_fn(posterior_variance)
            if self.pixel_region_radius is not None:
                selected_indices = self.select_window_around_pixels(
                    posterior_variance.shape,
                    selected_indices,
                    radius=self.pixel_region_radius,
                )
            for channel_idx in range(self.operator.mask.shape[-1]):
                selected_indices = list(selected_indices)  # Convert tuple to list
                selected_indices[-1] = channel_idx
                self.operator.mask = self.operator.mask.at[tuple(selected_indices)].set(
                    1
                )  # Convert back to tuple
            return self.operator

    def preprocess(self, x):
        """
        Maps an image from (0, 255) -> the image range for the diffusion model
        """
        return translate(
            x,
            (0, 255),
            self.diffusion_model.image_range,
        )

    def postprocess(self, x):
        """
        Maps an image from the image range of the diffusion model -> (0, 255)
        """
        return postprocess_image(
            x,
            self.diffusion_config.data.normalization,
        )


class KSpaceActiveSampler(ActiveSampler):
    """
    Active sampling for the k-space domain.

    Includes data loading, pre/post-processing, and measurement operator for
    active sampling MRI data.
    """

    def load_target_img(self, target_img_path):
        """
        Loads a target image from the validation dataset specified in the diffusion model config.yaml
        """
        if target_img_path.startswith("validation_dataset_"):
            target_index = int(target_img_path.split("_")[-1])
            _, val_dataset = get_datasets(
                self.data_root,
                config=self.diffusion_config,
                batch_size=1,
            )
            return ops.convert_to_tensor(
                next(islice(iter(val_dataset), target_index, target_index + 1))
            )
        elif target_img_path == "benchmark":
            # TODO: better implementation
            return ops.zeros(self.diffusion_config.data.image_shape)[None, ...]
        else:
            raise NotImplementedError(
                "Sepecific target paths are not yet supported. Please use target_img=validation_dataset_{i} to run inference on the ith sample."
            )

    def initialise_operator(self):
        """
        Initialise masked fourier operator and measurement
        """
        initial_mask = ops.zeros(self.target_img.shape)
        if self.initial_measurement:
            if self.selection_strategy == "column_entropy":
                initial_mask = ops.zeros(self.target_img.shape)
                # start with center line measurement
                initial_mask = initial_mask.at[:, :, 63, :].set(1)
            elif self.selection_strategy == "pixel_entropy":
                initial_mask = ops.zeros(self.target_img.shape)
                # start with center line measurement
                # TODO: make this a square instead of just single pixel
                initial_mask = initial_mask.at[:, 64:65, 64:65, :].set(1)
        operator, measurement = prepare_measurement(
            "masked_fourier", ops.convert_to_tensor(self.target_img), mask=initial_mask
        )
        return operator, measurement

    def update_operator(self, pred_images):
        """
        Compute posterior variance in the k-space and select next measurement mask
        """
        if self.selection_strategy == "column_variance":
            kspace_posterior = mri.fft2c(pred_images)
            posterior_variance = ops.var(kspace_posterior, axis=0)[None, ...]
            # prevent re-sampling (NOTE: ideally the model would know not to do this)
            taken_measurements = self.operator.mask
            posterior_variance = posterior_variance * ops.logical_not(
                taken_measurements
            )
            selected_indices = self.measurement_selection_fn(posterior_variance)
            self.operator.mask = self.operator.mask.at[selected_indices].set(1)
            return self.operator
        elif self.selection_strategy == "column_entropy":
            particles_in_kspace = mri.fft2c(pred_images)
            selected_column = self.measurement_selection_fn(particles_in_kspace)
            self.operator.mask = self.operator.mask.at[:, :, selected_column, :].set(1)
            return self.operator
        elif self.selection_strategy == "pixel_entropy":
            particles_in_kspace = mri.fft2c(pred_images)
            selected_indices = self.measurement_selection_fn(particles_in_kspace)
            self.operator.mask = self.operator.mask.at[selected_indices].set(1)
            return self.operator
        else:
            raise NotImplementedError(
                f"Selection strategy {self.selection_strategy} has not been implemented for KSpaceActiveSampler"
            )

    def postprocess(self, x):
        """
        Computes the magnitude of a complex input for visualisation
        """
        x = mri.complex_abs(x)[..., None]
        x = normalize(x)

        return postprocess_image(
            x,
            self.diffusion_config.data.normalization,
        )

    def sample_and_reconstruct(
        self,
        num_samples_to_take,
        sampling_window,
        posterior_shape,
        num_diffusion_steps,
        guidance_kwargs=None,
        guidance_method="dps",
        verbose=True,
        plot_callback=None,
        plotting_interval=None,
    ):
        """
        Run diffusion_model.active_sampling with the current agent state

        Params:
            num_samples_to_take (int): the number of measurements to take
            sampling_window (list of 2 integers): the diffusion steps at which
                to start and stop sampling
            posterior_shape (tuple): the shape of the diffusion model output
            num_diffusion_steps: the number of diffusion steps for the diffusion model
                to take
            guidance_kwargs: dict of kwargs specific to guided diffusion algorithm
            verbose: whether or not to print progbar and plot outputs
            plot_callback (PlotActiveInference): instance of active inference plotter

        Returns:
            poserior_samples (tensor): images reconstructed through active sampling
            measurements (tensor): total set of measurements taken by the sampler.
                this should equal active_sampler.operator.forward(active_sampler.target_img)
        """
        posterior_samples, measurements = super().sample_and_reconstruct(
            num_samples_to_take,
            sampling_window,
            posterior_shape,
            num_diffusion_steps,
            guidance_kwargs,
            guidance_method,
            verbose,
            plot_callback,
            plotting_interval=plotting_interval,
        )

        if self.hard_consistency:
            kspace_with_measurements = (
                mri.fft2c(posterior_samples) * ops.logical_not(self.operator.mask)
            ) + measurements
            posterior_samples = mri.ifft2c(kspace_with_measurements)

        return posterior_samples, measurements
