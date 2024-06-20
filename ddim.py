"""
DDIM model
Largely taken from https://keras.io/examples/generative/ddim/
"""

from pathlib import Path

import h5py
import keras
import tensorflow as tf
from keras import ops
from utils.usbmd import log

from guidance.latent_guidance import get_guidance as get_latent_guidance
from guidance.pixel_guidance import get_guidance as get_pixel_guidance
from load_model import load_model
from losses import KID
from models.unet import get_network
from utils.keras_utils import get_ram_info


@keras.saving.register_keras_serializable()
class DiffusionModel(keras.Model):
    def __init__(
        self,
        image_shape,
        widths,
        block_depth,
        ema_val=0.999,
        min_signal_rate=0.02,
        max_signal_rate=0.95,
        diffusion_steps=20,
        image_range=(0, 1),
        compute_kid=False,
        kid_diffusion_steps=5,
        kid_image_shape=None,
        mean=None,
        variance=None,
        latent_diffusion=False,
        latent_shape=None,
        autoencoder_checkpoint_directory=None,
        old_vae_backbone=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_shape = image_shape
        self.latent_shape = latent_shape
        self.widths = widths
        self.block_depth = block_depth
        self.ema_val = ema_val
        self.min_signal_rate = min_signal_rate
        self.max_signal_rate = max_signal_rate
        self.diffusion_steps = diffusion_steps
        self.image_range = image_range
        self.compute_kid = compute_kid
        self.kid_diffusion_steps = kid_diffusion_steps
        self.kid_image_shape = kid_image_shape if kid_image_shape else image_shape

        self.mean = mean
        self.variance = variance
        self.latent_diffusion = latent_diffusion
        self.autoencoder_checkpoint_directory = autoencoder_checkpoint_directory

        self._image_encoder = None
        self._decoder = None
        self.noise_loss_tracker = None
        self.image_loss_tracker = None

        self.img_height, self.img_width = self.image_shape[:2]
        if self.latent_diffusion:
            if self.latent_shape is None:
                self.latent_shape = [self.img_height // 8, self.img_width // 8, 4]
                log.info(
                    f"Using predefined latent shape {log.yellow(self.latent_shape)} "
                    "as it was not defined in `config.model`"
                )
            self.network = get_network(self.latent_shape, self.widths, self.block_depth)
        else:
            self.latent_shape = None
            self.network = get_network(self.image_shape, self.widths, self.block_depth)

        self.ema_network = keras.models.clone_model(self.network)

        # for tracking the progress of the reverse diffusion
        self.track_progress = []

        assert len(self.image_range) == 2, "image_range must be a tuple of (min, max)"
        assert self.image_range[0] < self.image_range[1], "min must be less than max"
        assert self.min_signal_rate < self.max_signal_rate, "min must be less than max"

        if self.latent_diffusion:
            if self.autoencoder_checkpoint_directory is None:
                raise ValueError(
                    "latent_diffusion requires a pretrained autoencoder model to be specified"
                )
            self.autoencoder = load_model(
                self.autoencoder_checkpoint_directory,
                image_shape=image_shape,
                old_backbone=old_vae_backbone,
            )
            # TODO: i think these autoencoder weighs are being overwritten by load_model of ddim itself
            assert self.autoencoder.image_range == self.image_range, (
                f"image_range mismatch between autoencoder ({self.autoencoder.image_range}) "
                f"and DiffusionModel ({self.image_range})"
            )

            if self.autoencoder.latent_dim is not None:
                if hasattr(self.autoencoder, "latent_shape"):
                    assert self.autoencoder.latent_shape == self.latent_shape, (
                        f"latent_shape mismatch between autoencoder ({self.autoencoder.latent_shape}) "
                        f"and DiffusionModel ({self.latent_shape}). Please update this in your configs."
                    )
        else:
            assert (
                self.autoencoder_checkpoint_directory is None
            ), "latent_diffusion must be True if autoencoder_checkpoint_directory is specified"

    def compile(self, run_eagerly=None, jit_compile="auto", **kwargs):
        super().compile(run_eagerly=run_eagerly, jit_compile=jit_compile, **kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        if self.compute_kid:
            self.kid = KID("kid", self.image_shape, self.kid_image_shape)
        if self.latent_diffusion:
            self.autoencoder.compile(run_eagerly=run_eagerly, jit_compile=jit_compile)

        if jit_compile:
            log.info("Model has been JIT compiled")
        if run_eagerly:
            log.warning("Model is running eagerly")

    @property
    def metrics(self):
        if self.compute_kid:
            return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]
        else:
            return [self.noise_loss_tracker, self.image_loss_tracker]

    def diffusion_schedule(self, diffusion_times):
        """Cosine diffusion schedule https://arxiv.org/abs/2102.09672
        Args:
            diffusion_times: tensor with diffusion times in [0, 1]
        Returns:
            noise_rates: tensor with noise rates
            signal_rates: tensor with signal rates

            according to:

            x_t = signal_rate * x_0 + noise_rate * noise
            x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise

        """
        # diffusion times -> angles
        start_angle = ops.cast(ops.arccos(self.max_signal_rate), "float32")
        end_angle = ops.cast(ops.arccos(self.min_signal_rate), "float32")

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = ops.cos(diffusion_angles)
        noise_rates = ops.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1
        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(
        self,
        initial_noise,
        diffusion_steps,
        initial_samples=None,
        initial_step=0,
        verbose=False,
    ):
        # reverse diffusion = sampling
        num_images = ops.shape(initial_noise)[0]
        step_size = 1.0 / ops.cast(diffusion_steps, "float32")

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        progbar = keras.utils.Progbar(diffusion_steps, verbose=verbose)

        assert (
            initial_step >= 0
        ), f"initial_step must be non-negative, got {initial_step}"
        assert (
            initial_step < diffusion_steps
        ), f"initial_step must be less than diffusion_steps, got {initial_step}"

        if initial_samples is not None and initial_step > 0:
            starting_diffusion_times = ops.ones((num_images, 1, 1, 1)) - (
                (initial_step - 1) * step_size
            )
            noise_rates, signal_rates = self.diffusion_schedule(
                starting_diffusion_times
            )
            next_noisy_images = (
                signal_rates * initial_samples + noise_rates * initial_noise
            )
        else:
            next_noisy_images = initial_noise

        # for diffusion animation we keep track of the diffusion progress
        # for large number of steps, we do not store all the images due to memory constraints
        self.track_progress = []
        if diffusion_steps > 50:
            track_progress_interval = diffusion_steps // 50
        else:
            track_progress_interval = 1

        for step in range(initial_step, diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = ops.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step
            progbar.update(step + 1)

            if step % track_progress_interval == 0:
                self.track_progress.append(pred_images)

        return pred_images

    def get_guidance_fn(
        self,
        guidance_method,
        operator,
    ):
        # get either latent or pixel guidance function
        if self.latent_diffusion:
            guidance_fn = get_latent_guidance(guidance_method)(self)
        else:
            guidance_fn = get_pixel_guidance(guidance_method)(self)

        def run_guidance(
            noisy_images,
            measurement,
            noise_rates,
            signal_rates,
            next_signal_rates,
            next_noise_rates,
            **guidance_kwargs,
        ):

            gradients, (measurement_error, (pred_noises, pred_images)) = guidance_fn(
                noisy_images,
                measurement=measurement,
                operator=operator,
                noise_rates=noise_rates,
                signal_rates=signal_rates,
                **guidance_kwargs,
            )
            gradients = ops.nan_to_num(gradients)
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            next_noisy_images = next_noisy_images - gradients
            return next_noisy_images, measurement_error, pred_images

        return run_guidance

    def generate(
        self,
        image_shape,
        diffusion_steps=None,
        initial_samples=None,
        initial_step=0,
        verbose=False,
        seed=None,
    ):
        # noise -> images -> denormalized images
        if diffusion_steps is None:
            diffusion_steps = self.diffusion_steps
        assert (
            len(image_shape) == 4
        ), "image_shape must be a tuple of (batch, height, width, channels)"
        num_images, image_height, image_width, n_channels = image_shape

        if self.latent_diffusion:
            if self.latent_shape is None:
                image_height = image_height // 8
                image_width = image_width // 8
                n_channels = 4
            else:
                image_height, image_width, n_channels = self.latent_shape

        initial_noise = keras.random.normal(
            shape=(num_images, image_height, image_width, n_channels),
            seed=seed,
        )
        if verbose:
            print("Generating images...")
        generated_images = self.reverse_diffusion(
            initial_noise,
            diffusion_steps,
            initial_samples=initial_samples,
            initial_step=initial_step,
            verbose=verbose,
        )
        # generated_images = self.denormalize(generated_images)

        if self.latent_diffusion:
            generated_images = self.decoder(generated_images)

        return generated_images

    def train_step(self, images):
        if self.latent_diffusion:
            images = self.image_encoder(images)

        batch_size, image_height, image_width, n_channels = ops.shape(images)

        # normalize images to have standard deviation of 1, like the noises
        # images = self.normalizer(images)
        noises = keras.random.normal(
            shape=(batch_size, image_height, image_width, n_channels)
        )

        # sample uniform random diffusion times
        diffusion_times = keras.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema_val * ema_weight + (1 - self.ema_val) * weight)

        # KID is not measured during the training phase for computational efficiency
        ram_usage = float(get_ram_info()["percentage"])
        return {m.name: m.result() for m in self.metrics} | {"RAM %": ram_usage}

    def test_step(self, images):
        if self.latent_diffusion:
            images = self.image_encoder(images)
        # normalize images to have standard deviation of 1, like the noises
        batch_size, image_height, image_width, n_channels = ops.shape(images)

        # images = self.normalizer(images)
        noises = keras.random.normal(
            shape=(batch_size, image_height, image_width, n_channels)
        )

        # sample uniform random diffusion times
        diffusion_times = keras.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        if self.compute_kid:
            # images = self.denormalize(images)
            image_shape = [batch_size, image_height, image_width, n_channels]
            generated_images = self.generate(
                image_shape=image_shape, diffusion_steps=self.kid_diffusion_steps
            )
            self.kid.update_state(images, generated_images)

        ram_usage = float(get_ram_info()["percentage"])
        return {m.name: m.result() for m in self.metrics} | {"RAM %": ram_usage}

    def active_sampling(
        self,
        image,
        initial_operator,
        update_operator_fn,
        num_samples_to_take,
        sampling_window,
        guidance_method="dps",
        initial_samples=None,
        initial_step=0,
        image_shape=None,
        diffusion_steps=None,
        guidance_kwargs=None,
        verbose=False,
        plot_callback=None,
        plotting_interval=50,
    ):
        if image_shape is None:
            image_shape = ops.shape(image)
        # noise -> images -> denormalized images
        if diffusion_steps is None:
            diffusion_steps = self.diffusion_steps

        # image = self.normalizer(image)
        assert (
            len(image_shape) == 4
        ), "image_shape must be a tuple of (batch, height, width, channels)"

        num_images, image_height, image_width, n_channels = image_shape

        if self.latent_shape:
            image_height = image_height // 8
            image_width = image_width // 8
            n_channels = 4

        initial_noise = keras.random.normal(
            shape=(num_images, image_height, image_width, n_channels)
        )

        num_images = ops.shape(initial_noise)[0]
        step_size = 1.0 / diffusion_steps

        # each guidance method has its own hyperparameters
        if guidance_kwargs is None:
            guidance_kwargs = {"omega": 10}

        # start the reverse conditional diffusion process
        progbar = keras.utils.Progbar(diffusion_steps, verbose=verbose)

        operator = initial_operator
        start_sampling, stop_sampling = sampling_window
        assert start_sampling >= initial_step, (
            "sampling_window = (start, stop), where start should be >= initial_step, "
            f"got sampling_window = {sampling_window}"
        )
        assert stop_sampling <= diffusion_steps, (
            "sampling_window = (start, stop), where stop should be <= diffusion_steps, "
            f"got sampling_window = {sampling_window}"
        )
        sampling_interval = (stop_sampling - start_sampling) // num_samples_to_take
        sampling_interval = max(sampling_interval, 1)
        measurements = operator.forward(image)
        run_guidance = self.get_guidance_fn(
            guidance_method,
            operator,
        )

        if initial_samples is not None and initial_step > 0:
            starting_diffusion_times = ops.ones((num_images, 1, 1, 1)) - (
                (initial_step - 1) * step_size
            )
            noise_rates, signal_rates = self.diffusion_schedule(
                starting_diffusion_times
            )
            next_noisy_images = (
                signal_rates * initial_samples + noise_rates * initial_noise
            )
        else:
            next_noisy_images = initial_noise
        for step in range(initial_step, diffusion_steps):
            noisy_images = next_noisy_images

            diffusion_times = ops.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images, measurement_error, pred_images = run_guidance(
                noisy_images,
                measurements,
                noise_rates,
                signal_rates,
                next_signal_rates,
                next_noise_rates,
                **guidance_kwargs,
            )
            if self.latent_diffusion:
                pred_images = self.decoder(pred_images)

            if (
                step >= start_sampling
                and step < stop_sampling
                and step % sampling_interval == 0
            ):

                operator = update_operator_fn(
                    pred_images,
                )
                measurements = operator.forward(image)

                # if verbose and plot_callback is not None:
                #     plot_callback.add_to_buffer(
                #         step, operator.mask, pred_images, noisy_images
                #     )

            elif plotting_interval is not None and step % plotting_interval == 0:
                if verbose and plot_callback is not None:
                    plot_callback.add_to_buffer(
                        step, operator.mask, pred_images, noisy_images
                    )

            # this new noisy image will be used in the next step
            progbar.update(step + 1, [("error", measurement_error)])

        return pred_images, measurements, operator

    @property
    def decoder(self):
        """decoder returns the diffusion image decoder model with pretrained
        weights. Can be overriden for tasks where the decoder needs to be
        modified.
        """
        return self.autoencoder.decode

    @property
    def image_encoder(self):
        """image_encoder returns the autoencoder Encoder with pretrained weights."""
        return self.autoencoder.encode

    def save_model_json(self, directory):
        """Save model as JSON file."""
        json_model = self.to_json()
        json_model_path = str(Path(directory) / "model.json")
        with open(json_model_path, "w", encoding="utf-8") as json_file:
            json_file.write(json_model)
        log.info(f"Succesfully saved model architecture to {json_model_path}")

    def load_weights(self, filepath, *args, **kwargs):
        # if 'layers/vae_model` is found in the weights file
        # we copy the weights to `autoencoder` and load the model
        with h5py.File(filepath, "r") as f:
            if "layers/vae_model" not in f:
                super().load_weights(filepath, *args, **kwargs)
            else:
                assert "layers/functional" in f, (
                    "The weights file must contain the 'layers/functional' group "
                    "to load the model weights"
                )
                temp_file = Path("temp.weights.h5")
                with h5py.File(temp_file, "w") as f_temp:
                    # copy layers/functional from f to f_temp
                    f.copy("layers/functional", f_temp, "layers/functional")
                    f.copy("vars", f_temp, "vars")
                    f.copy("optimizer", f_temp, "optimizer")
                    f.copy("ema_network", f_temp, "ema_network")
                    # copy layers/vae_model from f to f_temp
                    f.copy("layers/vae_model", f_temp, "autoencoder")
                super().load_weights(str(temp_file), *args, **kwargs)
                # delete temp file
                temp_file.unlink()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_shape": self.image_shape,
                "block_depth": self.block_depth,
                "widths": self.widths,
                "mean": self.mean,
                "variance": self.variance,
                "image_range": self.image_range,
                "latent_diffusion": self.latent_diffusion,
                "latent_shape": self.latent_shape,
                "autoencoder_checkpoint_directory": self.autoencoder_checkpoint_directory,
            }
        )
        return config
