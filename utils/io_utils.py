import os
from pathlib import Path

import keras
import keras.ops as ops
import matplotlib.pyplot as plt
import utils.lib.utils as lib_utils
from PIL import Image
from utils.lib import log
from utils.lib.io_lib import matplotlib_figure_to_numpy

from utils.keras_utils import plot_image_grid


class PlotActiveInference:
    def __init__(self, postprocess_func, target_img, plotting_interval=1):
        self.postprocess_func = postprocess_func
        self.plotting_interval = plotting_interval

        self.measurements_buffer = []
        self.pred_images_buffer = []
        self.step_buffer = []
        self.cmap = "gray" if self.postprocess_func(target_img).shape[-1] == 1 else None

        self.fig_pred = None
        self.pred_fig_contents = None
        self.fig_overview = None

    def add_to_buffer(self, step, measurements, pred_images, noisy_images):
        self.measurements_buffer.append(measurements)
        self.pred_images_buffer.append(pred_images)
        self.step_buffer.append(step)

    def create_animation(self, target_img, save_dir, filename, fps=1):
        log.info("Animating active inference...")
        frames = []
        progbar = keras.utils.Progbar(len(self.measurements_buffer))
        for step, measurements, images_from_posterior in zip(
            self.step_buffer,
            self.measurements_buffer,
            self.pred_images_buffer,
        ):
            frame = self.plot_active_diffusion_step_overview(
                step,
                measurements,
                target_img,
                images_from_posterior,
                save_dir=save_dir,
            )
            frames.append(frame)
            progbar.add(1)

        # repeat last frame 10% more
        frames = frames + [frames[-1]] * int(len(frames) * 0.1)

        lib_utils.save_to_gif(frames, os.path.join(save_dir, filename), fps=fps)

    def plot_active_diffusion_step_overview(
        self,
        step,
        measurements,
        target_img,
        pred_images,
        save_dir=None,
    ):
        """
        Plot the overview of an active diffusion step.

        Args:
            step (int): The current step number.
            measurements (ndarray): The measurement mask.
            target_img (ndarray): The target image.
            pred_images (ndarray): The posterior samples.

        Returns:
            ndarray: The overview plot as a numpy array.

        """
        target_img = self.postprocess_func(target_img)
        measurements = self.postprocess_func(measurements)
        pred_images = self.postprocess_func(pred_images)

        if self.fig_pred is None:
            self.fig_pred, self.pred_fig_contents = plot_image_grid(pred_images)
            self.fig_pred.patch.set_facecolor("white")
            self.fig_pred.tight_layout()
        else:
            plot_image_grid(
                pred_images, fig=self.fig_pred, fig_contents=self.pred_fig_contents
            )

        pixelwise_variance = ops.mean(ops.var(pred_images, axis=0), axis=-1)

        # convert each figure into rgb and plot next to each other in a new figure
        fig_pred = matplotlib_figure_to_numpy(self.fig_pred)
        if save_dir:
            save_dir = Path(f"{save_dir}/steps")
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            Image.fromarray(fig_pred).save(f"{save_dir}/pred_{step}.png")

        if self.fig_overview is None:
            self.fig_overview, axs = plt.subplots(1, 4, figsize=(14, 5))
            axs[0].imshow(target_img[0, :, :], cmap=self.cmap)
            axs[0].set_title("Target $\\mathbf{x}$", fontsize=15)
            axs[1].imshow(measurements[0, :, :], cmap=self.cmap)
            axs[1].set_title("Measurement mask $\\mathbf{m}$", fontsize=15)
            axs[2].imshow(pixelwise_variance)
            axs[2].set_title("Posterior variance", fontsize=15)
            axs[3].imshow(fig_pred)
            axs[3].set_title(
                "Posterior samples $p(\\mathbf{x} | \\mathbf{y})$", fontsize=15
            )
            self.fig_overview.tight_layout()
            for ax in axs:
                ax.axis("off")
        else:
            axs = self.fig_overview.axes
            axs[0].get_images()[0].set_data(target_img[0, :, :])
            axs[1].get_images()[0].set_data(measurements[0, :, :])
            axs[2].get_images()[0].set_data(pixelwise_variance)
            axs[3].get_images()[0].set_data(fig_pred)
            self.fig_overview.suptitle(
                f"Step {int(step)}/{int(self.step_buffer[-1])}", fontsize=20
            )

        plt.close("all")
        return matplotlib_figure_to_numpy(self.fig_overview)
