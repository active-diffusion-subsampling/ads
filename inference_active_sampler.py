"""
Inference with ADS
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def parse_args():
    """Parse arguments for training DDIM."""
    parser = argparse.ArgumentParser(description="DDIM training")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/inference/mnist_pixels.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "-dr",
        "--data_root",
        type=str,
        default="data/",
        help="The root directory in which your train and validation sets are stored.",
    )
    parser.add_argument(
        "-r",
        "--target_img",
        type=str,
        default="validation_dataset_0",
        help="Path to target image, or validation_dataset_X in the case of fastMRI, where X indexes the desired validation sample.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="jax",
        help="ML backend to use",
        choices=["tensorflow", "torch", "jax"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=99,
        help="Random seed",
    )
    return parser.parse_args()


args = parse_args()
os.environ["KERAS_BACKEND"] = args.backend
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MPLBACKEND"] = "Agg"

import keras
from keras.src import backend
from utils.lib.utils import log
from utils.lib.config import load_config_from_yaml

from ActiveSampler import get_active_diffusion_sampler_class
from utils.io_utils import PlotActiveInference

if __name__ == "__main__":
    print(f"Using {backend.backend()} backend 🔥")
    config = load_config_from_yaml(args.config)

    keras.utils.set_random_seed(args.seed)

    ActiveSamplerClass = get_active_diffusion_sampler_class(
        config.diffusion_sampler.data_domain
    )

    active_sampler = ActiveSamplerClass(
        image_shape=config.diffusion_sampler.image_shape,
        diffusion_model_run_dir=Path(config.diffusion_sampler.run_dir),
        target_img_path=args.target_img,
        selection_strategy=config.diffusion_sampler.selection_strategy,
        initial_measurement=config.diffusion_sampler.initial_measurement,
        pixel_region_radius=config.diffusion_sampler.get("pixel_region_radius"),
        data_root=args.data_root,
    )

    log.info(
        f"Running active sampling on {log.green(config.diffusion_sampler.data_domain)} in {log.yellow(active_sampler.save_dir)}"
    )

    plot_callback = PlotActiveInference(
        postprocess_func=active_sampler.postprocess,
        target_img=active_sampler.target_img,
        plotting_interval=config.diffusion_sampler.plotting_interval,
    )

    num_samples_to_take = config.diffusion_sampler.num_samples_to_take - int(
        config.diffusion_sampler.initial_measurement
    )

    posterior_shape = [
        config.diffusion_sampler.batch_size,
        *config.diffusion_sampler.image_shape,
    ]
    posterior_samples, measurements = active_sampler.sample_and_reconstruct(
        num_samples_to_take=num_samples_to_take,
        sampling_window=config.diffusion_sampler.sampling_window,
        posterior_shape=posterior_shape,
        num_diffusion_steps=config.diffusion_sampler.num_steps,
        guidance_kwargs=config.diffusion_sampler.guidance_kwargs,
        guidance_method=config.diffusion_sampler.guidance_method,
        verbose=True,
        plot_callback=plot_callback,
        plotting_interval=config.diffusion_sampler.plotting_interval,
    )

    active_sampler.save_result(posterior_samples, measurements)

    plot_callback.create_animation(
        target_img=active_sampler.target_img,
        save_dir=active_sampler.save_dir,
        filename="frames.gif",
        fps=5,
    )
