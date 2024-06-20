# pylint: disable=no-member

# load mnist test set
# load active sampler
# foreach num_samples_to_take in intervals
# foreach strategy
# save MAE per strategy

import os
from pathlib import Path
from distutils.util import strtobool
import argparse
import uuid


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark MNIST")
    parser.add_argument(
        "--sampling_interval",
        type=int,
        default=1,
        help="Only evaluates samples at this interval, e.g. 1 out of every 5 samples, for sampling_interval=5",
    )
    parser.add_argument(
        "--test_set_path",
        type=str,
        default=None,
        help="path to the MNIST test set",
    )
    parser.add_argument(
        "-d",
        "--results_dir",
        type=str,
        default="results",
        help="Path directory in which to save results",
    )
    parser.add_argument(
        "--dataset_variance_map_path",
        type=str,
        default="benchmark/data_assets/data_variance_map.npy",
        help="path to dataset variance map, as created by create_mnist_dataset_variance_map.py",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="How many samples to take",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark/mnist/mnist_pixel_random.yaml",
        help="which config to use",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="jax",
        help="ML backend to use",
        choices=["tensorflow", "torch", "jax"],
    )
    return parser.parse_args()


args = parse_args()
assert os.path.isdir(
    args.results_dir
), f"Directory results_dir={args.results_dir} does not exist."
os.environ["KERAS_BACKEND"] = args.backend
os.environ["MPLBACKEND"] = "Agg"

import os
import fnmatch
import numpy as np
from tqdm import tqdm
from ActiveSampler import ImageActiveSampler
import keras.ops as ops

from utils.lib.config import load_config_from_yaml
from utils.keras_utils import load_img_as_tensor
import utils.lib.utils as lib_utils
from utils.lib import log


def find_png_files(directory):
    png_files = []
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, "*.png"):
            png_files.append(os.path.join(root, filename))
    return png_files


config = load_config_from_yaml(args.config)

directory = args.test_set_path
png_files = find_png_files(directory)
assert len(png_files) > 0, f"0 png files found in {directory}"
print(f"Successfully loaded {len(png_files)} png files from {directory}")

active_sampler = ImageActiveSampler(
    diffusion_model_run_dir=Path(config.diffusion_sampler.run_dir),
    target_img_path="benchmark",
    image_shape=[32, 32, 1],
    selection_strategy=config.diffusion_sampler.selection_strategy,
    hard_consistency=config.diffusion_sampler.hard_consistency,
    initial_measurement=config.diffusion_sampler.initial_measurement,
    sigma=config.diffusion_sampler.sigma,
)

posterior_shape = [
    config.diffusion_sampler.batch_size,
    *config.diffusion_sampler.image_shape,
]

datestring = lib_utils.get_date_string()
job_id = str(uuid.uuid4())[:8]
log.info(f"JOBID={log.green(job_id)}")
outdir = Path(args.results_dir) / f"{datestring}_jobid={job_id}"
os.mkdir(outdir)
log.info(f"Created output directory at {log.yellow(outdir)}")

if args.num_samples is None:
    if active_sampler.selection_strategy.startswith("column"):
        NUM_SAMPLES = [2, 4, 8, 16, 24]
    else:
        NUM_SAMPLES = [10, 25, 50, 100, 250, 500]
else:
    NUM_SAMPLES = [args.num_samples]
for num_samples_to_take in NUM_SAMPLES:
    maes = []
    masks = []
    for i, test_img_path in enumerate(tqdm(png_files)):
        if i % args.sampling_interval != 0:
            continue
        test_img = load_img_as_tensor(
            str(test_img_path),
            image_shape=[32, 32],
            grayscale=True,
        )[None, ...]
        test_img = active_sampler.preprocess(test_img)
        active_sampler.target_img = ops.convert_to_tensor(test_img)
        active_sampler.operator, active_sampler.measurement = (
            active_sampler.initialise_operator()
        )

        sampling_window = config.diffusion_sampler.sampling_window
        num_diffusion_steps = config.diffusion_sampler.num_steps

        if active_sampler.selection_strategy == "pixel_random":
            selected_pixels = np.unravel_index(
                np.random.choice(
                    active_sampler.operator.mask.size,
                    size=num_samples_to_take,
                    replace=False,
                ),
                shape=active_sampler.operator.mask.shape,
            )
            active_sampler.operator.mask = active_sampler.operator.mask.at[
                selected_pixels
            ].set(1)
            sampling_window = [num_diffusion_steps, num_diffusion_steps]
        elif active_sampler.selection_strategy == "pixel_data_variance":
            dataset_variance_map = np.load(args.dataset_variance_map_path)
            selected_pixels = np.unravel_index(
                np.random.choice(
                    active_sampler.operator.mask.size,
                    p=np.ravel(active_sampler.dataset_variance_map),
                    size=num_samples_to_take,
                    replace=False,
                ),
                shape=active_sampler.operator.mask.shape,
            )
            active_sampler.operator.mask = active_sampler.operator.mask.at[
                selected_pixels
            ].set(1)
            sampling_window = [num_diffusion_steps, num_diffusion_steps]
        elif active_sampler.selection_strategy == "column_random":
            selected_columns = np.random.choice(
                active_sampler.operator.mask.shape[-2],
                size=num_samples_to_take,
                replace=False,
            )
            active_sampler.operator.mask = active_sampler.operator.mask.at[
                :, :, selected_columns, :
            ].set(1)
            sampling_window = [num_diffusion_steps, num_diffusion_steps]
        elif active_sampler.selection_strategy == "column_data_variance":
            selected_columns = np.random.choice(
                active_sampler.operator.mask.shape[-2],
                # sum variance across all axes apart from columns to get marginal probability per column
                p=ops.sum(active_sampler.dataset_variance_map, axis=[0, 1, 3]),
                size=num_samples_to_take,
                replace=False,
            )
            active_sampler.operator.mask = active_sampler.operator.mask.at[
                :, :, selected_columns, :
            ].set(1)
            sampling_window = [num_diffusion_steps, num_diffusion_steps]

        assert (
            not active_sampler.selection_strategy == "pixel_entropy"
            or not active_sampler.selection_strategy == "column_entropy"
            or ops.sum(active_sampler.operator.mask) == num_samples_to_take
            or ops.sum(active_sampler.operator.mask)
            / active_sampler.operator.mask.shape[-2]
            == num_samples_to_take
        )

        posterior_samples, measurements = active_sampler.sample_and_reconstruct(
            num_samples_to_take=num_samples_to_take,
            sampling_window=sampling_window,
            posterior_shape=posterior_shape,
            num_diffusion_steps=num_diffusion_steps,
            guidance_kwargs={"omega": config.diffusion_sampler.guidance_kwargs.omega},
            guidance_method=config.diffusion_sampler.guidance_method,
            verbose=True,
            plot_callback=None,  # we don't typically want plots for benchmark
        )
        mae = ops.mean(
            ops.abs(active_sampler.target_img - ops.mean(posterior_samples, axis=0))
        )
        maes.append(mae)
        masks.append(active_sampler.operator.mask)
    np.savez(
        outdir / f"_{active_sampler.selection_strategy}_{num_samples_to_take}",
        maes=np.array(maes),
        masks=np.array(masks),
    )
