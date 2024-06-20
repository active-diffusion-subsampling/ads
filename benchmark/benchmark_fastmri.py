import argparse
import os
import shutil
import uuid
from pathlib import Path
from distutils.util import strtobool


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark fastMRI")
    parser.add_argument(
        "--use_test_set",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether or not to use the test set",
    )
    parser.add_argument(
        "--test_set_path",
        type=str,
        default=None,
        help="path to the fastMRI test set",
    )
    parser.add_argument(
        "--sampling_interval",
        type=int,
        default=1,
        help="Only evaluates samples at this interval, e.g. 1 out of every 5 samples, for sampling_interval=5",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="active_sampling/configs/ads/fastmri.yaml",
        help="which config to use",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="jax",
        help="ML backend to use",
        choices=["tensorflow", "torch", "jax"],
    )
    parser.add_argument(
        "--save_output",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to save the output",
    )
    parser.add_argument(
        "-d",
        "--data_root",
        type=str,
        default="data",
        help="Path to the your data directory.",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default="results/fastMRI",
        help="Path directory in which to save results",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    return args


args = parse_args()

os.environ["KERAS_BACKEND"] = args.backend
os.environ["MPLBACKEND"] = "Agg"

import keras.ops as ops
import numpy as np
import utils.lib.utils as lib_utils

# from io_utils import PlotActiveInference
from keras.src import backend
from tqdm import tqdm
from utils.lib import log
from utils.lib.config import load_config_from_yaml
from benchmark.results import complex_abs, compute_ssim_torch
import torch

from ActiveSampler import ImageActiveSampler, KSpaceActiveSampler
from datasets import get_datasets


def get_active_diffusion_sampler_class(data_domain):
    if data_domain == "image":
        return ImageActiveSampler
    elif data_domain == "kspace":
        return KSpaceActiveSampler
    else:
        raise ValueError(f"data domain `{data_domain}` was not recognised.")


if __name__ == "__main__":
    print(f"Using {backend.backend()} backend 🔥")
    config = load_config_from_yaml(args.config)

    ActiveSamplerClass = get_active_diffusion_sampler_class(
        config.diffusion_sampler.data_domain
    )

    active_sampler = ActiveSamplerClass(
        image_shape=config.diffusion_sampler.image_shape,
        diffusion_model_run_dir=Path(config.diffusion_sampler.run_dir),
        target_img_path="benchmark",
        selection_strategy=config.diffusion_sampler.selection_strategy,
        hard_consistency=config.diffusion_sampler.hard_consistency,
        initial_measurement=config.diffusion_sampler.initial_measurement,
        sigma=config.diffusion_sampler.sigma,
    )

    log.info(f"Running active sampling in {log.yellow(active_sampler.save_dir)}")

    posterior_shape = [
        config.diffusion_sampler.batch_size,
        *config.diffusion_sampler.image_shape,
    ]

    # Choose number of target images = 1 here
    active_sampler.diffusion_config.data.shuffle = False
    active_sampler.diffusion_config.data.batch_size = 1
    active_sampler.diffusion_config.data.dataset_repetitions = 1
    if args.use_test_set:
        assert (
            args.test_set_path is not None
        ), "You must specify the test set dir in order to benchmark on the test set."
        active_sampler.diffusion_config.data.val_folder = args.test_set_path
    _, val_dataset = get_datasets(
        args.data_root,
        config=active_sampler.diffusion_config,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )

    split = "test" if args.use_test_set else "validation"
    datestring = lib_utils.get_date_string()
    job_id = str(uuid.uuid4())[:8]
    log.info(f"JOBID={log.green(job_id)}")
    outdir = (
        Path(args.out_dir)
        / f"benchmark_active_sampling_{split}_dataset_{datestring}_jobid={job_id}"
    )
    outdir.mkdir(parents=True)

    shutil.copy(
        args.config,
        outdir / "config.yaml",
    )

    num_samples_to_take = config.diffusion_sampler.num_samples_to_take - int(
        config.diffusion_sampler.initial_measurement
    )
    sampling_window = config.diffusion_sampler.sampling_window
    num_diffusion_steps = config.diffusion_sampler.num_steps

    for i, target_img in enumerate(tqdm(val_dataset)):
        if i % args.sampling_interval != 0:
            continue
        active_sampler.target_img = ops.convert_to_tensor(target_img)
        active_sampler.operator, active_sampler.measurement = (
            active_sampler.initialise_operator()
        )

        if active_sampler.selection_strategy == "column_random":
            selected_columns = np.random.choice(
                active_sampler.operator.mask.shape[-2],
                size=num_samples_to_take,
                replace=False,
            )
            active_sampler.operator.mask = active_sampler.operator.mask.at[
                :, :, selected_columns, :
            ].set(1)
            sampling_window = [
                num_diffusion_steps,
                num_diffusion_steps,
            ]
        elif active_sampler.selection_strategy == "column_equispaced":
            selected_columns = np.arange(0, 128, 4)  # 4x acceleration
            active_sampler.operator.mask = active_sampler.operator.mask.at[
                :, :, selected_columns, :
            ].set(1)
            sampling_window = [
                num_diffusion_steps,
                num_diffusion_steps,
            ]
        elif active_sampler.selection_strategy == "column_data_variance":
            try:
                train_data_variance = np.load(
                    "benchmarking/data_assets/fastmri_train_set_variance.npy"
                )
            except Exception as e:
                raise UserWarning(
                    "You must create a data variance map for fastMRI  \
                    using the `benchmarking/scripts/create_fastmri_data_variance_map.py` script \
                    in order to use the column_data_variance strategy."
                )
            variance_per_column = np.sum(
                train_data_variance, axis=(0, 2)
            )  # sum out height and channels
            selected_columns = np.random.choice(
                active_sampler.operator.mask.shape[
                    -2
                ],  # total number of columns = width
                p=(
                    variance_per_column / np.sum(variance_per_column)
                ),  # normalize to make valid probabilities,
                size=num_samples_to_take,
                replace=False,
            )
            assert len(np.unique(selected_columns)) == len(
                selected_columns
            ), "Sampling should be without replacement"
            active_sampler.operator.mask = active_sampler.operator.mask.at[
                :, :, selected_columns, :
            ].set(1)
            sampling_window = [
                num_diffusion_steps,
                num_diffusion_steps,
            ]
        elif active_sampler.selection_strategy == "column_data_variance_topk":
            selected_columns = np.argsort(variance_per_column)[-32:]
            active_sampler.operator.mask = active_sampler.operator.mask.at[
                :, :, selected_columns, :
            ].set(1)
            sampling_window = [
                num_diffusion_steps,
                num_diffusion_steps,
            ]
        elif active_sampler.selection_strategy == "fastmri_baseline":
            # 8% of the measurements = ~10 lines,
            # see https://arxiv.org/pdf/1811.08839 pg 12
            ncols = active_sampler.operator.mask.shape[-2]
            central_mask_indices = np.arange((ncols // 2) - 5, (ncols // 2) + 5)
            num_remaining_samples = num_samples_to_take - len(central_mask_indices)
            # Make a uniform distribution over the remaining sampling positions
            can_be_sampled = np.ones(shape=(ncols))
            can_be_sampled[central_mask_indices] = 0
            sampling_probabilities = can_be_sampled / np.sum(can_be_sampled)
            selected_columns = np.random.choice(
                active_sampler.operator.mask.shape[-2],
                size=num_remaining_samples,
                p=sampling_probabilities,
                replace=False,
            )
            selected_columns = np.append(selected_columns, central_mask_indices)
            active_sampler.operator.mask = active_sampler.operator.mask.at[
                :, :, selected_columns, :
            ].set(1)
            sampling_window = [
                num_diffusion_steps,
                num_diffusion_steps,
            ]
            assert len(selected_columns) == num_samples_to_take

        assert (
            num_samples_to_take + int(config.diffusion_sampler.initial_measurement)
            == 32
        ), "FastMRI should be subsampled for 4x acceleration"

        posterior_samples, measurements = active_sampler.sample_and_reconstruct(
            num_samples_to_take=num_samples_to_take,
            sampling_window=sampling_window,
            posterior_shape=posterior_shape,
            num_diffusion_steps=num_diffusion_steps,
            guidance_kwargs={"omega": config.diffusion_sampler.guidance_kwargs.omega},
            guidance_method=config.diffusion_sampler.guidance_method,
            verbose=True,
            plot_callback=None,
        )
        pred_mean_abs = torch.mean(
            complex_abs(torch.Tensor(ops.convert_to_numpy(posterior_samples))), dim=0
        )
        target_abs = complex_abs(torch.Tensor(ops.convert_to_numpy(target_img)))

        mean_ssim = float(
            compute_ssim_torch(pred_mean_abs[None, None, ...], target_abs[None, ...])
        )
        with open(outdir / "ssims.txt", "a", encoding="utf-8") as file:
            file.write(str(mean_ssim) + "\n")
        if args.save_output:
            np.savez(
                outdir / f"sample_{i}",
                pred=ops.convert_to_numpy(posterior_samples),
                target=ops.convert_to_numpy(target_img),
                mask=ops.convert_to_numpy(active_sampler.operator.mask),
            )
