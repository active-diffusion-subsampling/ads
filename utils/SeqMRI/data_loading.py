"""
Utils for data-preprocessing from https://github.com/tianweiy/SeqMRI
"""

import argparse
import pathlib
import abc
from typing import (
    Any,
    Dict,
    List,
    Tuple,
)

from torch.utils.data import DataLoader
import torch

from utils.SeqMRI.real_knee_data import RealKneeData


def get_args():
    parser = argparse.ArgumentParser(description="MRI Reconstruction Example")
    parser.add_argument(
        "--num-pools", type=int, default=4, help="Number of U-Net pooling layers"
    )
    parser.add_argument(
        "--num-step", type=int, default=2, help="Number of LSTM iterations"
    )
    parser.add_argument(
        "--drop-prob", type=float, default=0.0, help="Dropout probability"
    )
    parser.add_argument(
        "--num-chans", type=int, default=64, help="Number of U-Net channels"
    )

    parser.add_argument("--batch-size", default=4, type=int, help="Mini batch size")
    parser.add_argument(
        "--num-epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--noise-type",
        type=str,
        default="none",
        help="Type of additive noise to measurements",
    )
    parser.add_argument("--noise-level", type=float, default=5e-5, help="Noise level")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--lr-step-size", type=int, default=40, help="Period of learning rate decay"
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.1,
        help="Multiplicative factor of learning rate decay",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Strength of weight decay regularization",
    )
    parser.add_argument(
        "--report-interval", type=int, default=100, help="Period of loss reporting"
    )
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="If set, use multiple GPUs using data parallelism",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Which device to train on. Set to "cuda" to use the GPU',
    )
    parser.add_argument(
        "--exp-dir",
        type=pathlib.Path,
        required=True,
        help="Path where model and results should be saved",
    )
    parser.add_argument(
        "--checkpoint1",
        type=str,
        help='Path to an existing checkpoint. Used along with "--resume"',
    )
    parser.add_argument(
        "--entropy_weight",
        type=float,
        default=0.0,
        help="weight for the entropy/diversity loss",
    )
    parser.add_argument(
        "--recon_weight",
        type=float,
        default=1.0,
        help="weight for the reconsturction loss",
    )
    parser.add_argument(
        "--sparsity_weight",
        type=float,
        default=0.0,
        help="weight for the sparsity loss",
    )
    parser.add_argument(
        "--save-model",
        type=bool,
        default=False,
        help="save model every iteration or not",
    )

    parser.add_argument(
        "--seed", default=42, type=int, help="Seed for random number generators"
    )
    parser.add_argument(
        "--resolution",
        default=[128, 128],
        nargs="+",
        type=int,
        help="Resolution of images",
    )

    # Data parameters
    parser.add_argument(
        "--data-path", type=pathlib.Path, required=False, help="Path to the dataset"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=["fashion-mnist", "dicom-knee", "real-knee", "brain"],
        required=True,
        help="name of the dataset",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Fraction of total volumes to include",
    )

    # Mask parameters
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=float,
        help="Ratio of k-space columns to be sampled. If multiple values are "
        "provided, then one of those is chosen uniformly at random for "
        "each volume.",
    )
    parser.add_argument(
        "--label_range",
        nargs="+",
        type=int,
        help="train using images of specific class",
        default=None,
    )
    parser.add_argument(
        "--model", type=str, help="name of the model to run", required=True
    )
    parser.add_argument(
        "--input_chans",
        type=int,
        choices=[1, 2],
        required=True,
        help="number of input channels. One for real image, 2 for compelx image",
    )
    parser.add_argument(
        "--output_chans",
        type=int,
        default=1,
        help="number of output channels. One for real image",
    )
    parser.add_argument("--line-constrained", type=int, default=0)
    parser.add_argument("--unet", action="store_true")
    parser.add_argument("--preselect", type=int, default=0)
    parser.add_argument(
        "--conjugate_mask",
        action="store_true",
        help="force loupe model to use conjugate symmetry.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["l1", "ssim", "psnr", "xentropy"],
        default="l1",
    )
    parser.add_argument("--test_visual_frequency", type=int, default=1000)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--bi-dir", type=int, default=0)
    parser.add_argument("--preselect_num", type=int, default=2)
    parser.add_argument("--binary_sampler", type=int, default=0)
    parser.add_argument("--clamp", type=float, default=1e10)
    parser.add_argument("--old_recon", type=int, default=0)
    parser.add_argument("--uncertainty_loss", type=int, choices=[0, 1], default=0)
    parser.add_argument("--uncertainty_weight", type=float, default=0)
    parser.add_argument("--detach_kspace", type=int, default=0)
    parser.add_argument("--random_rotate", type=int, default=0)
    parser.add_argument("--kspace_weight", type=float, default=0)
    parser.add_argument("--pretrained_recon", type=int, default=0)

    args = parser.parse_args()

    args.pretrained_recon = args.pretrained_recon > 0
    args.random_rotate = args.random_rotate > 0
    args.line_constrained = args.line_constrained > 0

    if args.detach_kspace == 1:
        args.detach_kspace = True
    else:
        args.detach_kspace = False

    if args.uncertainty_loss == 1:
        args.uncertainty_loss = True
    else:
        args.uncertainty_loss = False

    if args.old_recon > 0:
        args.old_recon = True
    else:
        args.old_recon = False

    if args.checkpoint1 is not None:
        args.resume = True
    else:
        args.resume = False

    noise_str = ""
    if args.noise_type is "none":
        noise_str = "_no_noise_"
    else:
        noise_str = "_" + args.noise_type + str(args.noise_level) + "_"

    if args.preselect > 0:
        args.preselect = True
    else:
        args.preselect = False

    if args.bi_dir > 0:
        args.bi_dir = True
    else:
        args.bi_dir = False

    if args.binary_sampler > 0:
        args.binary_sampler = True
    else:
        args.bianry_sampler = False
    return args


class LOUPEDataEnv:
    def __init__(self, options):
        self._data_location = options.data_path
        self.options = options

    @abc.abstractmethod
    def _create_datasets(self):
        pass

    def _setup_data_handlers(self):
        train_data, val_data, test_data = self._create_datasets()

        display_data = [
            val_data[i] for i in range(0, len(val_data), len(val_data) // 16)
        ]

        train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.options.batch_size,
            # shuffle=True, # For actually training, this should be True
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            dataset=val_data,
            batch_size=self.options.batch_size,
            num_workers=8,
            shuffle=False,
            pin_memory=True,
        )
        test_loader = DataLoader(
            dataset=test_data,
            batch_size=self.options.batch_size,
            num_workers=8,
            shuffle=False,
            pin_memory=True,
        )
        display_loader = DataLoader(
            dataset=display_data,
            batch_size=16,
            num_workers=8,
            shuffle=False,
            pin_memory=True,
        )
        return train_loader, val_loader, test_loader, display_loader


class LOUPERealKspaceEnv(LOUPEDataEnv):
    def __init__(self, options):
        super().__init__(options)

    @staticmethod
    def _void_transform(
        kspace: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
        attrs: List[Dict[str, Any]],
        fname: List[str],
        slice_id: List[int],
    ) -> Tuple:
        return kspace, mask, target, attrs, fname, slice_id

    def _create_datasets(self):
        root_path = pathlib.Path(self._data_location)
        train_path = root_path / "knee_singlecoil_train"
        val_path = root_path / "knee_singlecoil_val"
        test_path = root_path / "knee_singlecoil_val"

        train_data = RealKneeData(
            train_path,
            self.options.resolution,
            LOUPERealKspaceEnv._void_transform,
            self.options.noise_type,
            self.options.noise_level,
            random_rotate=self.options.random_rotate,
        )
        val_data = RealKneeData(
            val_path,
            self.options.resolution,
            LOUPERealKspaceEnv._void_transform,
            self.options.noise_type,
            self.options.noise_level,
            custom_split="val",
            random_rotate=self.options.random_rotate,
        )
        test_data = RealKneeData(
            test_path,
            self.options.resolution,
            LOUPERealKspaceEnv._void_transform,
            self.options.noise_type,
            self.options.noise_level,
            custom_split="test",
            random_rotate=self.options.random_rotate,
        )

        return train_data, val_data, test_data
