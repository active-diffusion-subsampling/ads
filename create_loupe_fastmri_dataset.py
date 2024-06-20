"""
This script creates the train / val / test split with preprocessing used by:
Yin, T., Wu, Z., Sun, H., Dalca, A. V., Yue, Y., & Bouman, K. L. (2021). End-to-end sequential sampling and reconstruction for MRI. arXiv preprint arXiv:2105.06460.

See: https://github.com/tianweiy/SeqMRI


Usage:
- Specify the root path to your fastMRI data in INPUT_ROOT -- this should be the directory containining:
    - knee_singlecoil_train
    - knee_singlecoil_val
- Specify the desired location for your LOUPE dataset in OUTPUT_ROOT
- Run the script
"""

import numpy as np
from pathlib import Path
import h5py

from utils.SeqMRI.data_loading import get_args, LOUPERealKspaceEnv
from utils.SeqMRI.fastmri import ifft2c


INPUT_ROOT = "/path/to/FastMRI/"
OUTPUT_ROOT = Path("/path/to/FastMRI/LOUPE")

if __name__ == "__main__":

    args = get_args()
    args.data_path = INPUT_ROOT
    args.batch_size = 1
    env = LOUPERealKspaceEnv(args)
    train_loader, val_loader, test_loader, _ = env._setup_data_handlers()
    loaders = {
        "val": val_loader,
        "test": test_loader,
        "train": train_loader,
    }

    for name, loader in loaders.items():
        for iter, data in enumerate(loader):
            kspace, _, _, _, [file_id], slice_id = data
            complex_image = ifft2c(kspace)

            file_dir = OUTPUT_ROOT / name
            file_dir.mkdir(parents=True, exist_ok=True)
            filename = file_dir / (file_id.replace(".h5", f"_{int(slice_id)}.h5"))
            with h5py.File(filename, "w") as hdf5_file:
                hdf5_file.create_dataset("data/kspace", data=kspace.numpy())
                hdf5_file.create_dataset(
                    "data/complex_image", data=complex_image.numpy()
                )
