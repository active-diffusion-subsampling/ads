"""
This script computes the mean ssim for the entire dataset based on
ssims computed for each shard, created by running benchmark_fastmri.py
with num_shards > 0

The script will grab any output directories with the same arrayjobid and
compute the aggregate mean ssim from the ssims in each ssim.txt file
contained within each array job output directory.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from distutils.util import strtobool


def parse_args():
    """Parse arguments for training DDIM."""
    parser = argparse.ArgumentParser(description="SSIM result aggregation")
    parser.add_argument(
        "--jobid",
        type=str,
        default=None,
        help="The jobid for the result shards",
    )
    parser.add_argument(
        "--full_test_test",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Whether the results were computed on the full FastMRI test set",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="The parent directory containing your results directories. This is typically the run_dir of \
            your diffusion sampler in the config you used for benchmarking.",
    )
    args = parser.parse_args()
    if args.jobid is None:
        raise UserWarning("You must specify a jobid")
    return args


def read_file_contents(file_path, encoding="utf-8"):
    try:
        with open(file_path, "r", encoding=encoding) as file:
            contents = file.read()
            return contents
    except FileNotFoundError:
        return f"The file {file_path} does not exist."


args = parse_args()
ROOT_DIR = Path(args.results_dir)
JOB_ID = args.jobid
ssim_file_paths = list(ROOT_DIR.glob(f"*jobid={JOB_ID}*/ssims.txt"))
all_ssims = []
for path in ssim_file_paths:
    ssims_string = read_file_contents(path)
    ssims = [float(ssim) for ssim in ssims_string.split("\n")[:-1]]
    all_ssims.extend(ssims)

if args.full_test_test:
    assert (
        len(all_ssims) == 1851
    ), "There should be 1851 samples in the FastMRI test set."

ssims = np.array(all_ssims) * 100
plt.hist(ssims, bins=200)
mean_ssims = np.mean(ssims)
std_ssims = np.std(ssims)
plt.axvline(
    mean_ssims,
    color="red",
    linestyle="dashed",
    linewidth=1,
    label=f"mean={mean_ssims:.2f}",
)
plt.axvline(
    mean_ssims - std_ssims,
    color="green",
    alpha=0.5,
    linestyle="dashed",
    linewidth=1,
    label=f"std={std_ssims:.2f}",
)
plt.axvline(
    mean_ssims + std_ssims,
    color="green",
    alpha=0.7,
    linestyle="dashed",
    linewidth=1,
)
plt.legend()
plt.xlabel("SSIM")
plt.ylabel("Number of occurrences")
plt.grid(True, alpha=0.3)
datestr = str(datetime.now()).replace(" ", "_")
filename = f"hist_{datestr}_arrayjobid={JOB_ID}"
plt.savefig(ROOT_DIR / (filename + ".png"))
plt.savefig(ROOT_DIR / (filename + ".pdf"))
print(f"Mean ssim={mean_ssims}")
print(f"Saved hist to {ROOT_DIR / filename}.png")
