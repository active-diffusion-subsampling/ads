import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data**2).sum(dim=-1).sqrt()


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        assert isinstance(self.w, torch.Tensor)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        dims = tuple(range(1, len(X.shape)))

        return S.mean(dim=dims)


SSIM = SSIMLoss()


def compute_ssim_torch(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    global SSIM
    SSIM = SSIM.to(xs.device)
    data_range = [y.max() for y in ys]
    data_range = torch.stack(data_range, dim=0)

    return SSIM(xs, ys, data_range=data_range.detach())


def loop(sample_paths):
    ssims = []
    ssim_to_sample = {}

    for sample_path in tqdm(sample_paths):
        if not os.path.isfile(sample_path) and not sample_path.endswith(".npz"):
            continue

        sample = np.load(sample_path)
        pred_mean_abs = torch.mean(complex_abs(torch.Tensor(sample["pred"])), dim=0)
        target_abs = complex_abs(torch.Tensor(sample["target"]))

        mean_ssim = float(
            compute_ssim_torch(pred_mean_abs[None, None, ...], target_abs[None, ...])
        )
        ssims.append(mean_ssim)
        ssim_to_sample[mean_ssim] = {
            "pred": pred_mean_abs,
            "target": target_abs,
            "mask": sample["mask"],
        }
    return ssims, ssim_to_sample


def post(ssims, ssim_to_sample, prepend_path=""):
    ssims = np.array(ssims) * 100
    sorted_keys = sorted(ssim_to_sample.keys())
    plt.hist(ssims, bins=200)
    mean_ssims = np.mean(ssims)
    plt.axvline(
        mean_ssims,
        color="red",
        linestyle="dashed",
        linewidth=1,
        label=f"mean={mean_ssims:.2f}",
    )
    plt.legend()
    plt.xlabel("SSIM")
    plt.ylabel("Number of occurrences")
    plt.grid(True, alpha=0.3)
    filename = f"hist_{datetime.now()}"
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".pdf")
    print(f"Saved hist to {filename}")

    # mask distribution
    bad_masks = [ssim_to_sample[key]["mask"] for key in sorted_keys[:100]]
    plt.imsave(prepend_path + "bad_mask_dist.png", np.sum(bad_masks, axis=0)[0, ..., 0])
    plt.imsave(prepend_path + "bad_mask_dist.pdf", np.sum(bad_masks, axis=0)[0, ..., 0])
    okay_masks = [ssim_to_sample[key]["mask"] for key in sorted_keys[100:300]]
    plt.imsave(
        prepend_path + "okay_mask_dist.png", np.sum(okay_masks, axis=0)[0, ..., 0]
    )
    plt.imsave(
        prepend_path + "okay_mask_dist.pdf", np.sum(okay_masks, axis=0)[0, ..., 0]
    )
    good_masks = [ssim_to_sample[key]["mask"] for key in sorted_keys[300:]]
    plt.imsave(
        prepend_path + "good_mask_dist.png", np.sum(good_masks, axis=0)[0, ..., 0]
    )
    plt.imsave(
        prepend_path + "good_mask_dist.pdf", np.sum(good_masks, axis=0)[0, ..., 0]
    )

    print(f"Mean: {mean_ssims}, Var: {np.var(ssims)}")
