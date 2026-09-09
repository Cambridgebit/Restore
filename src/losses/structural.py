"""Structural losses: canonical SSIM implementation and gradient loss.

The SSIM machinery here is the single shared implementation; src/metrics/image.py
imports it for SSIM / MS-SSIM evaluation (one-directional coupling, intentional).
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _gaussian_kernel(window: int, sigma: float) -> Tensor:
    """Return a normalized 1D gaussian kernel of length `window`."""
    coords = torch.arange(window, dtype=torch.float32) - (window - 1) / 2.0
    kernel = torch.exp(-(coords**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _gaussian_kernel_2d(window: int, sigma: float) -> Tensor:
    """Return a normalized 2D gaussian kernel with shape (1, 1, window, window)."""
    kernel_1d = _gaussian_kernel(window, sigma)
    return (kernel_1d[:, None] * kernel_1d[None, :]).view(1, 1, window, window)


def ssim_components(
    pred: Tensor, target: Tensor, window: int = 11, sigma: float = 1.5
) -> tuple[Tensor, Tensor]:
    """Return per-pixel luminance and contrast*structure maps (Wang et al., 2004).

    Standard formulation with data range 1.0: C1 = 0.01^2, C2 = 0.03^2. The gaussian
    window is applied via conv2d with padding=window//2 (odd windows preserve size).
    """
    c1 = 0.01**2
    c2 = 0.03**2
    kernel = _gaussian_kernel_2d(window, sigma)
    pad = window // 2

    mu_p = F.conv2d(pred, kernel, padding=pad)
    mu_t = F.conv2d(target, kernel, padding=pad)
    mu_p_sq = mu_p * mu_p
    mu_t_sq = mu_t * mu_t
    mu_pt = mu_p * mu_t
    var_p = F.conv2d(pred * pred, kernel, padding=pad) - mu_p_sq
    var_t = F.conv2d(target * target, kernel, padding=pad) - mu_t_sq
    cov = F.conv2d(pred * target, kernel, padding=pad) - mu_pt

    l_map = (2.0 * mu_pt + c1) / (mu_p_sq + mu_t_sq + c1)
    cs_map = (2.0 * cov + c2) / (var_p + var_t + c2)
    return l_map, cs_map


def ssim_index(pred: Tensor, target: Tensor, window: int = 11, sigma: float = 1.5) -> Tensor:
    """Return the scalar mean SSIM (batch mean), data range 1.0."""
    l_map, cs_map = ssim_components(pred, target, window=window, sigma=sigma)
    return (l_map * cs_map).mean()


class SSIMLoss(nn.Module):
    """SSIM-based structural loss: 1 - SSIM."""

    def __init__(self, window: int = 11) -> None:
        super().__init__()
        self.window = window

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Return the scalar loss."""
        return 1.0 - ssim_index(pred, target, window=self.window)


class GradientLoss(nn.Module):
    """Gradient L1 loss (AGENT.md §8), computed on central differences:

    L_grad = ||dx(SR) - dx(GT)||_1 + ||dy(SR) - dy(GT)||_1 with
    dx = x[..., 2:] - x[..., :-2] and dy = x[..., 2:, :] - x[..., :-2, :].
    """

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Return the scalar loss."""
        dx_pred = pred[..., 2:] - pred[..., :-2]
        dx_target = target[..., 2:] - target[..., :-2]
        dy_pred = pred[..., 2:, :] - pred[..., :-2, :]
        dy_target = target[..., 2:, :] - target[..., :-2, :]
        return (dx_pred - dx_target).abs().mean() + (dy_pred - dy_target).abs().mean()
