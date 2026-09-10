"""Structural losses: canonical SSIM implementation and gradient loss.

The SSIM machinery here is the single shared implementation; src/metrics/image.py
imports it for SSIM / MS-SSIM evaluation (one-directional coupling, intentional).
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _gaussian_kernel(window: int, sigma: float, device=None, dtype=torch.float32) -> Tensor:
    """Return a normalized 1D gaussian kernel of length `window` on `device`."""
    coords = torch.arange(window, dtype=dtype, device=device) - (window - 1) / 2.0
    kernel = torch.exp(-(coords**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _gaussian_kernel_2d(window: int, sigma: float, device=None, dtype=torch.float32) -> Tensor:
    """Return a normalized 2D gaussian kernel with shape (1, 1, window, window)."""
    kernel_1d = _gaussian_kernel(window, sigma, device=device, dtype=dtype)
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
    kernel = _gaussian_kernel_2d(window, sigma, device=pred.device, dtype=pred.dtype)
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

    def forward(self, pred: Tensor, target: Tensor, lr: Tensor | None = None) -> Tensor:
        """Return the scalar loss; `lr` is accepted for interface parity and ignored."""
        return 1.0 - ssim_index(pred, target, window=self.window)


class GradientLoss(nn.Module):
    """Gradient L1 loss (AGENT.md §8), computed on central differences:

    L_grad = ||dx(SR) - dx(GT)||_1 + ||dy(SR) - dy(GT)||_1 with
    dx = x[..., 2:] - x[..., :-2] and dy = x[..., 2:, :] - x[..., :-2, :].
    """

    def forward(self, pred: Tensor, target: Tensor, lr: Tensor | None = None) -> Tensor:
        """Return the scalar loss; `lr` is accepted for interface parity and ignored."""
        dx_pred = pred[..., 2:] - pred[..., :-2]
        dx_target = target[..., 2:] - target[..., :-2]
        dy_pred = pred[..., 2:, :] - pred[..., :-2, :]
        dy_target = target[..., 2:, :] - target[..., :-2, :]
        return (dx_pred - dx_target).abs().mean() + (dy_pred - dy_target).abs().mean()


class GradientVarianceLoss(nn.Module):
    """L1 between per-patch variances of the gradient magnitude.

    g = sqrt(dx^2 + dy^2) from forward finite differences; each map is split into
    non-overlapping ``patch_size`` x ``patch_size`` patches, the per-patch variance
    (biased) is computed for pred and target, and the L1 distance between the two
    variance maps is averaged.
    """

    def __init__(self, patch_size: int = 8) -> None:
        super().__init__()
        self.patch_size = patch_size

    def forward(self, pred: Tensor, target: Tensor, lr: Tensor | None = None) -> Tensor:
        """Return the scalar loss; `lr` is accepted for interface parity and ignored."""
        var_pred = self._patch_variances(pred)
        if var_pred.numel() == 0:
            return pred.new_zeros(())
        var_target = self._patch_variances(target)
        return (var_pred - var_target).abs().mean()

    def _patch_variances(self, x: Tensor) -> Tensor:
        """Return per-patch gradient-magnitude variances, shape (B, C, n_patches)."""
        gx = x[..., 1:, :] - x[..., :-1, :]
        gy = x[..., :, 1:] - x[..., :, :-1]
        g = (gx[..., :, :-1] ** 2 + gy[..., :-1, :] ** 2).sqrt()
        p = self.patch_size
        h = (g.shape[-2] // p) * p
        w = (g.shape[-1] // p) * p
        if h == 0 or w == 0:
            return x.new_zeros(0)
        patches = g[..., :h, :w].reshape(*g.shape[:-2], h // p, p, w // p, p)
        patches = patches.permute(*range(g.dim() - 2), -4, -2, -3, -1)
        patches = patches.reshape(*g.shape[:-2], -1, p * p)
        return patches.var(dim=-1, unbiased=False)


_SECOND_DERIVATIVE_KERNELS: dict[str, tuple[tuple[float, ...], ...]] = {
    "xx": ((0.0, 0.0, 0.0), (1.0, -2.0, 1.0), (0.0, 0.0, 0.0)),
    "yy": ((0.0, 1.0, 0.0), (0.0, -2.0, 0.0), (0.0, 1.0, 0.0)),
    "xy": ((1.0, 0.0, -1.0), (0.0, 0.0, 0.0), (-1.0, 0.0, 1.0)),
}


def _second_derivative_kernel(axis: str, sigma: float, x: Tensor) -> Tensor:
    """Return a 3x3 second-derivative kernel scaled by 1/sigma^2 for `axis`."""
    kernel = torch.tensor(_SECOND_DERIVATIVE_KERNELS[axis], dtype=x.dtype, device=x.device)
    scale = 1.0 / (sigma * sigma)
    if axis == "xy":
        scale /= 4.0
    return (kernel * scale).view(1, 1, 3, 3)


class HessianStructureLoss(nn.Module):
    """L1 between Hessian structureness maps at several gaussian scales.

    For each sigma the images are gaussian-smoothed, the second derivatives
    Ixx, Iyy, Ixy are computed with fixed 3x3 kernels scaled by 1/sigma^2, and the
    structureness S = sqrt(Ixx^2 + 2*Ixy^2 + Iyy^2) is formed. Structureness maps
    are averaged over sigmas before the L1 comparison. The eps under the root keeps
    the gradient finite for flat (all-zero) inputs.
    """

    def __init__(self, sigmas: tuple[float, ...] = (1.0, 2.0)) -> None:
        super().__init__()
        self.sigmas = tuple(float(sigma) for sigma in sigmas)

    def forward(self, pred: Tensor, target: Tensor, lr: Tensor | None = None) -> Tensor:
        """Return the scalar loss; `lr` is accepted for interface parity and ignored."""
        return (self._structureness(pred) - self._structureness(target)).abs().mean()

    def _structureness(self, x: Tensor) -> Tensor:
        maps = [self._structureness_at(x, sigma) for sigma in self.sigmas]
        return torch.stack(maps).mean(dim=0)

    def _structureness_at(self, x: Tensor, sigma: float) -> Tensor:
        kernel = _gaussian_kernel_2d(3, sigma, device=x.device, dtype=x.dtype)
        smooth = F.conv2d(x, kernel, padding=1)
        ixx = F.conv2d(smooth, _second_derivative_kernel("xx", sigma, x), padding=1)
        iyy = F.conv2d(smooth, _second_derivative_kernel("yy", sigma, x), padding=1)
        ixy = F.conv2d(smooth, _second_derivative_kernel("xy", sigma, x), padding=1)
        return torch.sqrt(ixx * ixx + 2.0 * ixy * ixy + iyy * iyy + 1e-12)
