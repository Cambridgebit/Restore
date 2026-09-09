"""Image-quality metrics: PSNR, SSIM, MS-SSIM, ZNCC, FRC.

All functions accept float32 tensors (N, 1, H, W) with values in [0, 1] and return a
Python float averaged over the batch.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from src.losses.structural import ssim_components, ssim_index

_MS_SSIM_WEIGHTS = (0.0444827, 0.2856310, 0.3000826, 0.2381338, 0.1312362)


def psnr(pred: Tensor, target: Tensor) -> float:
    """Peak signal-to-noise ratio in dB (data range 1).

    The MSE is floored at 1e-12, so identical inputs yield a large finite value
    instead of inf.
    """
    mse = max(torch.mean((pred - target) ** 2).item(), 1e-12)
    return -10.0 * math.log10(mse)


def ssim(pred: Tensor, target: Tensor, window: int = 11) -> float:
    """Mean SSIM (batch mean); uses the shared implementation in src.losses.structural."""
    return ssim_index(pred, target, window=window).item()


def ms_ssim(pred: Tensor, target: Tensor, levels: int = 5, window: int = 11) -> float:
    """Multi-scale SSIM (Wang et al., 2003) with the standard weights normalized to
    sum 1: contrast*structure at every scale, luminance at the finest (last) scale."""
    height, width = pred.shape[-2:]
    min_size = window * 2 ** (levels - 1)
    if height < min_size or width < min_size:
        msg = f"ms_ssim requires H,W >= window * 2**(levels-1) = {min_size}; got {height}x{width}"
        raise ValueError(msg)
    weights = [w / sum(_MS_SSIM_WEIGHTS[:levels]) for w in _MS_SSIM_WEIGHTS[:levels]]

    x, y = pred, target
    cs_means: list[Tensor] = []
    for level in range(levels):
        l_map, cs_map = ssim_components(x, y, window=window)
        if level < levels - 1:
            cs_means.append(cs_map.mean())
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
    # Clamp contrast*structure at 0 before fractional powers to avoid NaN.
    result = (l_map.mean() * cs_map.mean().clamp(min=0.0)) ** weights[-1]
    for cs, weight in zip(cs_means, weights[:-1], strict=True):
        result = result * cs.clamp(min=0.0) ** weight
    return result.item()


def zncc(pred: Tensor, target: Tensor) -> float:
    """Zero-normalized cross-correlation, computed per image then batch-averaged."""
    a = pred.reshape(pred.shape[0], -1)
    b = target.reshape(target.shape[0], -1)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    eps = 1e-12
    num = (a * b).sum(dim=1)
    den = torch.sqrt((a * a).sum(dim=1) * (b * b).sum(dim=1) + eps)
    return (num / den).mean().item()


def frc_resolution(pred: Tensor, target: Tensor, threshold: float = 1 / 7) -> float:
    """Fourier Ring Correlation effective-resolution fraction.

    Bins the rfft2 spectra of both images into integer radial-frequency rings
    (r = round(sqrt(fy^2 + fx^2)) over bin indices), averages the per-image FRC curve
    over the batch, and returns (r_found - 1) / max_r where r_found is the smallest
    ring whose FRC drops below `threshold`; 1.0 if FRC never drops below it. max_r is
    the largest ring index containing at least one frequency bin.
    """
    batch = pred.shape[0]
    height, width = pred.shape[-2:]
    spec_pred = torch.fft.rfft2(pred)
    spec_target = torch.fft.rfft2(target)

    fy = torch.fft.fftfreq(height) * height
    fx = torch.arange(width // 2 + 1, dtype=torch.float32)
    ring = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2).round().to(torch.int64).flatten()
    max_r = int(ring.max())

    cross = (spec_pred * spec_target.conj()).real.reshape(batch, -1)
    power_pred = spec_pred.abs().square().reshape(batch, -1)
    power_target = spec_target.abs().square().reshape(batch, -1)

    num_rings = max_r + 1
    num = torch.zeros(batch, num_rings)
    den_pred = torch.zeros(batch, num_rings)
    den_target = torch.zeros(batch, num_rings)
    num.index_add_(1, ring, cross)
    den_pred.index_add_(1, ring, power_pred)
    den_target.index_add_(1, ring, power_target)
    counts = torch.bincount(ring, minlength=num_rings)

    frc = num / torch.sqrt(den_pred * den_target + 1e-12)
    curve = frc.mean(dim=0)
    below = (counts > 0) & (curve < threshold)
    hits = torch.nonzero(below, as_tuple=False)
    if hits.numel() == 0:
        return 1.0
    return (int(hits[0]) - 1) / max_r
