"""Frequency-domain losses."""

import torch
from torch import Tensor, nn


class FourierLoss(nn.Module):
    """Amplitude-spectrum L1 loss (AGENT.md §8 optional Fourier loss):

    L_FFT = || |F(SR)| - |F(GT)| ||_1 over all rfft2 bins (unnormalized FFT),
    averaged over batch and frequency bins.
    """

    def forward(self, pred: Tensor, target: Tensor, lr: Tensor | None = None) -> Tensor:
        """Return the scalar loss; `lr` is accepted for interface parity and ignored."""
        amp_pred = torch.fft.rfft2(pred).abs()
        amp_target = torch.fft.rfft2(target).abs()
        return (amp_pred - amp_target).abs().mean()


class FocalFrequencyLoss(nn.Module):
    """Focal frequency loss (Jiang et al., 2021) on the complex FFT difference.

    With D = |F(pred) - F(target)|, the focal weight w = D**alpha is normalized to
    mean 1 and the loss is mean(w * D**2). The eps in the normalizer keeps the
    identical-input case at exactly zero instead of 0/0.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: Tensor, target: Tensor, lr: Tensor | None = None) -> Tensor:
        """Return the scalar loss; `lr` is accepted for interface parity and ignored."""
        magnitude = (torch.fft.rfft2(pred) - torch.fft.rfft2(target)).abs()
        weight = magnitude.pow(self.alpha)
        weight = weight / weight.mean().clamp_min(1e-8)
        return (weight * magnitude.pow(2)).mean()
