"""Frequency-domain losses."""

import torch
from torch import Tensor, nn


class FourierLoss(nn.Module):
    """Amplitude-spectrum L1 loss (AGENT.md §8 optional Fourier loss):

    L_FFT = || |F(SR)| - |F(GT)| ||_1 over all rfft2 bins (unnormalized FFT),
    averaged over batch and frequency bins.
    """

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Return the scalar loss."""
        amp_pred = torch.fft.rfft2(pred).abs()
        amp_target = torch.fft.rfft2(target).abs()
        return (amp_pred - amp_target).abs().mean()
