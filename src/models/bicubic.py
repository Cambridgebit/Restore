"""Bicubic x2 upsampling: the non-learning reference baseline (AGENT.md §7)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BicubicSR(nn.Module):
    """Reference interpolator with the same forward contract as DFCAN/RCAN.

    (N, C, H, W) -> (N, C, scale*H, scale*W) via antialiased bicubic upsampling.
    Not restricted to scale=2: it is the comparison floor for any scale. It has no
    parameters, so an "epochs=0" run evaluates it directly on the held-out fold.
    """

    def __init__(self, in_channels: int = 1, scale: int = 2) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) float32 in [0, 1] -> (N, C, scale*H, scale*W)."""
        return F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False, antialias=True)
