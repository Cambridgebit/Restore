"""Pixel-fidelity losses."""

import torch
from torch import Tensor, nn


class CharbonnierLoss(nn.Module):
    """Charbonnier loss: robust L1 with an eps-stabilized square root.

    Per pixel: sqrt((pred - target)^2 + eps^2), averaged over batch and pixels.
    """

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor, lr: Tensor | None = None) -> Tensor:
        """Return the scalar loss; `lr` is accepted for interface parity and ignored."""
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()
