"""Consistency losses tying the super-resolved output back to its low-resolution input.

Both components take the optional low-resolution tensor as `lr`; when it is absent
(e.g. during evaluation or an ablation) they return a zero scalar so the weighted
sum stays well-defined.
"""

import torch.nn.functional as F
from torch import Tensor, nn


class DataConsistencyLoss(nn.Module):
    """Downsample consistency: ||down(pred) - lr||_1, averaged.

    ``mode="avgpool"`` downsamples with average pooling by ``scale``;
    ``mode="bicubic_down"`` downsamples with bicubic interpolation. Returns a zero
    scalar when ``lr is None``.
    """

    _MODES = ("avgpool", "bicubic_down")

    def __init__(self, scale: int = 2, mode: str = "avgpool") -> None:
        super().__init__()
        if mode not in self._MODES:
            msg = f"Unknown DataConsistencyLoss mode: {mode!r}; expected one of {self._MODES}"
            raise ValueError(msg)
        self.scale = scale
        self.mode = mode

    def forward(self, pred: Tensor, target: Tensor, lr: Tensor | None = None) -> Tensor:
        """Return the scalar loss; zero when lr is None."""
        if lr is None:
            return pred.new_zeros(())
        if self.mode == "avgpool":
            down = F.avg_pool2d(pred, self.scale)
        else:
            down = F.interpolate(
                pred,
                scale_factor=1.0 / self.scale,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        return (down - lr).abs().mean()


class ResidualLoss(nn.Module):
    """Residual magnitude: ||pred - up(lr)||_1, averaged.

    ``up`` is bicubic upsampling of the low-resolution input by ``scale``. Returns a
    zero scalar when ``lr is None``.
    """

    def __init__(self, scale: int = 2) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, pred: Tensor, target: Tensor, lr: Tensor | None = None) -> Tensor:
        """Return the scalar loss; zero when lr is None."""
        if lr is None:
            return pred.new_zeros(())
        up = F.interpolate(
            lr,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        return (pred - up).abs().mean()
