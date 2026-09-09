"""RCAN x2 super-resolution baseline with channel attention (Zhang et al. 2018)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

_SCALE_ERROR = "Only scale=2 is supported: the current protocol is x2 linear-SIM only."


class RCAB(nn.Module):
    """Residual Channel Attention block: conv-ReLU-conv with a local residual.

    Channel attention: global average pool -> 1x1 conv (nf -> nf//reduction) -> ReLU
    -> 1x1 conv -> sigmoid, applied multiplicatively to the residual path.
    """

    def __init__(self, nf: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = nf // reduction
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, padding=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(nf, reduced, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, nf, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, nf, H, W) -> (B, nf, H, W)."""
        residual = self.body(x)
        squeeze = residual.mean(dim=(-2, -1), keepdim=True)
        return x + residual * self.gate(squeeze)


class _RIRG(nn.Module):
    """Residual-in-residual group: chained RCABs + conv, with a group-level skip."""

    def __init__(self, nf: int, num_blocks: int, reduction: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[RCAB(nf, reduction) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(nf, nf, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, nf, H, W) -> (B, nf, H, W)."""
        return x + self.conv(self.blocks(x))


class RCAN(nn.Module):
    """RCAN x2 super-resolution network.

    Shallow conv -> residual-in-residual groups of RCABs -> pixel-shuffle upsampling
    -> final 3x3 conv. With ``residual_prediction=True`` (default) the network learns
    the missing high frequencies on top of a bicubic upsample of the input:
    SR = U(LR) + R_theta(LR). With ``residual_prediction=False`` it returns the
    plain network output R_theta(LR).
    """

    def __init__(
        self,
        in_channels: int = 1,
        nf: int = 64,
        num_groups: int = 10,
        num_blocks: int = 20,
        reduction: int = 16,
        scale: int = 2,
        residual_prediction: bool = True,
    ) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError(f"scale must be 2, got {scale}. {_SCALE_ERROR}")
        self.in_channels = in_channels
        self.nf = nf
        self.num_groups = num_groups
        self.num_blocks = num_blocks
        self.reduction = reduction
        self.scale = scale
        self.residual_prediction = residual_prediction
        # Channels fed to PixelShuffle(2) must be a multiple of 4; keep nf//4 when possible.
        pre_shuffle = max(4, (nf // 16) * 4)
        self.head = nn.Conv2d(in_channels, nf, 3, padding=1)
        self.groups = nn.Sequential(*[_RIRG(nf, num_blocks, reduction) for _ in range(num_groups)])
        self.upsample = nn.Sequential(
            nn.Conv2d(nf, pre_shuffle, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.tail = nn.Conv2d(pre_shuffle // 4, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) float32 in [0, 1] -> (N, C, 2H, 2W)."""
        residual = self.tail(self.upsample(self.groups(self.head(x))))
        if not self.residual_prediction:
            return residual
        upsampled = F.interpolate(
            x, scale_factor=self.scale, mode="bicubic", align_corners=False, antialias=True
        )
        return upsampled + residual
