"""NAFNet-lite x2 super-resolution backbone (Chen et al., ECCV 2022)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

_SCALE_ERROR = "Only scale=2 is supported: the current protocol is x2 linear-SIM only."


class _LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm over an (N, C, H, W) tensor (normalizes the C axis)."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) -> (N, C, H, W)."""
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    """Non-linear activation: split channels in half and multiply elementwise."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 2C, H, W) -> (N, C, H, W)."""
        first, second = x.chunk(2, dim=1)
        return first * second


class _SimplifiedChannelAttention(nn.Module):
    """Global-average-pooled gate: pool -> 1x1 conv -> SiLU -> 1x1 conv -> Sigmoid."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) -> (N, C, 1, 1)."""
        return self.fc(self.pool(x))


class NAFBlock(nn.Module):
    """NAF block: LayerNorm -> 3x3 conv -> SimpleGate -> channel attention -> 3x3 conv.

    The block output is scaled by the pooled channel gate and added to the input.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = _LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, 2 * channels, 3, padding=1)
        self.gate = SimpleGate()
        self.sca = _SimplifiedChannelAttention(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) -> (N, C, H, W)."""
        residual = x
        x = self.norm(x)
        x = self.gate(self.conv1(x))
        x = x * self.sca(x)
        return residual + self.conv2(x)


class NAFNet(nn.Module):
    """NAFNet-lite x2 super-resolution network.

    Shallow 3x3 conv -> stack of NAF blocks -> pixel-shuffle x2 upsampling -> 3x3 conv.
    The block uses SimpleGate (channel-split multiply) in place of ReLU/GELU, which the
    paper shows is enough for image restoration.

    Config kwargs (keys match the constructor names for ``build_model``):
        in_channels: input/output channel count (default 1).
        nf: number of feature channels (default 32).
        num_blocks: number of NAF blocks in the body (default 8).
        scale: upsampling factor; only 2 is supported.
        residual_prediction: if True (default) predict the residual over a bicubic
            upsample: SR = U(LR) + net(LR); if False return net(LR) directly.
    """

    def __init__(
        self,
        in_channels: int = 1,
        nf: int = 32,
        num_blocks: int = 8,
        scale: int = 2,
        residual_prediction: bool = True,
    ) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError(f"scale must be 2, got {scale}. {_SCALE_ERROR}")
        self.in_channels = in_channels
        self.nf = nf
        self.num_blocks = num_blocks
        self.scale = scale
        self.residual_prediction = residual_prediction
        # Channels fed to PixelShuffle(2) must be a multiple of 4; keep nf//4 when possible.
        pre_shuffle = max(4, (nf // 16) * 4)
        self.head = nn.Conv2d(in_channels, nf, 3, padding=1)
        self.body = nn.Sequential(*[NAFBlock(nf) for _ in range(num_blocks)])
        self.upsample = nn.Sequential(
            nn.Conv2d(nf, pre_shuffle, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.tail = nn.Conv2d(pre_shuffle // 4, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) float32 in [0, 1] -> (N, C, 2H, 2W)."""
        residual = self.tail(self.upsample(self.body(self.head(x))))
        if not self.residual_prediction:
            return residual
        upsampled = F.interpolate(
            x, scale_factor=self.scale, mode="bicubic", align_corners=False, antialias=True
        )
        return upsampled + residual
