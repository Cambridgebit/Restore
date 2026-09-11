"""Compact Restormer x2 super-resolution backbone (Zamir et al., CVPR 2022)."""

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


class _MDTA(nn.Module):
    """Multi-Dconv-head transposed attention: channel-wise attention over heads.

    qkv 1x1 conv -> 3x3 depthwise conv -> split into q/k/v heads -> attention
    (Q @ K^T) over the channel axis with a learned per-head temperature -> 1x1 conv.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) -> (N, C, H, W)."""
        n, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        head_dim = c // self.num_heads
        q = q.view(n, self.num_heads, head_dim, h * w)
        k = k.view(n, self.num_heads, head_dim, h * w)
        v = v.view(n, self.num_heads, head_dim, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.view(n, c, h, w)
        return self.project_out(out)


class _GDFN(nn.Module):
    """Gated-Dconv feed-forward network: 1x1 conv -> 3x3 depthwise conv -> gate -> 1x1 conv."""

    def __init__(self, dim: int, expansion_factor: float = 2.66) -> None:
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.project_out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) -> (N, C, H, W)."""
        x = self.project_in(x)
        first, second = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(first) * second)


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer block: MDTA + GDFN, both with residual connections."""

    def __init__(self, dim: int, num_heads: int, expansion_factor: float = 2.66) -> None:
        super().__init__()
        self.norm1 = _LayerNorm2d(dim)
        self.attn = _MDTA(dim, num_heads)
        self.norm2 = _LayerNorm2d(dim)
        self.ffn = _GDFN(dim, expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) -> (N, C, H, W)."""
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class _Downsample(nn.Module):
    """Conv + pixel-unshuffle: (N, C, H, W) -> (N, 2C, H//2, W//2)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) -> (N, 2C, H//2, W//2)."""
        return self.body(x)


class _Upsample(nn.Module):
    """Conv + pixel-shuffle: (N, C, H, W) -> (N, C//2, 2H, 2W)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) -> (N, C//2, 2H, 2W)."""
        return self.body(x)


class Restormer(nn.Module):
    """Compact Restormer x2 super-resolution network.

    Shallow 3x3 conv -> U-shaped encoder/decoder of ``num_levels`` levels, each
    applying ``num_blocks[i]`` Transformer blocks (MDTA + GDFN) with
    pixel-unshuffle/pixel-shuffle down/upsampling between levels and additive skip
    connections -> pixel-shuffle x2 upsampling -> 3x3 conv. Level i operates at
    ``nf * 2**i`` channels with ``num_heads[i]`` attention heads. With
    ``residual_prediction=True`` (default) the network learns the missing high
    frequencies on top of a bicubic upsample of the input: SR = U(LR) + R_theta(LR).
    With ``residual_prediction=False`` it returns the plain network output R_theta(LR).

    Config kwargs (keys match the constructor names for ``build_model``):
        in_channels: input/output channel count (default 1).
        nf: number of feature channels at the first level (default 32).
        num_levels: number of encoder/decoder levels (default 3).
        num_blocks: per-level Transformer block counts, length num_levels (default (2, 2, 2)).
        num_heads: per-level MDTA head counts, length num_levels (default (1, 2, 4)).
        scale: upsampling factor; only 2 is supported.
        residual_prediction: if True (default) predict the residual over a bicubic
            upsample: SR = U(LR) + net(LR); if False return net(LR) directly.
    """

    def __init__(
        self,
        in_channels: int = 1,
        nf: int = 32,
        num_levels: int = 3,
        num_blocks: tuple[int, ...] = (2, 2, 2),
        num_heads: tuple[int, ...] = (1, 2, 4),
        scale: int = 2,
        residual_prediction: bool = True,
    ) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError(f"scale must be 2, got {scale}. {_SCALE_ERROR}")
        self.in_channels = in_channels
        self.nf = nf
        self.num_levels = num_levels
        self.num_blocks = tuple(num_blocks)
        self.num_heads = tuple(num_heads)
        self.scale = scale
        self.residual_prediction = residual_prediction
        self.head = nn.Conv2d(in_channels, nf, 3, padding=1)
        self.enc_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *[_TransformerBlock(nf * 2**i, num_heads[i]) for _ in range(num_blocks[i])]
                )
                for i in range(num_levels)
            ]
        )
        self.downs = nn.ModuleList([_Downsample(nf * 2**i) for i in range(num_levels - 1)])
        self.ups = nn.ModuleList([_Upsample(nf * 2**i) for i in range(num_levels - 1, 0, -1)])
        self.dec_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        _TransformerBlock(nf * 2 ** (i - 1), num_heads[i - 1])
                        for _ in range(num_blocks[i - 1])
                    ]
                )
                for i in range(num_levels - 1, 0, -1)
            ]
        )
        # Channels fed to PixelShuffle(2) must be a multiple of 4; keep nf//4 when possible.
        pre_shuffle = max(4, (nf // 16) * 4)
        self.upsample = nn.Sequential(
            nn.Conv2d(nf, pre_shuffle, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.tail = nn.Conv2d(pre_shuffle // 4, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) float32 in [0, 1] -> (N, C, 2H, 2W)."""
        enc = [self.head(x)]
        for i in range(self.num_levels):
            enc[i] = self.enc_blocks[i](enc[i])
            if i < self.num_levels - 1:
                enc.append(self.downs[i](enc[i]))
        dec = enc[-1]
        for i in range(self.num_levels - 1, 0, -1):
            dec = self.ups[self.num_levels - 1 - i](dec) + enc[i - 1]
            dec = self.dec_blocks[self.num_levels - 1 - i](dec)
        residual = self.tail(self.upsample(dec))
        if not self.residual_prediction:
            return residual
        upsampled = F.interpolate(
            x, scale_factor=self.scale, mode="bicubic", align_corners=False, antialias=True
        )
        return upsampled + residual
