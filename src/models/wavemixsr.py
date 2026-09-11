"""WaveMixSR x2 super-resolution backbone with wavelet token mixing (Jeevan et al. 2023)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

_SCALE_ERROR = "Only scale=2 is supported: the current protocol is x2 linear-SIM only."


def _haar_dwt(x: torch.Tensor) -> torch.Tensor:
    """2D Haar forward transform via reshape/stride tricks (no pywt).

    (N, C, H, W) -> (N, C, H//2, W//2, 4); the last dim holds the four sub-bands in
    LL, LH, HL, HH order. Requires even H and W.
    """
    n, c, h, w = x.shape
    x = x.view(n, c, h // 2, 2, w // 2, 2)  # (i, j, k, l) -> pixel (2i + j, 2k + l)
    a = x[:, :, :, 0, :, 0]
    b = x[:, :, :, 0, :, 1]
    c_ = x[:, :, :, 1, :, 0]
    d = x[:, :, :, 1, :, 1]
    ll = (a + b + c_ + d) * 0.5
    lh = (a - b + c_ - d) * 0.5
    hl = (a + b - c_ - d) * 0.5
    hh = (a - b - c_ + d) * 0.5
    return torch.stack((ll, lh, hl, hh), dim=-1)


def _haar_idwt(bands: torch.Tensor) -> torch.Tensor:
    """Inverse of _haar_dwt: (N, C, H//2, W//2, 4) -> (N, C, H, W)."""
    ll, lh, hl, hh = bands.unbind(dim=-1)
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c_ = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    n, c, h2, w2 = ll.shape
    x = torch.stack((a, b, c_, d), dim=-1)  # (N, C, H2, W2, 4)
    x = x.view(n, c, h2, w2, 2, 2)  # last dims: (row, col) offsets
    return x.permute(0, 1, 2, 4, 3, 5).reshape(n, c, h2 * 2, w2 * 2)


class _WaveMixBlock(nn.Module):
    """Wavelet token-mixing block: Haar DWT -> band mixing -> channel MLP -> IDWT.

    The residual branch runs a 2D Haar DWT, stacks the four sub-bands as channels,
    mixes them with a depthwise 3x3 conv (spatial token mixing within each band)
    followed by a 1x1 conv (mixing across sub-bands and channels), applies a channel
    MLP, and reconstructs with the inverse Haar DWT. Odd spatial sizes are padded to
    even before the DWT and cropped back after the IDWT, so arbitrary sizes work.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        band_channels = channels * 4
        self.band_mix = nn.Sequential(
            nn.Conv2d(band_channels, band_channels, 3, padding=1, groups=band_channels),
            nn.Conv2d(band_channels, band_channels, 1),
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(band_channels, band_channels, 1),
            nn.GELU(),
            nn.Conv2d(band_channels, band_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) -> (N, C, H, W)."""
        n, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        h2, w2 = (h + pad_h) // 2, (w + pad_w) // 2
        bands = _haar_dwt(x)  # (N, C, h2, w2, 4)
        y = bands.permute(0, 4, 1, 2, 3).reshape(n, c * 4, h2, w2)
        y = self.mlp(self.band_mix(y))
        y = y.view(n, 4, c, h2, w2).permute(0, 2, 3, 4, 1)
        out = x + _haar_idwt(y)
        if pad_h or pad_w:
            out = out[..., :h, :w]
        return out


class WaveMixSR(nn.Module):
    """WaveMixSR x2 super-resolution network (wavelet token mixing).

    Shallow 3x3 conv -> ``num_blocks`` WaveMix blocks -> 3x3 conv -> pixel-shuffle x2
    upsampling -> 3x3 conv. Each WaveMix block runs a 2D Haar DWT, mixes the four
    sub-bands with learned depthwise/pointwise token-mixing weights plus a channel
    MLP, and reconstructs with the inverse Haar DWT as a residual branch. With
    ``residual_prediction=True`` (default) the network learns the missing high
    frequencies on top of a bicubic upsample of the input: SR = U(LR) + R_theta(LR).
    With ``residual_prediction=False`` it returns the plain network output R_theta(LR).

    Config kwargs (keys match the constructor names for ``build_model``):
        in_channels: input/output channel count (default 1).
        embed_dim: number of feature channels (default 64).
        num_blocks: number of WaveMix blocks in the body (default 8).
        scale: upsampling factor; only 2 is supported.
        residual_prediction: if True (default) predict the residual over a bicubic
            upsample: SR = U(LR) + net(LR); if False return net(LR) directly.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 64,
        num_blocks: int = 8,
        scale: int = 2,
        residual_prediction: bool = True,
    ) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError(f"scale must be 2, got {scale}. {_SCALE_ERROR}")
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.scale = scale
        self.residual_prediction = residual_prediction
        # Channels fed to PixelShuffle(2) must be a multiple of 4; keep embed_dim//4 when possible.
        pre_shuffle = max(4, (embed_dim // 16) * 4)
        self.head = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.body = nn.Sequential(*[_WaveMixBlock(embed_dim) for _ in range(num_blocks)])
        self.body_conv = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, pre_shuffle, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.tail = nn.Conv2d(pre_shuffle // 4, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) float32 in [0, 1] -> (N, C, 2H, 2W)."""
        residual = self.tail(self.upsample(self.body_conv(self.body(self.head(x)))))
        if not self.residual_prediction:
            return residual
        upsampled = F.interpolate(
            x, scale_factor=self.scale, mode="bicubic", align_corners=False, antialias=True
        )
        return upsampled + residual
