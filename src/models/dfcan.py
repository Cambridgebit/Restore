"""DFCAN x2 super-resolution backbone with Fourier channel attention (Qiao et al. 2021)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

_SCALE_ERROR = "Only scale=2 is supported: the current protocol is x2 linear-SIM only."


class FCAB(nn.Module):
    """Fourier Channel Attention block with a local residual connection.

    Path: 3x3 conv -> rFFT2 -> amplitude |spectrum|**gamma -> global max pool over
    frequencies -> per-sample channel normalization (relative energy in [0, 1]) ->
    1x1 MLP gate (sigmoid) -> gated amplitude with preserved phase -> irFFT2 ->
    3x3 conv. Output: x + path(x).

    The pooled-amplitude normalization is essential: raw FFT amplitudes scale with
    the spatial size of the input, which saturates the sigmoid into hard 0/1 gates
    (measured: gate std=0.50, bimodal) and degenerates the attention. Dividing by
    the per-sample channel max keeps the gate input O(1) and scale-invariant, so
    training-patch and inference-tile statistics agree.
    """

    def __init__(self, nf: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        reduced = nf // 4
        self.spatial_conv = nn.Conv2d(nf, nf, 3, padding=1)
        self.gate = nn.Sequential(
            nn.Conv2d(nf, reduced, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, nf, 1),
            nn.Sigmoid(),
        )
        self.out_conv = nn.Conv2d(nf, nf, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, nf, H, W) -> (B, nf, H, W)."""
        height, width = x.shape[-2:]
        feat = self.spatial_conv(x)
        spectrum = torch.fft.rfft2(feat)
        amplitude = spectrum.abs() ** self.gamma
        phase = torch.atan2(spectrum.imag, spectrum.real)
        pooled = amplitude.amax(dim=(-2, -1), keepdim=True)
        pooled = pooled / pooled.amax(dim=1, keepdim=True).clamp_min(1e-12)
        gate = self.gate(pooled)
        feat = torch.fft.irfft2(torch.polar(amplitude * gate, phase), s=(height, width))
        return x + self.out_conv(feat)


class _RIRG(nn.Module):
    """Residual-in-residual group: chained FCABs + conv, with a group-level skip."""

    def __init__(self, nf: int, num_blocks: int, gamma: float) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[FCAB(nf, gamma) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(nf, nf, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, nf, H, W) -> (B, nf, H, W)."""
        return x + self.conv(self.blocks(x))


class DFCAN(nn.Module):
    """DFCAN x2 super-resolution network.

    Shallow conv -> residual-in-residual groups of FCABs -> pixel-shuffle upsampling
    -> final 3x3 conv. With ``residual_prediction=True`` (default) the network learns
    the missing high frequencies on top of a bicubic upsample of the input:
    SR = U(LR) + R_theta(LR), where U is bicubic x2 upsampling and R_theta is the
    network's residual branch. With ``residual_prediction=False`` it returns the
    plain network output R_theta(LR).
    """

    def __init__(
        self,
        in_channels: int = 1,
        nf: int = 64,
        num_groups: int = 4,
        num_blocks: int = 10,
        scale: int = 2,
        gamma: float = 1.0,
        residual_prediction: bool = True,
    ) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError(f"scale must be 2, got {scale}. {_SCALE_ERROR}")
        self.in_channels = in_channels
        self.nf = nf
        self.num_groups = num_groups
        self.num_blocks = num_blocks
        self.gamma = gamma
        self.scale = scale
        self.residual_prediction = residual_prediction
        # Channels fed to PixelShuffle(2) must be a multiple of 4; keep nf//4 when possible.
        pre_shuffle = max(4, (nf // 16) * 4)
        self.head = nn.Conv2d(in_channels, nf, 3, padding=1)
        self.groups = nn.Sequential(*[_RIRG(nf, num_blocks, gamma) for _ in range(num_groups)])
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
