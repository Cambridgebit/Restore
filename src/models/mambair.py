"""MambaIR x2 super-resolution backbone with 2D selective-scan state spaces (Guo et al., ECCV 2024)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # optional fused kernel; the pure-PyTorch scan below always works
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except Exception:  # pragma: no cover - mamba_ssm is not a dependency
    selective_scan_fn = None

_SCALE_ERROR = "Only scale=2 is supported: the current protocol is x2 linear-SIM only."


class _SS2D(nn.Module):
    """2D selective-scan (SSM) module: depthwise conv, 4-direction scan, per-channel gate.

    Path: linear in-projection -> depthwise conv (``d_conv``) -> SiLU -> selective scan
    over forward/backward rows and columns (``d_state`` is the SSM state dim) -> sum of
    the four directions -> per-channel gate (SiLU) -> linear out-projection. ``expand``
    controls the inner width (``expand * dim``). The scan is a pure-PyTorch sequential
    linear recurrence (O(L) per direction), so no mamba_ssm dependency is required; the
    fused CUDA kernel is used only when importable.
    """

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        d_inner = int(expand * dim)
        self.d_inner = d_inner
        self.dt_rank = max(1, d_inner // 16)
        self.in_proj = nn.Linear(dim, 2 * d_inner, bias=False)
        self.conv2d = nn.Conv2d(
            d_inner, d_inner, groups=d_inner, bias=True, kernel_size=d_conv, padding=d_conv - 1
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        # A_log stores log(-A); A = -exp(A_log) is negative, keeping the recurrence stable.
        a_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a_init))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, W, C) -> (B, H, W, C)."""
        batch, height, width, _ = x.shape
        x, z = self.in_proj(x).chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2)
        x = self.act(self.conv2d(x)[..., :height, :width])
        x = x.permute(0, 2, 3, 1)
        y = self._scan_4dir(x)
        return self.out_proj(y * self.act(z))

    def _scan_4dir(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, W, D) -> (B, H, W, D), summed over the four scan directions."""
        batch, height, width, _ = x.shape
        y_fwd = self._scan(x.reshape(batch * height, width, -1)).reshape(batch, height, width, -1)
        x_bwd = torch.flip(x, dims=[2])
        y_bwd = torch.flip(
            self._scan(x_bwd.reshape(batch * height, width, -1)).reshape(batch, height, width, -1),
            dims=[2],
        )
        x_col = x.permute(0, 2, 1, 3)
        y_col = self._scan(x_col.reshape(batch * width, height, -1)).reshape(batch, width, height, -1)
        y_col = y_col.permute(0, 2, 1, 3)
        x_col_bwd = torch.flip(x, dims=[1]).permute(0, 2, 1, 3)
        y_col_bwd = self._scan(x_col_bwd.reshape(batch * width, height, -1)).reshape(
            batch, width, height, -1
        )
        y_col_bwd = torch.flip(y_col_bwd.permute(0, 2, 1, 3), dims=[1])
        return y_fwd + y_bwd + y_col + y_col_bwd

    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        """Selective scan over the last dim of (B, L, D) -> (B, L, D)."""
        x_proj = self.x_proj(x)
        dt, b, c = torch.split(x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)
        a = -torch.exp(self.A_log)
        if selective_scan_fn is not None:
            return self._scan_fast(x, dt, b, c, a)
        return self._scan_slow(x, dt, b, c, a)

    def _scan_slow(
        self, x: torch.Tensor, dt: torch.Tensor, b: torch.Tensor, c: torch.Tensor, a: torch.Tensor
    ) -> torch.Tensor:
        """Sequential recurrence: h_t = A_bar h_{t-1} + B_bar x_t; y_t = C h_t + D x_t."""
        batch, length, _ = x.shape
        delta = F.softplus(dt)
        a_bar = torch.exp(delta.unsqueeze(-1) * a)
        b_bar = delta.unsqueeze(-1) * b.unsqueeze(2)
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(length):
            h = a_bar[:, t] * h + b_bar[:, t] * x[:, t].unsqueeze(-1)
            outputs.append((h * c[:, t].unsqueeze(1)).sum(-1) + self.D * x[:, t])
        return torch.stack(outputs, dim=1)

    def _scan_fast(
        self, x: torch.Tensor, dt: torch.Tensor, b: torch.Tensor, c: torch.Tensor, a: torch.Tensor
    ) -> torch.Tensor:
        """Fused mamba_ssm CUDA kernel path (used only when mamba_ssm is importable)."""
        y = selective_scan_fn(
            x.transpose(1, 2).contiguous(),
            dt.transpose(1, 2).contiguous(),
            a,
            b.transpose(1, 2).contiguous(),
            c.transpose(1, 2).contiguous(),
            self.D,
            delta_softplus=True,
        )
        return y.transpose(1, 2)


class _GatedMLP(nn.Module):
    """Two-layer MLP with a SiLU-gated hidden expansion (``expand`` ratio)."""

    def __init__(self, dim: int, expand: int) -> None:
        super().__init__()
        hidden = int(expand * dim)
        self.fc1 = nn.Linear(dim, 2 * hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(..., C) -> (..., C)."""
        hidden, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(hidden * self.act(gate))


class _VSSBlock(nn.Module):
    """Visual State Space block: LayerNorm -> SS2D -> LayerNorm -> gated MLP, with residuals."""

    def __init__(self, dim: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ss2d = _SS2D(dim, d_state, d_conv, expand)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _GatedMLP(dim, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, C, H, W)."""
        x = x.permute(0, 2, 3, 1)
        x = x + self.ss2d(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x.permute(0, 3, 1, 2)


class MambaIR(nn.Module):
    """MambaIR x2 super-resolution network (compact, pure PyTorch, no GAN).

    Shallow 3x3 conv embedding -> residual stack of Visual State Space (VSS) blocks
    (LayerNorm -> SS2D -> LayerNorm -> gated MLP, each with a residual) grouped per
    stage -> 3x3 conv -> pixel-shuffle x2 upsampling -> final 3x3 conv. The SS2D runs
    a 2D selective scan in four directions (forward/backward rows and columns) with a
    per-channel gate. The scan is implemented as a pure-PyTorch sequential linear
    recurrence, so no mamba_ssm dependency is required. NOTE: the pure-PyTorch scan is
    O(L) sequential per direction and therefore slower than the fused CUDA kernel; it
    is intended for correctness and portability, not peak throughput.

    Config kwargs (keys match the constructor names for ``build_model``):
        in_channels: input/output channel count (default 1).
        embed_dim: feature width after the shallow conv (default 48).
        depths: per-stage VSS block counts, e.g. (4, 4, 4, 4).
        d_state: SSM state dimension of the selective scan (default 16).
        d_conv: depthwise conv kernel size inside SS2D (default 4).
        expand: inner-width expansion ratio of SS2D and the gated MLP (default 2).
        scale: upsampling factor; only 2 is supported.
        residual_prediction: if True (default) predict the residual over a bicubic
            upsample: SR = U(LR) + net(LR); if False return net(LR) directly.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 48,
        depths: tuple[int, ...] = (4, 4, 4, 4),
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        scale: int = 2,
        residual_prediction: bool = True,
    ) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError(f"scale must be 2, got {scale}. {_SCALE_ERROR}")
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.depths = tuple(depths)
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.scale = scale
        self.residual_prediction = residual_prediction
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(*[_VSSBlock(embed_dim, d_state, d_conv, expand) for _ in range(depth)])
                for depth in self.depths
            ]
        )
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, 4 * embed_dim, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.conv_last = nn.Conv2d(embed_dim, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) float32 in [0, 1] -> (N, C, 2H, 2W)."""
        features = self.conv_first(x)
        for layer in self.layers:
            features = layer(features)
        features = features + self.conv_after_body(features)
        residual = self.conv_last(self.upsample(features))
        if not self.residual_prediction:
            return residual
        upsampled = F.interpolate(
            x, scale_factor=self.scale, mode="bicubic", align_corners=False, antialias=True
        )
        return upsampled + residual
