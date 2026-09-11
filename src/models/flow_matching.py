"""Conditional flow-matching x2 super-resolution with a U-Net velocity field.

The network is a U-Net velocity field v_theta(x_t, t, lr) trained with the
conditional flow-matching objective: x_t = (1-t)*x_0 + t*x_1 interpolates between
noise and the target, and the model regresses the constant velocity x_1 - x_0. At
inference the ODE x_{k+1} = x_k + (1/steps) * v_theta(x_k, t_k, lr) is integrated
with Euler steps. With ``residual_prediction=True`` the target is the residual on
top of a bicubic x2 upsample of the LR input (SR = U(LR) + x_1), matching the
DFCAN convention; with ``residual_prediction=False`` the target is the HR image
itself.

The velocity network is a standard time-conditioned U-Net (encoder/downsampling,
bottleneck, decoder/upsampling with cross-scale skips); the time embedding is
injected additively into every residual block.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_SCALE_ERROR = "Only scale=2 is supported: the current protocol is x2 linear-SIM only."


def _bicubic_up(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Bicubic upsampling shared by forward, sample, and the target helper."""
    return F.interpolate(x, scale_factor=scale, mode="bicubic", align_corners=False, antialias=True)


def _time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal time embedding; scalar or (N,) t in [0, 1] -> (N, dim)."""
    t = t.reshape(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _norm_groups(channels: int) -> int:
    """GroupNorm group count that always divides ``channels``."""
    return math.gcd(8, channels)


class _ResBlock(nn.Module):
    """Time-conditioned residual block: GroupNorm-SiLU-conv (+time) -> GroupNorm-SiLU-conv, residual."""

    def __init__(self, channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_norm_groups(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_proj = nn.Linear(time_embed_dim, channels)
        self.norm2 = nn.GroupNorm(_norm_groups(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W), (B, time_embed_dim) -> (B, C, H, W)."""
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


def _block_stack(channels: int, num_blocks: int, time_embed_dim: int) -> nn.ModuleList:
    """A stack of ``num_blocks`` time-conditioned residual blocks at ``channels``."""
    return nn.ModuleList([_ResBlock(channels, time_embed_dim) for _ in range(num_blocks)])


class ConditionalFlowSR(nn.Module):
    """Conditional flow-matching x2 super-resolution network (U-Net velocity field).

    Config kwargs (all match the constructor):
        in_channels (int): input channels (default 1).
        nf (int): base feature width; level ``i`` uses ``min(nf * 2**i, nf * 8)`` (default 64).
        num_blocks (int): residual blocks per U-Net level (default 2).
        time_embed_dim (int): width of the sinusoidal time embedding / MLP (default 128).
        num_levels (int): number of U-Net levels, i.e. ``num_levels - 1`` downsamples (default 3).
        scale (int): upsampling factor; only 2 is supported (default 2).
        residual_prediction (bool): predict the residual on top of bicubic U(LR)
            (default True), matching the DFCAN convention.

    The velocity field takes the noisy/generated HR tensor x_t, the time t in
    [0, 1], and the LR input; it returns a single-channel velocity prediction of
    the same spatial size as x_t.
    """

    def __init__(
        self,
        in_channels: int = 1,
        nf: int = 64,
        num_blocks: int = 2,
        time_embed_dim: int = 128,
        num_levels: int = 3,
        scale: int = 2,
        residual_prediction: bool = True,
    ) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError(f"scale must be 2, got {scale}. {_SCALE_ERROR}")
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")
        self.in_channels = in_channels
        self.nf = nf
        self.num_blocks = num_blocks
        self.time_embed_dim = time_embed_dim
        self.num_levels = num_levels
        self.scale = scale
        self.residual_prediction = residual_prediction
        channels = [min(nf * (2**level), nf * 8) for level in range(num_levels)]

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.stem = nn.Conv2d(2 * in_channels, channels[0], 3, padding=1)

        # Encoder: residual stack per level, then a stride-2 conv to the next level.
        self.enc_blocks = nn.ModuleList(
            [_block_stack(channels[level], num_blocks, time_embed_dim) for level in range(num_levels)]
        )
        self.downs = nn.ModuleList(
            [
                nn.Conv2d(channels[level], channels[level + 1], 3, stride=2, padding=1)
                for level in range(num_levels - 1)
            ]
        )
        # Bottleneck at the deepest level.
        self.mid_blocks = _block_stack(channels[-1], num_blocks, time_embed_dim)

        # Decoder: upsample -> channel conv -> concat skip -> fuse conv -> residual stack.
        self.ups = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for level in range(num_levels - 2, -1, -1):
            self.ups.append(nn.Conv2d(channels[level + 1], channels[level], 3, padding=1))
            self.fuse.append(nn.Conv2d(2 * channels[level], channels[level], 3, padding=1))
            self.dec_blocks.append(_block_stack(channels[level], num_blocks, time_embed_dim))

        self.tail = nn.Conv2d(channels[0], in_channels, 3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        """(N, C, 2H, 2W), (N,) or scalar t in [0, 1], (N, C, H, W) -> (N, C, 2H, 2W)."""
        cond = _bicubic_up(lr, self.scale)
        t_emb = self.time_mlp(_time_embedding(t, self.time_embed_dim))

        # Pad to a multiple of 2**(num_levels-1) so every downsample halves exactly;
        # odd inputs are cropped back at the end.
        factor = 2 ** (self.num_levels - 1)
        height, width = x_t.shape[-2:]
        pad_h = (-height) % factor
        pad_w = (-width) % factor
        if pad_h or pad_w:
            padding = (0, pad_w, 0, pad_h)
            x_t = F.pad(x_t, padding, mode="replicate")
            cond = F.pad(cond, padding, mode="replicate")

        h = self.stem(torch.cat([x_t, cond], dim=1))
        skips: list[torch.Tensor] = []
        for level in range(self.num_levels):
            for block in self.enc_blocks[level]:
                h = block(h, t_emb)
            skips.append(h)
            if level < self.num_levels - 1:
                h = self.downs[level](h)

        for block in self.mid_blocks:
            h = block(h, t_emb)

        for level in range(self.num_levels - 2, -1, -1):
            index = self.num_levels - 2 - level
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = self.ups[index](h)
            h = torch.cat([h, skips[level]], dim=1)
            h = self.fuse[index](h)
            for block in self.dec_blocks[index]:
                h = block(h, t_emb)

        velocity = self.tail(h)
        if pad_h or pad_w:
            velocity = velocity[..., :height, :width]
        return velocity

    @torch.no_grad()
    def sample(
        self,
        lr: torch.Tensor,
        steps: int = 20,
        sigma: float = 1.0,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Euler integration of the flow ODE; (N, C, H, W) -> (N, C, 2H, 2W).

        ``noise`` is the raw initial x_0 including ``sigma`` (i.e. already scaled);
        when None, ``sigma * randn_like(cond)`` is drawn. With ``residual_prediction``
        the final sample is ``cond + x_steps``, else ``x_steps``.
        """
        cond = _bicubic_up(lr, self.scale)
        if noise is None:
            x = sigma * torch.randn_like(cond)
        else:
            x = noise
        for k in range(steps):
            t = torch.full((x.shape[0],), k / steps, device=x.device, dtype=x.dtype)
            x = x + (1.0 / steps) * self.forward(x, t, lr)
        if self.residual_prediction:
            return cond + x
        return x


def flow_velocity_target(
    hr: torch.Tensor,
    lr: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    scale: int = 2,
    residual_prediction: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the conditional flow-matching training pair (x_t, velocity_target).

    x_t = (1 - t) * noise + t * x1, target = x1 - noise, where x1 = hr - cond with
    cond = bicubic_up(lr) when ``residual_prediction`` else x1 = hr.
    """
    cond = _bicubic_up(lr, scale)
    x1 = hr - cond if residual_prediction else hr
    t = t.reshape(-1, 1, 1, 1)  # (N,) or scalar t broadcasts over the batch dim
    x_t = (1.0 - t) * noise + t * x1
    return x_t, x1 - noise
