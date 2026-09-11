"""Conditional flow-matching x2 super-resolution model (Lipman et al. 2023).

The network is a velocity field v_theta(x_t, t, lr) trained with the conditional
flow-matching objective: x_t = (1-t)*x_0 + t*x_1 interpolates between noise and
the target, and the model regresses the constant velocity x_1 - x_0. At inference
the ODE x_{k+1} = x_k + (1/steps) * v_theta(x_k, t_k, lr) is integrated with Euler
steps. With ``residual_prediction=True`` the target is the residual on top of a
bicubic x2 upsample of the LR input (SR = U(LR) + x_1), matching the DFCAN
convention; with ``residual_prediction=False`` the target is the HR image itself.
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


class _ResBlock(nn.Module):
    """Conv residual block with additive time conditioning.

    Path: 3x3 conv -> ReLU -> + time projection -> 3x3 conv -> ReLU. Output: x + path(x).
    """

    def __init__(self, nf: int, time_embed_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, padding=1)
        self.time_proj = nn.Linear(time_embed_dim, nf)
        self.conv2 = nn.Conv2d(nf, nf, 3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """(B, nf, H, W), (B, time_embed_dim) -> (B, nf, H, W)."""
        h = F.relu(self.conv1(x))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = F.relu(self.conv2(h))
        return x + h


class ConditionalFlowSR(nn.Module):
    """Conditional flow-matching x2 super-resolution network.

    Config kwargs (all match the constructor):
        in_channels (int): input channels (default 1).
        nf (int): feature width of the conv residual body (default 64).
        num_blocks (int): number of residual blocks in the body (default 8).
        time_embed_dim (int): width of the sinusoidal time embedding / MLP (default 128).
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
        num_blocks: int = 8,
        time_embed_dim: int = 128,
        scale: int = 2,
        residual_prediction: bool = True,
    ) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError(f"scale must be 2, got {scale}. {_SCALE_ERROR}")
        self.in_channels = in_channels
        self.nf = nf
        self.num_blocks = num_blocks
        self.time_embed_dim = time_embed_dim
        self.scale = scale
        self.residual_prediction = residual_prediction
        self.head = nn.Conv2d(2 * in_channels, nf, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.blocks = nn.ModuleList([_ResBlock(nf, time_embed_dim) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(nf, in_channels, 3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        """(N, C, 2H, 2W), (N,) or scalar t in [0, 1], (N, C, H, W) -> (N, C, 2H, 2W)."""
        cond = _bicubic_up(lr, self.scale)
        h = self.head(torch.cat([x_t, cond], dim=1))
        t_emb = self.time_mlp(_time_embedding(t, self.time_embed_dim))
        for block in self.blocks:
            h = block(h, t_emb)
        return self.tail(h)

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
