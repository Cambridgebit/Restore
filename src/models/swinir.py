"""SwinIR-light x2 super-resolution backbone (Liang et al., ICCV 2021), plain L1, no GAN."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_SCALE_ERROR = "Only scale=2 is supported: the current protocol is x2 linear-SIM only."


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (B * nW, window_size, window_size, C)."""
    batch, height, width, channels = x.shape
    x = x.view(batch, height // window_size, window_size, width // window_size, window_size, channels)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, channels)


def _window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int) -> torch.Tensor:
    """(B * nW, window_size, window_size, C) -> (B, H, W, C)."""
    batch = int(windows.shape[0] // (height * width / window_size / window_size))
    x = windows.view(batch, height // window_size, width // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch, height, width, -1)


class _Mlp(nn.Module):
    """Two-layer MLP with GELU used inside each Swin block."""

    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(..., C) -> (..., C)."""
        return self.fc2(self.act(self.fc1(x)))


class _WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias and optional mask."""

    def __init__(self, dim: int, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """(num_windows * B, N, C) -> (num_windows * B, N, C)."""
        batch, tokens, channels = x.shape
        head_dim = channels // self.num_heads
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn = (query * self.scale) @ key.transpose(-2, -1)
        index = self.relative_position_index.view(-1)
        bias = self.relative_position_bias_table[index]
        bias = bias.view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(batch // num_windows, num_windows, self.num_heads, tokens, tokens)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, tokens, tokens)
        attn = self.softmax(attn)
        x = (attn @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(x)


class _SwinTransformerBlock(nn.Module):
    """Swin block: (shifted) window attention + LayerNorm + MLP, both with residuals."""

    def __init__(
        self, dim: int, num_heads: int, window_size: int, shift_size: int, mlp_ratio: float
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """(B, C, H, W) -> (B, C, H, W)."""
        _, _, height, width = x.shape
        x = x.permute(0, 2, 3, 1)
        shortcut = x
        x = self.norm1(x)
        pad_r = (self.window_size - width % self.window_size) % self.window_size
        pad_b = (self.window_size - height % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, padded_h, padded_w, _ = x.shape
        if self.shift_size > 0:
            shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask
        else:
            shifted = x
            attn_mask = None
        windows = _window_partition(shifted, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, self.dim)
        windows = self.attn(windows, mask=attn_mask)
        windows = windows.view(-1, self.window_size, self.window_size, self.dim)
        shifted = _window_reverse(windows, self.window_size, padded_h, padded_w)
        if self.shift_size > 0:
            x = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted
        if pad_r > 0 or pad_b > 0:
            x = x[:, :height, :width, :].contiguous()
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x.permute(0, 3, 1, 2)


class _BasicLayer(nn.Module):
    """Stack of Swin blocks alternating plain and shifted windows, with attention masks."""

    def __init__(
        self, dim: int, depth: int, num_heads: int, window_size: int, mlp_ratio: float
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.blocks = nn.ModuleList(
            [
                _SwinTransformerBlock(
                    dim,
                    num_heads,
                    window_size,
                    shift_size=0 if index % 2 == 0 else self.shift_size,
                    mlp_ratio=mlp_ratio,
                )
                for index in range(depth)
            ]
        )

    def _attention_mask(self, height: int, width: int) -> torch.Tensor | None:
        if self.shift_size == 0:
            return None
        mask = torch.zeros((1, height, width, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                mask[:, h_slice, w_slice, :] = count
                count += 1
        mask_windows = _window_partition(mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, C, H, W)."""
        _, _, height, width = x.shape
        padded_h = math.ceil(height / self.window_size) * self.window_size
        padded_w = math.ceil(width / self.window_size) * self.window_size
        mask = self._attention_mask(padded_h, padded_w)
        for block in self.blocks:
            x = block(x, mask)
        return x


class SwinIR(nn.Module):
    """SwinIR-light x2 super-resolution network (L1 training, no pretrained weights, no GAN).

    Shallow conv patch embedding -> residual stack of Swin Transformer blocks (window
    multi-head self-attention with shifted windows and an attention mask) -> a 3x3 conv
    before the body residual -> pixel-shuffle x2 upsampling -> final 3x3 conv. Inputs
    smaller than the window (or not aligned to it) are padded inside each block and
    cropped back, so arbitrary spatial sizes work.

    Config kwargs (keys match the constructor names for ``build_model``):
        in_channels: input/output channel count (default 1).
        embed_dim: transformer embedding width (default 60).
        depths: per-layer block counts, e.g. (6, 6, 6, 6).
        num_heads: per-layer attention head counts, e.g. (6, 6, 6, 6).
        window_size: local attention window side (default 8).
        mlp_ratio: hidden/embed ratio of the block MLP (default 2.0).
        scale: upsampling factor; only 2 is supported.
        residual_prediction: if True (default) predict the residual over a bicubic
            upsample: SR = U(LR) + net(LR); if False return net(LR) directly.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 60,
        depths: tuple[int, ...] = (6, 6, 6, 6),
        num_heads: tuple[int, ...] = (6, 6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        scale: int = 2,
        residual_prediction: bool = True,
    ) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError(f"scale must be 2, got {scale}. {_SCALE_ERROR}")
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.depths = tuple(depths)
        self.num_heads = tuple(num_heads)
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.scale = scale
        self.residual_prediction = residual_prediction
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.layers = nn.ModuleList(
            [
                _BasicLayer(embed_dim, depth, num_heads[index], window_size, mlp_ratio)
                for index, depth in enumerate(self.depths)
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
