"""Tiled model inference for images larger than one forward pass."""

from __future__ import annotations

import torch


@torch.no_grad()
def predict_tiled(
    model: torch.nn.Module,
    lr: torch.Tensor,
    tile: int = 256,
    overlap: int = 32,
    scale: int = 2,
) -> torch.Tensor:
    """Run `model` over reflect-padded LR tiles; return (1, 1, scale*H, scale*W).

    lr is (1, 1, H, W) float32 on the model's device. Each `tile`-sized window is
    cut with `overlap` context pixels on every side; the model output for that
    window is trimmed by overlap*scale on interior sides and pasted into the
    canvas, so interior seams carry full context. Residual-prediction semantics
    live inside the model; this helper only tiles the forward pass.
    """
    if tile <= 2 * overlap:
        raise ValueError(f"tile must be > 2*overlap, got tile={tile}, overlap={overlap}")
    _, _, h, w = lr.shape
    stride = tile - 2 * overlap
    n_y = -(-h // stride)
    n_x = -(-w // stride)
    pad_t = overlap
    pad_l = overlap
    pad_b = n_y * stride - h + overlap
    pad_r = n_x * stride - w + overlap
    pads = (pad_l, pad_r, pad_t, pad_b)
    if pad_t < h and pad_b < h and pad_l < w and pad_r < w:
        padded = torch.nn.functional.pad(lr, pads, mode="reflect")
    else:  # image smaller than the requested context; replicate still pads safely
        padded = torch.nn.functional.pad(lr, pads, mode="replicate")
    canvas = torch.empty((*lr.shape[:2], scale * h, scale * w), device=lr.device, dtype=lr.dtype)
    margin = scale * overlap
    for i in range(n_y):
        y0 = i * stride
        y1 = min(h, y0 + stride)
        for j in range(n_x):
            x0 = j * stride
            x1 = min(w, x0 + stride)
            out = model(padded[:, :, y0 : y0 + tile, x0 : x0 + tile])
            canvas[:, :, scale * y0 : scale * y1, scale * x0 : scale * x1] = out[
                :,
                :,
                margin : margin + scale * (y1 - y0),
                margin : margin + scale * (x1 - x0),
            ]
    return canvas
