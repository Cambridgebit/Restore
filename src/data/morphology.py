"""Synthetic morphology-OOD paired images for the reserved augmentation ablation.

Renders label-free ``(lr, hr)`` pairs from procedurally drawn microscopy-like
morphologies (filaments, arcs, rings, blobs, Voronoi borders, branching trees)
through a plausible forward model: block-mean downsampling + a mild separable
Gaussian blur + optional Gaussian noise. Everything is numpy + torch only
(no scipy/skimage/cv2), and nothing is ever read from or written to disk.

This module is the data source for the ``augmentation.morphology_ood`` ablation
(AGENT.md §9); the structure label never reaches the model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["FAMILIES", "SyntheticMorphologyDataset", "make_synthetic_pair"]

#: Morphology families the dataset can draw from (the dataset default order).
FAMILIES = ("filament", "curve", "ring", "dot", "voronoi", "fractal")

_EPS = 1e-6


# --- geometry helpers --------------------------------------------------------


def _grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalized ``[-1, 1]`` coordinate grids ``(X, Y)`` of shape ``(size, size)``."""
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    return np.meshgrid(axis, axis)


def _normalize(image: np.ndarray) -> np.ndarray:
    """Min-max normalize to ``[0, 1]`` float32 (all-flat input -> zeros)."""
    low, high = float(image.min()), float(image.max())
    if high - low < 1e-12:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - low) / (high - low)).astype(np.float32)


def _segment_distance(X: np.ndarray, Y: np.ndarray, start, end) -> np.ndarray:
    """Distance from every grid point to the segment ``start``-``end`` (vectorized)."""
    abx, aby = float(end[0] - start[0]), float(end[1] - start[1])
    denom = abx * abx + aby * aby + 1e-12
    t = np.clip(((X - start[0]) * abx + (Y - start[1]) * aby) / denom, 0.0, 1.0)
    return np.hypot(X - (start[0] + t * abx), Y - (start[1] + t * aby))


def _draw_segments(image: np.ndarray, segments: list, width: float) -> None:
    """Composite anti-aliased ``max(0, 1 - dist/width)`` strokes onto ``image`` in place."""
    X, Y = _grid(image.shape[0])
    distance = np.full(image.shape, np.inf, dtype=np.float32)
    for start, end in segments:
        np.minimum(distance, _segment_distance(X, Y, start, end), out=distance)
    np.maximum(image, np.clip(1.0 - distance / width, 0.0, 1.0), out=image)


def _draw_polyline(image: np.ndarray, points: np.ndarray, width: float) -> None:
    """Stroke a dense polyline as connected anti-aliased segments."""
    _draw_segments(image, list(zip(points[:-1], points[1:], strict=True)), width)


def _bezier_points(control: np.ndarray, count: int = 96) -> np.ndarray:
    """Sample a quadratic Bezier ``(3, 2)`` control polygon into ``(count, 2)`` points."""
    t = np.linspace(0.0, 1.0, count)[:, None]
    p0, p1, p2 = control
    return ((1.0 - t) ** 2 * p0 + 2.0 * (1.0 - t) * t * p1 + t**2 * p2).astype(np.float32)


# --- morphology families -----------------------------------------------------


def _gen_filament(rng: np.random.Generator, size: int) -> np.ndarray:
    """One or more smooth curved thin lines (quadratic Bezier strokes)."""
    image = np.zeros((size, size), dtype=np.float32)
    for _ in range(int(rng.integers(1, 4))):
        control = rng.uniform(-0.9, 0.9, (3, 2))
        _draw_polyline(image, _bezier_points(control), float(rng.uniform(0.02, 0.05)))
    return _normalize(image)


def _gen_curve(rng: np.random.Generator, size: int) -> np.ndarray:
    """One or two broader random arcs."""
    image = np.zeros((size, size), dtype=np.float32)
    for _ in range(int(rng.integers(1, 3))):
        cx, cy = rng.uniform(-0.6, 0.6, 2)
        radius = float(rng.uniform(0.25, 0.9))
        start = float(rng.uniform(0.0, 2.0 * np.pi))
        angles = np.linspace(start, start + float(rng.uniform(np.pi / 3, 1.8 * np.pi)), 80)
        points = np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1)
        _draw_polyline(image, points.astype(np.float32), float(rng.uniform(0.04, 0.1)))
    return _normalize(image)


def _gen_ring(rng: np.random.Generator, size: int) -> np.ndarray:
    """One or more annuli of random radius and width."""
    X, Y = _grid(size)
    image = np.zeros((size, size), dtype=np.float32)
    for _ in range(int(rng.integers(1, 4))):
        cx, cy = rng.uniform(-0.35, 0.35, 2)
        radius = float(rng.uniform(0.15, 0.8))
        width = float(rng.uniform(0.02, 0.08))
        distance = np.hypot(X - cx, Y - cy)
        image += np.clip(1.0 - np.abs(distance - radius) / width, 0.0, 1.0)
    return _normalize(image)


def _gen_dot(rng: np.random.Generator, size: int) -> np.ndarray:
    """Several random Gaussian blobs."""
    X, Y = _grid(size)
    image = np.zeros((size, size), dtype=np.float32)
    for _ in range(int(rng.integers(3, 9))):
        cx, cy = rng.uniform(-0.9, 0.9, 2)
        sigma = float(rng.uniform(0.03, 0.12))
        amplitude = float(rng.uniform(0.4, 1.0))
        image += amplitude * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma**2))
    return _normalize(image)


def _gen_voronoi(rng: np.random.Generator, size: int) -> np.ndarray:
    """Cell borders of a random Voronoi tessellation as thin bright lines."""
    X, Y = _grid(size)
    seeds = rng.uniform(-1.0, 1.0, (int(rng.integers(8, 18)), 2))
    distances = np.stack([np.hypot(X - sx, Y - sy) for sx, sy in seeds])
    nearest_two = np.partition(distances, 1, axis=0)[:2]
    width = float(rng.uniform(0.01, 0.03))
    border = np.clip(1.0 - (nearest_two[1] - nearest_two[0]) / width, 0.0, 1.0)
    return _normalize(border)


def _gen_fractal(rng: np.random.Generator, size: int) -> np.ndarray:
    """Recursive random branching tree drawn as thin strokes."""
    image = np.zeros((size, size), dtype=np.float32)
    segments: list = []

    def branch(start: np.ndarray, angle: float, length: float, depth: int) -> None:
        end = np.array(
            [start[0] + length * np.cos(angle), start[1] + length * np.sin(angle)],
            dtype=np.float32,
        )
        segments.append((start.astype(np.float32), end))
        if depth <= 0:
            return
        for _ in range(int(rng.integers(2, 4))):
            branch(end, angle + float(rng.uniform(-0.9, 0.9)), length * float(rng.uniform(0.5, 0.8)), depth - 1)

    branch(np.array([0.0, 0.9], dtype=np.float32), -np.pi / 2.0, 0.5, int(rng.integers(3, 5)))
    _draw_segments(image, segments, float(rng.uniform(0.015, 0.035)))
    return _normalize(image)


_FAMILY_GENERATORS: dict[str, Callable[[np.random.Generator, int], np.ndarray]] = {
    "filament": _gen_filament,
    "curve": _gen_curve,
    "ring": _gen_ring,
    "dot": _gen_dot,
    "voronoi": _gen_voronoi,
    "fractal": _gen_fractal,
}


def _fallback(rng: np.random.Generator, size: int) -> np.ndarray:
    """Guaranteed non-degenerate blob image used when a draw collapses to zeros."""
    X, Y = _grid(size)
    image = np.zeros((size, size), dtype=np.float32)
    for _ in range(2):
        cx, cy = rng.uniform(-0.6, 0.6, 2)
        sigma = float(rng.uniform(0.1, 0.25))
        image += np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma**2))
    return _normalize(image)


def _render(rng: np.random.Generator, size: int, families: Sequence[str]) -> np.ndarray:
    """Composite 1-2 randomly chosen families into one normalized ``(size, size)`` image."""
    image = np.zeros((size, size), dtype=np.float32)
    count = int(rng.integers(1, min(2, len(families)) + 1))
    for family in rng.choice(list(families), size=count, replace=False):
        generator = _FAMILY_GENERATORS.get(str(family))
        if generator is None:
            raise ValueError(f"unknown morphology family {family!r}; valid: {sorted(_FAMILY_GENERATORS)}")
        np.maximum(image, generator(rng, size), out=image)
    image = _normalize(image)
    if float(image.max()) <= _EPS:
        image = _fallback(rng, size)
    return image


# --- forward model (numpy only) ----------------------------------------------


def _block_mean(image: np.ndarray, scale: int) -> np.ndarray:
    """Block-mean downsample by ``scale`` (hr_size must be divisible by scale)."""
    if scale == 1:
        return image.astype(np.float32)
    height, width = image.shape
    return image.reshape(height // scale, scale, width // scale, scale).mean(axis=(1, 3)).astype(np.float32)


def _gaussian_kernel(sigma: float, truncate: float = 3.0) -> np.ndarray:
    """Normalized 1-D Gaussian kernel."""
    radius = max(1, int(truncate * sigma + 0.5))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma * sigma))
    return (kernel / kernel.sum()).astype(np.float32)


def _blur_axis(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Convolve ``image`` along one axis with a 1-D kernel (edge padding)."""
    radius = len(kernel) // 2
    padding = [(0, 0), (0, 0)]
    padding[axis] = (radius, radius)
    padded = np.pad(image, padding, mode="edge")
    out = np.zeros_like(image, dtype=np.float32)
    length = image.shape[axis]
    for offset, weight in enumerate(kernel):
        window = padded[offset : offset + length] if axis == 0 else padded[:, offset : offset + length]
        out += weight * window
    return out


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur written from scratch (numpy only)."""
    if sigma <= 0.0:
        return image
    kernel = _gaussian_kernel(sigma)
    return _blur_axis(_blur_axis(image, kernel, 0), kernel, 1)


def make_synthetic_pair(
    rng: np.random.Generator,
    hr_size: int,
    scale: int,
    families: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Render one synthetic ``(lr, hr)`` pair through a plausible forward model.

    ``hr`` is a ``(hr_size, hr_size)`` float32 image in ``[0, 1]``; ``lr`` is the
    ``(hr_size // scale, hr_size // scale)`` block-mean downsample of ``hr``,
    additionally blurred with a mild Gaussian and optionally corrupted with a
    small amount of Gaussian noise, clamped to ``[0, 1]``. The draw is fully
    determined by ``rng`` (label-free: no structure label is used or returned).
    """
    if hr_size < 1:
        raise ValueError(f"hr_size must be >= 1, got {hr_size}")
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    if hr_size % scale != 0:
        raise ValueError(f"hr_size {hr_size} must be divisible by scale {scale}")
    if not families:
        raise ValueError("families must be non-empty")

    hr = _render(rng, hr_size, families)
    lr = _block_mean(hr, scale)
    lr = _gaussian_blur(lr, float(rng.uniform(0.5, 1.0)))
    if rng.random() < 0.7:
        lr = lr + rng.normal(0.0, float(rng.uniform(0.0, 0.03)), size=lr.shape).astype(np.float32)
    lr = np.clip(lr, 0.0, 1.0).astype(np.float32)
    return lr, hr.astype(np.float32)


class SyntheticMorphologyDataset(Dataset):
    """Label-free synthetic ``(lr, hr)`` pairs for the morphology-OOD ablation.

    Index ``i`` renders with ``numpy.random.Generator(numpy.random.PCG64(seed + i))``,
    so the same ``(seed, index)`` always yields the same pair (reproducible
    training) while different indices differ. Each index composites one or two
    families drawn at random from ``families``.
    """

    def __init__(
        self,
        num_samples: int,
        lr_size: int = 32,
        scale: int = 2,
        families: Sequence[str] = FAMILIES,
        seed: int = 42,
    ) -> None:
        if num_samples < 0:
            raise ValueError(f"num_samples must be >= 0, got {num_samples}")
        if lr_size < 1:
            raise ValueError(f"lr_size must be >= 1, got {lr_size}")
        if scale < 1:
            raise ValueError(f"scale must be >= 1, got {scale}")
        if not families:
            raise ValueError("families must be non-empty")
        unknown = sorted(set(families) - set(_FAMILY_GENERATORS))
        if unknown:
            raise ValueError(f"unknown morphology families {unknown}; valid: {sorted(_FAMILY_GENERATORS)}")
        self.num_samples = int(num_samples)
        self.lr_size = int(lr_size)
        self.scale = int(scale)
        self.families = tuple(families)
        self.seed = int(seed)
        self.hr_size = self.lr_size * self.scale

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(lr, hr)`` as float32 ``(1, lr_size, lr_size)`` / ``(1, hr_size, hr_size)``."""
        rng = np.random.Generator(np.random.PCG64(self.seed + index))
        lr, hr = make_synthetic_pair(rng, self.hr_size, self.scale, self.families)
        return torch.from_numpy(lr).unsqueeze(0), torch.from_numpy(hr).unsqueeze(0)
