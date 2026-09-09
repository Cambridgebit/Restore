"""Structure-balanced sampling (AGENT.md §5): every structure contributes equally."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import torch
from torch.utils.data import WeightedRandomSampler


def make_balanced_sampler(labels: Sequence[str], seed: int = 42) -> WeightedRandomSampler:
    """WeightedRandomSampler with per-item weight ``1 / count[label]``.

    Equalizes the sampling probability of every distinct label so the training
    structures contribute P(S1) ~ P(S2) ~ P(S3) regardless of sample counts.
    ``num_samples`` equals ``len(labels)`` per epoch, drawn with replacement.
    """
    counts = Counter(labels)
    if not counts:
        raise ValueError("labels must be non-empty")
    weights = [1.0 / counts[label] for label in labels]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True, generator=generator)
