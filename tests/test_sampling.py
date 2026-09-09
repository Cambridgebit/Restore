"""Tests for the structure-balanced sampler (AGENT.md §5)."""

from collections import Counter

import pytest
from torch.utils.data import DataLoader

from src.data.biosr import BioSRDataset
from src.data.sampling import make_balanced_sampler
from src.data.splits import make_loso_split


def _train_labels(manifest, held_out="F-actin", seed=42):
    split = make_loso_split(manifest, held_out=held_out, val_fraction=0.34, seed=seed)
    return [sample.structure for sample in split.train]


def test_sampler_balances_structures(manifest):
    labels = _train_labels(manifest)
    big_labels = labels * 200  # scale up for stable frequency statistics
    sampler = make_balanced_sampler(big_labels, seed=42)
    indices = list(iter(sampler))
    assert len(indices) == len(big_labels)
    frequencies = Counter(big_labels[index] for index in indices)
    uniform = 1.0 / len(frequencies)
    for structure, count in frequencies.items():
        share = count / len(indices)
        assert abs(share - uniform) < 0.15 * uniform, f"{structure}: {share:.3f} vs {uniform:.3f}"


def test_sampler_deterministic(manifest):
    labels = _train_labels(manifest)
    first = list(iter(make_balanced_sampler(labels, seed=7)))
    second = list(iter(make_balanced_sampler(labels, seed=7)))
    assert first == second
    assert all(0 <= index < len(labels) for index in first)


def test_sampler_empty_labels_raises():
    with pytest.raises(ValueError):
        make_balanced_sampler([])


def test_sampler_works_in_dataloader(manifest, biosr_root):
    split = make_loso_split(manifest, held_out="F-actin", val_fraction=0.34, seed=42)
    dataset = BioSRDataset(split.train, root=biosr_root)
    loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=make_balanced_sampler([s.structure for s in split.train], seed=42),
    )
    lr, gt = next(iter(loader))
    assert lr.shape[0] == 4
    assert lr.shape[1] == 1
    assert gt.shape[-2] == 2 * lr.shape[-2]
    assert gt.shape[-1] == 2 * lr.shape[-1]
