"""Tests for the synthetic morphology-OOD paired-image generator."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.morphology import FAMILIES, SyntheticMorphologyDataset, make_synthetic_pair


def test_make_synthetic_pair_shapes_and_range():
    rng = np.random.default_rng(0)
    lr, hr = make_synthetic_pair(rng, hr_size=64, scale=2, families=FAMILIES)
    assert lr.shape == (32, 32)
    assert hr.shape == (64, 64)
    assert lr.dtype == np.float32 and hr.dtype == np.float32
    for image in (lr, hr):
        assert np.isfinite(image).all()
        assert image.min() >= 0.0 and image.max() <= 1.0
    assert hr.std() > 0.0


def test_make_synthetic_pair_scale_respected():
    rng = np.random.default_rng(1)
    lr, hr = make_synthetic_pair(rng, hr_size=48, scale=3, families=["dot"])
    assert lr.shape == (16, 16)
    assert hr.shape == (48, 48)


@pytest.mark.parametrize("family", FAMILIES)
def test_each_family_non_degenerate(family):
    for seed in range(4):
        rng = np.random.default_rng(seed)
        _, hr = make_synthetic_pair(rng, hr_size=64, scale=2, families=[family])
        assert hr.std() > 0.0, f"{family} seed={seed}"
        assert hr.max() > 0.0, f"{family} seed={seed}"


def test_dataset_shapes_and_dtype():
    dataset = SyntheticMorphologyDataset(num_samples=5, lr_size=16, scale=2, seed=42)
    assert len(dataset) == 5
    lr, hr = dataset[0]
    assert lr.shape == (1, 16, 16)
    assert hr.shape == (1, 32, 32)
    assert lr.dtype == torch.float32 and hr.dtype == torch.float32
    assert lr.min() >= 0.0 and lr.max() <= 1.0
    assert hr.min() >= 0.0 and hr.max() <= 1.0


def test_dataset_lr_size_and_scale_respected():
    dataset = SyntheticMorphologyDataset(num_samples=2, lr_size=24, scale=3, seed=7)
    lr, hr = dataset[0]
    assert lr.shape == (1, 24, 24)
    assert hr.shape == (1, 72, 72)


def test_dataset_deterministic_per_index():
    dataset = SyntheticMorphologyDataset(num_samples=8, lr_size=16, scale=2, seed=42)
    first_lr, first_hr = dataset[3]
    second_lr, second_hr = dataset[3]
    assert torch.equal(first_lr, second_lr)
    assert torch.equal(first_hr, second_hr)
    other_lr, other_hr = dataset[4]
    assert not torch.equal(first_hr, other_hr)
    assert not torch.equal(first_lr, other_lr)


def test_dataset_seed_changes_pair():
    a = SyntheticMorphologyDataset(num_samples=1, lr_size=16, scale=2, seed=42)[0][1]
    b = SyntheticMorphologyDataset(num_samples=1, lr_size=16, scale=2, seed=43)[0][1]
    assert not torch.equal(a, b)


def test_dataset_unknown_family_raises():
    with pytest.raises(ValueError, match="families"):
        SyntheticMorphologyDataset(num_samples=1, families=["nope"])


def test_dataset_in_dataloader():
    dataset = SyntheticMorphologyDataset(num_samples=4, lr_size=16, scale=2, seed=0)
    loader = DataLoader(dataset, batch_size=2)
    lr, hr = next(iter(loader))
    assert lr.shape == (2, 1, 16, 16)
    assert hr.shape == (2, 1, 32, 32)
    assert lr.dtype == torch.float32 and hr.dtype == torch.float32
