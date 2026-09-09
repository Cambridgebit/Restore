"""Tests for src.metrics (AGENT.md §10-11: PSNR, SSIM/MS-SSIM, ZNCC, FRC, structural P/R/F1)."""

import math

import pytest
import torch

from src.metrics.image import frc_resolution, ms_ssim, psnr, ssim, zncc
from src.metrics.structure import edge_map, structural_precision_recall


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def test_psnr_identical_inputs_is_large_finite() -> None:
    x = torch.rand(2, 1, 32, 32)
    value = psnr(x, x)
    assert math.isfinite(value)
    assert value > 60.0  # floored mse 1e-12 -> 120 dB


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ssim_metric_on_cuda_inputs() -> None:
    """SSIM machinery must build its kernel on the input device."""
    x = torch.rand(1, 1, 32, 32, device="cuda")
    assert ssim(x, x, window=11) == pytest.approx(1.0, abs=1e-5)
    assert math.isfinite(psnr(x, x))
    assert math.isfinite(zncc(x, x))


def test_psnr_decreases_with_noise() -> None:
    x = torch.rand(1, 1, 64, 64)
    values = [
        psnr(x, (x + sigma * torch.randn_like(x)).clamp(0.0, 1.0)) for sigma in (0.05, 0.1, 0.2)
    ]
    assert values[0] > values[1] > values[2]


def test_ssim_metric_identical_is_one() -> None:
    x = torch.rand(2, 1, 32, 32)
    assert ssim(x, x) == pytest.approx(1.0, abs=1e-6)


def test_ssim_metric_different_below_one() -> None:
    x = torch.rand(2, 1, 32, 32)
    y = torch.rand(2, 1, 32, 32)
    assert ssim(x, y) < 1.0


def test_ms_ssim_identical_is_one() -> None:
    x = torch.rand(1, 1, 256, 256)
    assert ms_ssim(x, x) == pytest.approx(1.0, abs=1e-6)


def test_ms_ssim_too_small_input_raises() -> None:
    x = torch.rand(1, 1, 32, 32)  # needs H,W >= 11 * 2**4 = 176
    with pytest.raises(ValueError, match="ms_ssim"):
        ms_ssim(x, x)


def test_zncc_identical_is_one() -> None:
    x = torch.rand(2, 1, 32, 32)
    assert zncc(x, x) == pytest.approx(1.0, abs=1e-6)


def test_zncc_anticorrelated_is_minus_one() -> None:
    x = torch.rand(1, 1, 32, 32)
    assert zncc(x, 1.0 - x) == pytest.approx(-1.0, abs=1e-6)


def test_frc_identical_is_one() -> None:
    x = torch.rand(1, 1, 64, 64)
    assert frc_resolution(x, x) == pytest.approx(1.0, abs=1e-6)


def test_frc_uncorrelated_is_low() -> None:
    x = torch.rand(2, 1, 32, 32)
    y = torch.rand(2, 1, 32, 32)
    value = frc_resolution(x, y)
    assert math.isfinite(value)
    assert 0.0 <= value < 0.3


def test_edge_map_shape_and_binary_values() -> None:
    x = torch.rand(2, 1, 32, 32)
    edges = edge_map(x)
    assert edges.shape == (2, 1, 32, 32)
    assert edges.dtype == torch.float32
    assert set(edges.unique().tolist()) <= {0.0, 1.0}


def test_edge_map_flags_step_edge() -> None:
    x = torch.zeros(1, 1, 32, 32)
    x[..., 16:] = 1.0
    # percentile=50 -> threshold 0 -> every nonzero-magnitude pixel is flagged
    edges = edge_map(x, percentile=50.0)
    assert edges[..., :15].sum() == 0.0
    assert edges[..., 15:17].sum() == 64.0  # both sides of the step, all rows
    assert edges[..., 17:].sum() == 0.0


def test_edge_map_constant_image_is_empty() -> None:
    x = torch.full((1, 1, 32, 32), 0.5)
    assert edge_map(x).sum() == 0.0  # zero gradient everywhere -> empty edge map


def test_structural_pr_identical_inputs_perfect() -> None:
    x = torch.rand(1, 1, 64, 64)
    result = structural_precision_recall(x, x)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


def test_structural_pr_spurious_edge_lowers_precision() -> None:
    gt = torch.zeros(1, 1, 64, 64)
    gt[..., 32:] = 1.0  # step edge
    sr = gt.clone()
    sr[..., :8] = 1.0  # extra step far from any GT edge
    result = structural_precision_recall(sr, gt, percentile=50.0)
    assert result["precision"] < 1.0
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] < 1.0


def test_structural_pr_missing_edge_lowers_recall() -> None:
    gt = torch.zeros(1, 1, 64, 64)
    gt[..., 32:] = 1.0
    gt[..., :8] = 1.0  # second edge the SR output lacks
    sr = torch.zeros(1, 1, 64, 64)
    sr[..., 32:] = 1.0
    result = structural_precision_recall(sr, gt, percentile=50.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] < 1.0
    assert result["f1"] < 1.0


def test_structural_pr_both_empty_returns_zeros() -> None:
    x = torch.full((1, 1, 32, 32), 0.5)
    result = structural_precision_recall(x, x)
    assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_structural_pr_batch_mean_matches_single() -> None:
    x = torch.rand(1, 1, 64, 64)
    y = torch.rand(1, 1, 64, 64)
    single = structural_precision_recall(x, y)
    batch = structural_precision_recall(torch.cat([x, x]), torch.cat([y, y]))
    assert batch["precision"] == pytest.approx(single["precision"])
    assert batch["recall"] == pytest.approx(single["recall"])


def test_structural_pr_tolerance_sweep_in_unit_range() -> None:
    x = torch.rand(1, 1, 64, 64)
    y = (0.5 * x + 0.5 * torch.rand_like(x)).clamp(0.0, 1.0)
    for tolerance in (1, 2, 3):
        result = structural_precision_recall(x, y, tolerance_px=tolerance)
        assert all(0.0 <= v <= 1.0 for v in result.values())
