"""Tests for the extended loss package: new components and the optional lr context."""

import pytest
import torch
import torch.nn.functional as F

from src.losses import COMPONENT_DEFAULTS, WeightedSumLoss, build_loss
from src.losses.consistency import DataConsistencyLoss, ResidualLoss
from src.losses.frequency import FocalFrequencyLoss
from src.losses.structural import GradientVarianceLoss, HessianStructureLoss

SCALE = 2
SIZE = 32


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def _batch(size: int = SIZE, scale: int = SCALE) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (pred, target, lr) with lr downsampled by `scale`."""
    pred = torch.rand(1, 1, size, size)
    target = torch.rand(1, 1, size, size)
    lr = torch.rand(1, 1, size // scale, size // scale)
    return pred, target, lr


LR_INDEPENDENT = (
    ("focal_frequency", FocalFrequencyLoss),
    ("gradient_variance", GradientVarianceLoss),
    ("hessian", HessianStructureLoss),
)
LR_DEPENDENT = (
    ("data_consistency", DataConsistencyLoss),
    ("residual", ResidualLoss),
)


@pytest.mark.parametrize("name,factory", LR_INDEPENDENT, ids=[n for n, _ in LR_INDEPENDENT])
def test_lr_independent_new_losses_finite_with_finite_grads(
    name: str, factory: type[torch.nn.Module]
) -> None:
    pred, target, _ = _batch()
    pred.requires_grad_(True)
    value = factory()(pred, target)
    assert torch.isfinite(value), name
    value.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all(), name


@pytest.mark.parametrize("name,factory", LR_DEPENDENT, ids=[n for n, _ in LR_DEPENDENT])
def test_lr_dependent_new_losses_finite_with_finite_grads(
    name: str, factory: type[torch.nn.Module]
) -> None:
    pred, target, lr = _batch()
    pred.requires_grad_(True)
    value = factory()(pred, target, lr=lr)
    assert torch.isfinite(value), name
    assert value.item() > 0.0, name
    value.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all(), name


@pytest.mark.parametrize("name,factory", LR_INDEPENDENT, ids=[n for n, _ in LR_INDEPENDENT])
def test_lr_independent_new_losses_zero_on_identical(
    name: str, factory: type[torch.nn.Module]
) -> None:
    x = torch.rand(1, 1, SIZE, SIZE)
    assert factory()(x, x).item() == pytest.approx(0.0, abs=1e-6), name


@pytest.mark.parametrize("name,factory", LR_DEPENDENT, ids=[n for n, _ in LR_DEPENDENT])
def test_lr_dependent_new_losses_zero_when_lr_is_none(
    name: str, factory: type[torch.nn.Module]
) -> None:
    pred, target, _ = _batch()
    assert factory()(pred, target, lr=None).item() == 0.0, name


@pytest.mark.parametrize("name,factory", LR_DEPENDENT, ids=[n for n, _ in LR_DEPENDENT])
def test_lr_dependent_new_losses_change_with_lr(
    name: str, factory: type[torch.nn.Module]
) -> None:
    pred, target, lr = _batch()
    component = factory()
    with_lr = component(pred, target, lr=lr)
    other_lr = component(pred, target, lr=lr + 0.5)
    assert not torch.allclose(with_lr, other_lr), name


def test_existing_components_accept_and_ignore_lr() -> None:
    from src.losses.frequency import FourierLoss
    from src.losses.pixel import CharbonnierLoss
    from src.losses.structural import GradientLoss, SSIMLoss

    pred, target, lr = _batch()
    for component in (CharbonnierLoss(), SSIMLoss(), GradientLoss(), FourierLoss()):
        assert torch.allclose(component(pred, target, lr=lr), component(pred, target))


def test_data_consistency_matches_manual_avgpool_l1() -> None:
    pred, target, lr = _batch()
    expected = (F.avg_pool2d(pred, SCALE) - lr).abs().mean()
    assert torch.allclose(DataConsistencyLoss()(pred, target, lr=lr), expected)


def test_data_consistency_bicubic_down_is_finite() -> None:
    pred, target, lr = _batch()
    value = DataConsistencyLoss(mode="bicubic_down")(pred, target, lr=lr)
    assert torch.isfinite(value)
    assert value.item() > 0.0


def test_data_consistency_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="strided"):
        DataConsistencyLoss(mode="strided")


def test_residual_matches_manual_bicubic_upsample() -> None:
    pred, target, lr = _batch()
    up = F.interpolate(lr, scale_factor=SCALE, mode="bicubic", align_corners=False, antialias=True)
    expected = (pred - up).abs().mean()
    assert torch.allclose(ResidualLoss()(pred, target, lr=lr), expected)


def test_gradient_variance_patch_size_configurable() -> None:
    pred, target, _ = _batch()
    value = GradientVarianceLoss(patch_size=4)(pred, target)
    assert torch.isfinite(value)
    assert value.item() > 0.0


def test_focal_frequency_perturbation_increases_loss() -> None:
    x = torch.rand(1, 1, SIZE, SIZE)
    loss = FocalFrequencyLoss()
    near = loss(x, x + 0.01 * torch.rand_like(x))
    far = loss(x, x + 0.5 * torch.rand_like(x))
    assert near.item() < far.item()


def test_build_loss_accepts_new_component_keys() -> None:
    loss = build_loss(
        {
            "charbonnier": 0.0,
            "ssim": 0.0,
            "gradient": 0.0,
            "fourier": 0.0,
            "data_consistency": 1.0,
            "residual": 1.0,
            "focal_frequency": 0.5,
            "gradient_variance": 0.5,
            "hessian": 0.5,
        }
    )
    pred, target, lr = _batch()
    total, parts = loss(pred, target, lr=lr)
    assert set(parts) == {
        "data_consistency",
        "residual",
        "focal_frequency",
        "gradient_variance",
        "hessian",
        "total",
    }
    expected = (
        parts["data_consistency"]
        + parts["residual"]
        + 0.5 * parts["focal_frequency"]
        + 0.5 * parts["gradient_variance"]
        + 0.5 * parts["hessian"]
    )
    assert torch.allclose(total, expected)


def test_build_loss_new_defaults_are_inactive_zero() -> None:
    loss = build_loss({})
    for name in ("data_consistency", "residual", "focal_frequency", "gradient_variance", "hessian"):
        assert COMPONENT_DEFAULTS[name] == 0.0
        assert name not in loss.component_names


def test_build_loss_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="bogus"):
        build_loss({"bogus": 1.0})


def test_build_loss_scale_forwarded_to_consistency_components() -> None:
    pred = torch.rand(1, 1, SIZE, SIZE)
    target = torch.rand(1, 1, SIZE, SIZE)
    lr = torch.rand(1, 1, SIZE // 4, SIZE // 4)
    loss = build_loss(
        {
            "charbonnier": 0.0,
            "ssim": 0.0,
            "gradient": 0.0,
            "fourier": 0.0,
            "data_consistency": 1.0,
            "residual": 1.0,
        },
        scale=4,
    )
    total, parts = loss(pred, target, lr=lr)
    assert torch.isfinite(total)
    assert torch.isfinite(parts["data_consistency"])
    assert torch.isfinite(parts["residual"])
    assert torch.allclose(total, parts["data_consistency"] + parts["residual"])


def test_weighted_sum_loss_passes_lr_to_components() -> None:
    pred, target, lr = _batch()
    loss = WeightedSumLoss(
        {
            "charbonnier": 0.0,
            "ssim": 0.0,
            "gradient": 0.0,
            "fourier": 0.0,
            "data_consistency": 1.0,
        }
    )
    total_with, parts_with = loss(pred, target, lr=lr)
    total_without, parts_without = loss(pred, target, lr=None)
    assert parts_with["data_consistency"].item() > 0.0
    assert parts_without["data_consistency"].item() == 0.0
    assert torch.allclose(total_with, parts_with["data_consistency"])
    assert total_without.item() == 0.0


def test_weighted_sum_loss_backward_with_lr_is_finite() -> None:
    pred, target, lr = _batch()
    pred.requires_grad_(True)
    loss = build_loss(
        {
            "charbonnier": 0.0,
            "ssim": 0.0,
            "gradient": 0.0,
            "fourier": 0.0,
            "data_consistency": 1.0,
            "residual": 1.0,
            "focal_frequency": 1.0,
            "gradient_variance": 1.0,
            "hessian": 1.0,
        }
    )
    total, _ = loss(pred, target, lr=lr)
    total.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
