"""Tests for src.losses (AGENT.md §16: loss finite, gradient finite)."""

import pytest
import torch

from src.losses import COMPONENT_DEFAULTS, WeightedSumLoss, build_loss
from src.losses.frequency import FourierLoss
from src.losses.pixel import CharbonnierLoss
from src.losses.structural import GradientLoss, SSIMLoss, ssim_index


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def test_charbonnier_identical_inputs_equals_eps() -> None:
    x = torch.rand(2, 1, 8, 8)
    loss = CharbonnierLoss()(x, x)
    assert loss.item() == pytest.approx(1e-3, rel=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_losses_run_on_cuda_inputs_with_cpu_module() -> None:
    """Loss modules left on CPU must still work on CUDA tensors (device-aware internals)."""
    x = torch.rand(2, 1, 16, 16, device="cuda")
    y = torch.rand(2, 1, 16, 16, device="cuda")
    for component in (CharbonnierLoss(), SSIMLoss(), GradientLoss(), FourierLoss()):
        value = component(x, y)
        assert torch.isfinite(value)
    total, parts = build_loss({"charbonnier": 1.0, "ssim": 0.1, "gradient": 0.1, "fourier": 0.5})(x, y)
    assert torch.isfinite(total)
    for name, value in parts.items():
        assert torch.isfinite(value), name


def test_charbonnier_different_inputs_larger_and_finite() -> None:
    x = torch.rand(2, 1, 8, 8)
    y = torch.rand(2, 1, 8, 8)
    loss = CharbonnierLoss()(x, y)
    assert torch.isfinite(loss)
    assert loss.item() > 1e-3


def test_charbonnier_backward_gives_finite_grads() -> None:
    x = torch.rand(1, 1, 8, 8, requires_grad=True)
    y = torch.rand(1, 1, 8, 8)
    CharbonnierLoss()(x, y).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_ssim_index_identical_is_one() -> None:
    x = torch.rand(2, 1, 32, 32)
    assert ssim_index(x, x).item() == pytest.approx(1.0, abs=1e-6)


def test_ssim_index_different_in_half_open_range() -> None:
    x = torch.rand(2, 1, 32, 32)
    y = torch.rand(2, 1, 32, 32)
    value = ssim_index(x, y).item()
    assert 0.0 <= value < 1.0


def test_ssim_loss_identical_is_zero() -> None:
    x = torch.rand(1, 1, 32, 32)
    assert SSIMLoss()(x, x).item() == pytest.approx(0.0, abs=1e-6)


def test_gradient_loss_matches_manual_l1() -> None:
    pred = torch.tensor([[[[1.0, 2.0, 4.0], [0.0, 5.0, 9.0], [3.0, 1.0, 2.0]]]])
    target = torch.tensor([[[[2.0, 1.0, 3.0], [1.0, 4.0, 2.0], [0.0, 5.0, 1.0]]]])
    # Central differences: dx(pred) = [3, 9, -1], dy(pred) = [2, -1, -2];
    # dx(target) = [1, 1, 1], dy(target) = [-2, 4, -2].
    # L1 = mean(|2, 8, -2|) + mean(|4, -5, 0|) = 4.0 + 3.0
    actual = GradientLoss()(pred, target)
    assert torch.allclose(actual, torch.tensor(7.0), atol=1e-6)
    # Versus zeros: mean(|3, 9, -1|) + mean(|2, -1, -2|) = 13/3 + 5/3
    zeros = torch.zeros(1, 1, 3, 3)
    assert torch.allclose(GradientLoss()(pred, zeros), torch.tensor(6.0), atol=1e-6)


def test_gradient_loss_identical_is_zero() -> None:
    x = torch.rand(1, 1, 16, 16)
    assert GradientLoss()(x, x).item() == pytest.approx(0.0, abs=1e-7)


def test_fourier_loss_identical_is_zero() -> None:
    x = torch.rand(1, 1, 32, 32)
    assert FourierLoss()(x, x).item() == pytest.approx(0.0, abs=1e-7)


def test_fourier_loss_amplitude_perturbation_increases_and_finite() -> None:
    x = torch.rand(1, 1, 32, 32)
    y = x + 0.1 * torch.rand_like(x)
    loss = FourierLoss()(x, y)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_build_loss_matches_contract_weights() -> None:
    x = torch.rand(1, 1, 32, 32)
    y = torch.rand(1, 1, 32, 32)
    loss = build_loss({"charbonnier": 1.0, "ssim": 0.1, "gradient": 0.1, "fourier": 0.0})
    total, parts = loss(x, y)
    assert "fourier" not in parts
    assert set(parts) == {"charbonnier", "ssim", "gradient", "total"}
    expected = 1.0 * parts["charbonnier"] + 0.1 * parts["ssim"] + 0.1 * parts["gradient"]
    assert torch.allclose(total, expected)
    assert loss.component_names == ("charbonnier", "ssim", "gradient")


def test_build_loss_weight_change_reflected() -> None:
    x = torch.rand(1, 1, 32, 32)
    y = torch.rand(1, 1, 32, 32)
    total_a, parts_a = build_loss({"ssim": 0.0, "fourier": 0.0})(x, y)
    total_b, parts_b = build_loss({"charbonnier": 2.0, "ssim": 0.0, "fourier": 0.0})(x, y)
    assert torch.allclose(parts_a["charbonnier"], parts_b["charbonnier"])
    assert torch.allclose(total_b - total_a, parts_b["charbonnier"])


def test_build_loss_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="bogus"):
        build_loss({"bogus": 1.0})


def test_build_loss_empty_cfg_uses_component_defaults() -> None:
    loss = build_loss({})
    assert loss.weights == COMPONENT_DEFAULTS
    # fourier defaults to 0.0 -> inactive
    assert loss.component_names == ("charbonnier", "ssim", "gradient")
    total, parts = loss(torch.rand(1, 1, 32, 32), torch.rand(1, 1, 32, 32))
    assert set(parts) == {"charbonnier", "ssim", "gradient", "total"}
    assert torch.allclose(parts["total"], total)


def test_weighted_sum_loss_parts_consistent() -> None:
    x = torch.rand(1, 1, 16, 16)
    y = torch.rand(1, 1, 16, 16)
    loss = WeightedSumLoss({"charbonnier": 1.0, "ssim": 0.0, "gradient": 0.0, "fourier": 0.5})
    assert loss.component_names == ("charbonnier", "fourier")
    total, parts = loss(x, y)
    assert set(parts) == {"charbonnier", "fourier", "total"}
    assert torch.allclose(parts["total"], total)
    assert torch.allclose(total, parts["charbonnier"] + 0.5 * parts["fourier"])


def test_weighted_sum_loss_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        WeightedSumLoss({"nope": 1.0})


@pytest.mark.parametrize(
    "loss",
    [
        CharbonnierLoss(),
        SSIMLoss(),
        GradientLoss(),
        FourierLoss(),
        build_loss({}),
    ],
    ids=["charbonnier", "ssim", "gradient", "fourier", "weighted_sum"],
)
def test_all_losses_finite_with_finite_grads(loss: torch.nn.Module) -> None:
    x = torch.rand(1, 1, 16, 16, requires_grad=True)
    y = torch.rand(1, 1, 16, 16)
    output = loss(x, y)
    objective = output[0] if isinstance(loss, WeightedSumLoss) else output
    assert torch.isfinite(objective).all()
    objective.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
