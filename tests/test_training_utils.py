"""Standalone unit tests for the training utilities: ModelEMA and SAM."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from src.utils.ema import ModelEMA
from src.utils.sam import SAM


def _linear_model(dtype: torch.dtype = torch.float32) -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(4, 1).to(dtype=dtype)


def _regression_batch(n: int = 32, d: int = 4, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=generator)
    noise = 0.1 * torch.randn(n, 1, generator=generator)
    target = x @ torch.randn(d, 1, generator=generator) + noise
    return x, target


def _loss(model: nn.Module, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(model(x), target)


# ------------------------------------------------------------------------------ EMA


def test_ema_approaches_source_parameters():
    model = _linear_model()
    ema = ModelEMA(model, decay=0.999)
    with torch.no_grad():
        model.weight.add_(1.5)
        model.bias.sub_(0.5)
    for _ in range(200):
        ema.update(model)
    assert ema.step == 200
    for ema_param, param in zip(ema.module.parameters(), model.parameters(), strict=True):
        assert torch.allclose(ema_param, param, atol=1e-3)


def test_ema_wraps_a_frozen_eval_copy():
    model = _linear_model()
    ema = ModelEMA(model)
    assert ema.module is not model
    assert not ema.module.training
    assert all(not p.requires_grad for p in ema.module.parameters())
    assert ema.decay == 0.999


def test_ema_copy_to_replaces_model_parameters():
    model = _linear_model()
    ema = ModelEMA(model, decay=0.9)
    with torch.no_grad():
        model.weight.add_(2.0)
    ema.update(model)
    assert not torch.equal(ema.module.weight, model.weight)  # ramp decay < 1: not a plain copy
    ema.copy_to(model)
    for ema_param, param in zip(ema.module.parameters(), model.parameters(), strict=True):
        assert torch.equal(ema_param, param)


def test_ema_copies_buffers_without_averaging():
    model = nn.BatchNorm1d(3)
    ema = ModelEMA(model, decay=0.9)
    with torch.no_grad():
        model.running_mean.fill_(5.0)
        model.running_var.fill_(7.0)
    ema.update(model)
    assert torch.equal(ema.module.running_mean, torch.full_like(model.running_mean, 5.0))
    assert torch.equal(ema.module.running_var, torch.full_like(model.running_var, 7.0))
    ema.copy_to(model)
    assert torch.equal(model.running_mean, torch.full_like(model.running_mean, 5.0))
    assert torch.equal(model.running_var, torch.full_like(model.running_var, 7.0))


def test_ema_state_dict_round_trips():
    model = _linear_model()
    ema = ModelEMA(model, decay=0.98)
    with torch.no_grad():
        model.weight.add_(1.0)
    for _ in range(5):
        ema.update(model)
    state = ema.state_dict()
    assert state["decay"] == 0.98
    assert state["step"] == 5
    restored = ModelEMA(copy.deepcopy(model), decay=0.5)
    restored.load_state_dict(state)
    assert restored.decay == ema.decay
    assert restored.step == ema.step
    for restored_param, ema_param in zip(
        restored.module.parameters(), ema.module.parameters(), strict=True
    ):
        assert torch.equal(restored_param, ema_param)
    for restored_buffer, ema_buffer in zip(
        restored.module.buffers(), ema.module.buffers(), strict=True
    ):
        assert torch.equal(restored_buffer, ema_buffer)


def test_ema_preserves_dtype_and_device():
    model = _linear_model(dtype=torch.float64)
    ema = ModelEMA(model, decay=0.5)
    assert ema.module.weight.dtype == torch.float64
    assert ema.module.weight.device == model.weight.device
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    assert ema.module.weight.dtype == torch.float64
    assert ema.module.weight.device == model.weight.device
    ema.copy_to(model)
    assert model.weight.dtype == torch.float64


# ------------------------------------------------------------------------------ SAM


def test_sam_first_step_perturbs_and_second_step_restores():
    model = _linear_model()
    optimizer = SAM(model.parameters(), base_optimizer_cls=torch.optim.Adam, lr=0.0, rho=0.05)
    assert isinstance(optimizer.base_optimizer, torch.optim.Adam)
    x, target = _regression_batch()
    _loss(model, x, target).backward()
    original = [p.detach().clone() for p in model.parameters()]

    optimizer.first_step(zero_grad=True)
    perturbed = [p.detach().clone() for p in model.parameters()]
    assert all(p.grad is None for p in model.parameters())
    perturbation = torch.cat(
        [(pert - orig).flatten() for pert, orig in zip(perturbed, original, strict=True)]
    )
    assert torch.allclose(perturbation.norm(), torch.tensor(0.05), atol=1e-3)
    for p, pert, orig in zip(model.parameters(), perturbed, original, strict=True):
        assert torch.allclose(optimizer.state[p]["e_w"], pert - orig)
    assert any(
        not torch.allclose(pert, orig) for pert, orig in zip(perturbed, original, strict=True)
    )

    # Recompute gradients at the perturbed point; second_step restores w then applies
    # the base optimizer at lr=0, so the parameters must return to their originals.
    _loss(model, x, target).backward()
    optimizer.second_step(zero_grad=True)
    for param, orig in zip(model.parameters(), original, strict=True):
        assert torch.allclose(param, orig)
    assert all(p.grad is None for p in model.parameters())


def test_sam_reduces_loss_over_training_cycles():
    torch.manual_seed(0)
    model = _linear_model()
    optimizer = SAM(model.parameters(), base_optimizer_cls=torch.optim.Adam, lr=0.05, rho=0.05)
    x, target = _regression_batch()
    initial = _loss(model, x, target).item()
    for _ in range(20):
        _loss(model, x, target).backward()
        optimizer.first_step(zero_grad=True)
        _loss(model, x, target).backward()
        optimizer.second_step(zero_grad=True)
    final = _loss(model, x, target).item()
    assert final < initial


def test_sam_state_dict_round_trips():
    torch.manual_seed(0)
    model = _linear_model()
    optimizer = SAM(model.parameters(), base_optimizer_cls=torch.optim.Adam, lr=0.01, rho=0.1)
    x, target = _regression_batch()
    _loss(model, x, target).backward()
    optimizer.first_step(zero_grad=True)
    _loss(model, x, target).backward()
    optimizer.second_step(zero_grad=True)

    state = optimizer.state_dict()
    assert state["param_groups"][0]["lr"] == 0.01
    assert state["param_groups"][0]["rho"] == 0.1
    other = _linear_model()
    restored = SAM(other.parameters(), base_optimizer_cls=torch.optim.Adam, lr=0.01, rho=0.1)
    restored.load_state_dict(state)
    assert restored.param_groups[0]["lr"] == 0.01
    assert restored.param_groups[0]["rho"] == 0.1
    assert restored.param_groups[0]["params"][0] is other.weight
    restored_state = restored.state_dict()
    for index, saved_param_state in state["state"].items():
        for key, value in saved_param_state.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(restored_state["state"][index][key], value)
            else:
                assert restored_state["state"][index][key] == value


def test_sam_step_raises_directly():
    model = _linear_model()
    optimizer = SAM(model.parameters(), base_optimizer_cls=torch.optim.Adam, lr=0.01)
    with pytest.raises(RuntimeError):
        optimizer.step()


def test_sam_zero_grad_forwards_to_base_optimizer():
    model = _linear_model()
    optimizer = SAM(model.parameters(), base_optimizer_cls=torch.optim.Adam, lr=0.01)
    x, target = _regression_batch()
    _loss(model, x, target).backward()
    assert all(p.grad is not None for p in model.parameters())
    optimizer.zero_grad()
    assert all(p.grad is None for p in model.parameters())
