"""Tests for the NAFNet / SwinIR x2 super-resolution backbones and the model factory."""

import pytest
import torch
import torch.nn.functional as F

from src.models import build_model
from src.models.nafnet import NAFNet
from src.models.swinir import SwinIR

NAFNET_KW: dict[str, object] = {"nf": 8, "num_blocks": 1}
SWINIR_KW: dict[str, object] = {
    "embed_dim": 12,
    "depths": (1, 1),
    "num_heads": (1, 1),
    "window_size": 4,
}
MODELS = [(NAFNet, NAFNET_KW, "nafnet"), (SwinIR, SWINIR_KW, "swinir")]


@pytest.mark.parametrize(("cls", "kw", "name"), MODELS, ids=["nafnet", "swinir"])
def test_forward_shape(cls: type[torch.nn.Module], kw: dict[str, object], name: str) -> None:
    """(2, 1, 16, 16) input -> exactly (2, 1, 32, 32) output."""
    model = cls(**kw)
    assert model(torch.rand(2, 1, 16, 16)).shape == (2, 1, 32, 32)


@pytest.mark.parametrize(("cls", "kw", "name"), MODELS, ids=["nafnet", "swinir"])
def test_residual_prediction_toggle(
    cls: type[torch.nn.Module], kw: dict[str, object], name: str
) -> None:
    """Both residual_prediction modes run and stay x2."""
    x = torch.rand(1, 1, 16, 16)
    assert cls(**kw, residual_prediction=True)(x).shape == (1, 1, 32, 32)
    assert cls(**kw, residual_prediction=False)(x).shape == (1, 1, 32, 32)


@pytest.mark.parametrize(("cls", "kw", "name"), MODELS, ids=["nafnet", "swinir"])
def test_backward_all_grads_finite(
    cls: type[torch.nn.Module], kw: dict[str, object], name: str
) -> None:
    model = cls(**kw)
    loss = F.l1_loss(model(torch.rand(1, 1, 16, 16)), torch.rand(1, 1, 32, 32))
    loss.backward()
    assert torch.isfinite(loss)
    for param in model.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


@pytest.mark.parametrize(("cls", "kw", "name"), MODELS, ids=["nafnet", "swinir"])
def test_small_and_non_aligned_inputs(
    cls: type[torch.nn.Module], kw: dict[str, object], name: str
) -> None:
    """16x16 inputs work, and non-window-aligned sizes still round-trip to exactly x2."""
    model = cls(**kw)
    assert model(torch.rand(1, 1, 16, 16)).shape == (1, 1, 32, 32)
    assert model(torch.rand(1, 1, 10, 6)).shape == (1, 1, 20, 12)


@pytest.mark.parametrize(("cls", "kw", "name"), MODELS, ids=["nafnet", "swinir"])
def test_rejects_non_x2_scale(cls: type[torch.nn.Module], kw: dict[str, object], name: str) -> None:
    with pytest.raises(ValueError, match="scale"):
        cls(**kw, scale=3)


@pytest.mark.parametrize(("name", "kw"), [("nafnet", NAFNET_KW), ("swinir", SWINIR_KW)])
def test_build_model_constructs(name: str, kw: dict[str, object]) -> None:
    model = build_model({"name": name, **kw})
    assert isinstance(model, (NAFNet, SwinIR))
    assert model(torch.rand(1, 1, 16, 16)).shape == (1, 1, 32, 32)


def test_build_model_rejects_incompatible_keys() -> None:
    with pytest.raises(TypeError, match="window_size"):
        build_model({"name": "nafnet", "window_size": 8})
