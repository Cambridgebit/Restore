"""Tests for the MambaIR x2 super-resolution backbone."""

import pytest
import torch
import torch.nn.functional as F

from src.models.mambair import MambaIR

MAMBAIR_KW: dict[str, object] = {"embed_dim": 8, "depths": (1, 1), "d_state": 4}


def test_forward_shape() -> None:
    """(2, 1, 16, 16) input -> exactly (2, 1, 32, 32) output."""
    model = MambaIR(**MAMBAIR_KW)
    assert model(torch.rand(2, 1, 16, 16)).shape == (2, 1, 32, 32)


def test_residual_prediction_toggle() -> None:
    """Both residual_prediction modes run and stay x2."""
    x = torch.rand(1, 1, 16, 16)
    assert MambaIR(**MAMBAIR_KW, residual_prediction=True)(x).shape == (1, 1, 32, 32)
    assert MambaIR(**MAMBAIR_KW, residual_prediction=False)(x).shape == (1, 1, 32, 32)


def test_backward_all_grads_finite() -> None:
    """Backward through the sequential scan yields finite grads for every parameter."""
    model = MambaIR(**MAMBAIR_KW)
    loss = F.l1_loss(model(torch.rand(1, 1, 16, 16)), torch.rand(1, 1, 32, 32))
    loss.backward()
    assert torch.isfinite(loss)
    for param in model.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_small_and_non_aligned_inputs() -> None:
    """16x16 inputs work, and non-window-friendly sizes still round-trip to exactly x2."""
    model = MambaIR(**MAMBAIR_KW)
    assert model(torch.rand(1, 1, 16, 16)).shape == (1, 1, 32, 32)
    assert model(torch.rand(1, 1, 10, 6)).shape == (1, 1, 20, 12)


def test_rejects_non_x2_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        MambaIR(**MAMBAIR_KW, scale=3)
