"""Tests for the conditional flow-matching x2 super-resolution model."""

import pytest
import torch
import torch.nn.functional as F

from src.models.flow_matching import ConditionalFlowSR, flow_velocity_target

KW: dict[str, object] = {"nf": 8, "num_blocks": 1}


def test_forward_shape_matches_input() -> None:
    """(2, 1, 32, 32) x_t -> (2, 1, 32, 32) velocity, with (N,) t."""
    torch.manual_seed(0)
    model = ConditionalFlowSR(**KW)
    x_t = torch.rand(2, 1, 32, 32)
    lr = torch.rand(2, 1, 16, 16)
    t = torch.rand(2)
    assert model(x_t, t, lr).shape == x_t.shape


def test_forward_scalar_t_broadcast() -> None:
    """A scalar t broadcasts to the batch."""
    torch.manual_seed(0)
    model = ConditionalFlowSR(**KW)
    x_t = torch.rand(2, 1, 32, 32)
    lr = torch.rand(2, 1, 16, 16)
    assert model(x_t, torch.tensor(0.5), lr).shape == x_t.shape


def test_forward_odd_size_pads_and_crops() -> None:
    """Odd spatial sizes are padded internally and cropped back to the input size."""
    torch.manual_seed(0)
    model = ConditionalFlowSR(**KW)
    x_t = torch.rand(1, 1, 18, 10)
    lr = torch.rand(1, 1, 9, 5)
    assert model(x_t, torch.tensor(0.5), lr).shape == (1, 1, 18, 10)


def test_sample_output_is_2x_lr() -> None:
    """(2, 1, 16, 16) LR -> (2, 1, 32, 32) SR."""
    torch.manual_seed(0)
    model = ConditionalFlowSR(**KW).eval()
    lr = torch.rand(2, 1, 16, 16)
    assert model.sample(lr, steps=5).shape == (2, 1, 32, 32)


def test_sample_deterministic_with_fixed_noise() -> None:
    """A fixed noise argument makes sampling deterministic."""
    torch.manual_seed(0)
    model = ConditionalFlowSR(**KW).eval()
    lr = torch.rand(2, 1, 16, 16)
    noise = torch.randn(2, 1, 32, 32)
    first = model.sample(lr, steps=5, noise=noise)
    second = model.sample(lr, steps=5, noise=noise)
    assert torch.equal(first, second)


def test_sample_stochastic_without_noise() -> None:
    """Without noise, two calls draw different x_0 and differ."""
    torch.manual_seed(0)
    model = ConditionalFlowSR(**KW).eval()
    lr = torch.rand(2, 1, 16, 16)
    assert not torch.equal(model.sample(lr, steps=5), model.sample(lr, steps=5))


def test_backward_all_grads_finite() -> None:
    """Backward through forward gives finite gradients on every parameter."""
    torch.manual_seed(0)
    model = ConditionalFlowSR(**KW)
    x_t = torch.rand(2, 1, 32, 32)
    lr = torch.rand(2, 1, 16, 16)
    t = torch.rand(2)
    loss = F.l1_loss(model(x_t, t, lr), torch.rand(2, 1, 32, 32))
    loss.backward()
    assert torch.isfinite(loss)
    for param in model.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_flow_velocity_target_shapes() -> None:
    """x_t and target both match the HR spatial size."""
    torch.manual_seed(0)
    hr = torch.rand(2, 1, 32, 32)
    lr = torch.rand(2, 1, 16, 16)
    noise = torch.randn(2, 1, 32, 32)
    t = torch.rand(2)
    x_t, target = flow_velocity_target(hr, lr, noise, t)
    assert x_t.shape == (2, 1, 32, 32)
    assert target.shape == (2, 1, 32, 32)


def test_flow_velocity_target_t1_identity() -> None:
    """At t=1, x_t == x1 and target == x1 - noise (residual convention)."""
    torch.manual_seed(0)
    hr = torch.rand(2, 1, 32, 32)
    lr = torch.rand(2, 1, 16, 16)
    noise = torch.randn(2, 1, 32, 32)
    t = torch.ones(2)
    x_t, target = flow_velocity_target(hr, lr, noise, t)
    cond = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False, antialias=True)
    x1 = hr - cond
    assert torch.allclose(x_t, x1)
    assert torch.allclose(target, x1 - noise)


def test_flow_velocity_target_no_residual() -> None:
    """Without residual prediction, x1 is the HR image itself."""
    torch.manual_seed(0)
    hr = torch.rand(2, 1, 32, 32)
    lr = torch.rand(2, 1, 16, 16)
    noise = torch.randn(2, 1, 32, 32)
    t = torch.ones(2)
    x_t, target = flow_velocity_target(hr, lr, noise, t, residual_prediction=False)
    assert torch.allclose(x_t, hr)
    assert torch.allclose(target, hr - noise)


def test_rejects_non_x2_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        ConditionalFlowSR(**KW, scale=3)
