"""Tests for the DFCAN / RCAN x2 super-resolution backbones and the model factory."""

import pytest
import torch
import torch.nn.functional as F

from src.models import build_model
from src.models.dfcan import DFCAN
from src.models.rcan import RCAN

DFCAN_KW: dict[str, object] = {"nf": 16, "num_groups": 1, "num_blocks": 1}
RCAN_KW: dict[str, object] = {"nf": 16, "num_groups": 1, "num_blocks": 1}
MODELS = [(DFCAN, DFCAN_KW), (RCAN, RCAN_KW)]


@pytest.mark.parametrize(("cls", "kw"), MODELS, ids=["dfcan", "rcan"])
def test_forward_shape(cls: type[torch.nn.Module], kw: dict[str, object]) -> None:
    """(2, 1, 16, 16) input -> (2, 1, 32, 32) output."""
    model = cls(**kw)
    assert model(torch.rand(2, 1, 16, 16)).shape == (2, 1, 32, 32)


@pytest.mark.parametrize(("cls", "kw"), MODELS, ids=["dfcan", "rcan"])
def test_forward_shape_odd_input(cls: type[torch.nn.Module], kw: dict[str, object]) -> None:
    """Odd spatial dims (15) round-trip to exactly 30 after x2 upsampling."""
    model = cls(**kw)
    assert model(torch.rand(1, 1, 15, 15)).shape == (1, 1, 30, 30)


def test_dfcan_rejects_non_x2_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        DFCAN(**DFCAN_KW, scale=3)


def test_rcan_accepts_x2_scale() -> None:
    assert isinstance(RCAN(**RCAN_KW, scale=2), RCAN)


@pytest.mark.parametrize(("cls", "kw"), MODELS, ids=["dfcan", "rcan"])
def test_residual_prediction_false_shape(cls: type[torch.nn.Module], kw: dict[str, object]) -> None:
    """Without the bicubic skip the output is the plain network output, still x2."""
    model = cls(**kw, residual_prediction=False)
    assert model(torch.rand(1, 1, 16, 16)).shape == (1, 1, 32, 32)


@pytest.mark.parametrize(("cls", "kw"), MODELS, ids=["dfcan", "rcan"])
def test_backward_all_grads_finite(cls: type[torch.nn.Module], kw: dict[str, object]) -> None:
    model = cls(**kw)
    loss = F.l1_loss(model(torch.rand(1, 1, 16, 16)), torch.rand(1, 1, 32, 32))
    loss.backward()
    assert torch.isfinite(loss)
    for param in model.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


@pytest.mark.parametrize(("cls", "kw"), MODELS, ids=["dfcan", "rcan"])
def test_output_finite_and_differs_from_bicubic(
    cls: type[torch.nn.Module], kw: dict[str, object]
) -> None:
    """Outputs on [0, 1] inputs are finite and the network actually changes something."""
    x = torch.rand(1, 1, 16, 16)
    out = cls(**kw)(x)
    assert torch.isfinite(out).all()
    bicubic = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False, antialias=True)
    assert not torch.allclose(out, bicubic)


def test_build_model_returns_requested_types() -> None:
    dfcan = build_model({"name": "dfcan", "nf": 8, "num_groups": 1, "num_blocks": 1})
    rcan = build_model({"name": "rcan", "nf": 16, "num_groups": 1, "num_blocks": 1, "reduction": 4})
    assert isinstance(dfcan, DFCAN)
    assert isinstance(rcan, RCAN)
    assert dfcan.nf == 8
    assert rcan.reduction == 4


def test_build_model_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="valid names"):
        build_model({"name": "unet"})


def test_build_model_rejects_constructor_incompatible_keys() -> None:
    with pytest.raises(TypeError, match="gamma"):
        build_model({"name": "rcan", "gamma": 0.5})


def test_build_model_passes_kwargs_through() -> None:
    model = build_model(
        {
            "name": "dfcan",
            "nf": 16,
            "num_groups": 2,
            "num_blocks": 2,
            "gamma": 0.5,
            "residual_prediction": False,
        }
    )
    assert (model.nf, model.num_groups, model.num_blocks) == (16, 2, 2)
    assert model.gamma == 0.5
    assert model.residual_prediction is False


def test_deterministic_build_and_forward() -> None:
    torch.manual_seed(0)
    model = DFCAN(**DFCAN_KW).eval()
    x = torch.rand(1, 1, 16, 16)
    assert torch.equal(model(x), model(x))
    torch.manual_seed(0)
    rebuilt = DFCAN(**DFCAN_KW).eval()
    assert torch.equal(rebuilt(x), model(x))


def test_bicubic_forward_shape_odd_input() -> None:
    from src.models import BicubicSR

    model = BicubicSR(in_channels=1, scale=2)
    assert model(torch.rand(2, 1, 16, 16)).shape == (2, 1, 32, 32)
    assert model(torch.rand(1, 1, 15, 15)).shape == (1, 1, 30, 30)


def test_bicubic_matches_plain_interpolation() -> None:
    from src.models import BicubicSR

    x = torch.rand(1, 1, 16, 16)
    reference = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False, antialias=True)
    assert torch.allclose(BicubicSR()(x), reference)


def test_build_model_bicubic_entry() -> None:
    model = build_model({"name": "bicubic", "scale": 2})
    assert model(torch.rand(1, 1, 16, 16)).shape == (1, 1, 32, 32)


def test_dfcan_fca_gates_not_saturated() -> None:
    """FCA gates must stay in the sigmoid interior (no hard 0/1 saturation)."""
    from src.models.dfcan import FCAB

    torch.manual_seed(0)
    model = DFCAN(nf=16, num_groups=1, num_blocks=2)
    model.eval()
    collected: list[torch.Tensor] = []
    hooks = [
        block.gate[-1].register_forward_hook(lambda m, i, o: collected.append(o.detach()))
        for block in model.modules()
        if isinstance(block, FCAB)
    ]
    try:
        with torch.no_grad():
            model(torch.rand(2, 1, 16, 16))
        gates = torch.cat([g.flatten() for g in collected])
        assert gates.numel() > 0
        assert gates.min() > 0.01 and gates.max() < 0.99  # smooth interior, not bimodal {0, 1}
    finally:
        for hook in hooks:
            hook.remove()


def test_dfcan_fca_gates_scale_invariant() -> None:
    """Gates must survive extreme input rescaling without saturating.

    Conv biases break exact scale equivariance, so gates are not bit-identical
    across scales - but the per-sample normalization must keep them inside the
    sigmoid interior (pre-fix they collapsed to hard {0, 1} at large scales).
    """
    from src.models.dfcan import FCAB

    torch.manual_seed(0)
    model = DFCAN(nf=16, num_groups=1, num_blocks=2)
    model.eval()
    collected: list[torch.Tensor] = []
    hooks = [
        block.gate[-1].register_forward_hook(lambda m, i, o: collected.append(o.detach()))
        for block in model.modules()
        if isinstance(block, FCAB)
    ]
    try:
        x = torch.rand(1, 1, 16, 16)

        def collect(scale: float) -> torch.Tensor:
            collected.clear()
            with torch.no_grad():
                model(x * scale)
            return torch.cat([g.flatten() for g in collected])

        base = collect(1.0)
        scaled = collect(50.0)  # simulates a much larger spatial/feature amplitude scale
        for gates in (base, scaled):
            assert gates.min() > 0.01 and gates.max() < 0.99
    finally:
        for hook in hooks:
            hook.remove()
