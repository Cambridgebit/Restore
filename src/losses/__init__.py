"""Loss package: composable components plus a config-driven weighted-sum factory.

AGENT.md §8: all loss weights come from config; COMPONENT_DEFAULTS are the fallbacks
applied when a config omits a component weight.
"""

from torch import Tensor, nn

from src.losses.consistency import DataConsistencyLoss, ResidualLoss
from src.losses.frequency import FocalFrequencyLoss, FourierLoss
from src.losses.pixel import CharbonnierLoss
from src.losses.structural import (
    GradientLoss,
    GradientVarianceLoss,
    HessianStructureLoss,
    SSIMLoss,
    ssim_index,
)

COMPONENT_DEFAULTS: dict[str, float] = {
    "charbonnier": 1.0,
    "ssim": 0.1,
    "gradient": 0.1,
    "fourier": 0.0,
    "data_consistency": 0.0,
    "residual": 0.0,
    "focal_frequency": 0.0,
    "gradient_variance": 0.0,
    "hessian": 0.0,
}


class WeightedSumLoss(nn.Module):
    """Weighted sum of loss components; only components with weight > 0 are computed.

    `forward` returns (total, parts) where parts maps each computed component name to
    its UNWEIGHTED value, plus parts["total"] == total. The optional `lr` tensor is
    passed through to every component; components that do not use it ignore it.
    """

    def __init__(
        self,
        weights: dict[str, float],
        charbonnier_eps: float = 1e-3,
        ssim_window: int = 11,
        scale: int = 2,
    ) -> None:
        super().__init__()
        unknown = sorted(set(weights) - set(COMPONENT_DEFAULTS))
        if unknown:
            msg = f"Unknown loss component key(s): {unknown}; valid keys: {sorted(COMPONENT_DEFAULTS)}"
            raise ValueError(msg)
        self.weights: dict[str, float] = {**COMPONENT_DEFAULTS, **weights}
        self.component_names: tuple[str, ...] = tuple(
            name for name in COMPONENT_DEFAULTS if self.weights[name] > 0.0
        )
        self._components = nn.ModuleDict(
            {
                "charbonnier": CharbonnierLoss(eps=charbonnier_eps),
                "ssim": SSIMLoss(window=ssim_window),
                "gradient": GradientLoss(),
                "fourier": FourierLoss(),
                "data_consistency": DataConsistencyLoss(scale=scale),
                "residual": ResidualLoss(scale=scale),
                "focal_frequency": FocalFrequencyLoss(),
                "gradient_variance": GradientVarianceLoss(),
                "hessian": HessianStructureLoss(),
            }
        )

    def forward(
        self, pred: Tensor, target: Tensor, lr: Tensor | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return (total, parts) with parts["total"] == total; `lr` is optional."""
        parts: dict[str, Tensor] = {}
        total = pred.new_zeros(())
        for name in self.component_names:
            value = self._components[name](pred, target, lr=lr)
            parts[name] = value
            total = total + self.weights[name] * value
        parts["total"] = total
        return total, parts


def build_loss(loss_cfg: dict, scale: int = 2) -> WeightedSumLoss:
    """Build a WeightedSumLoss from a config dict.

    Recognized keys: component weights (any subset of COMPONENT_DEFAULTS; missing keys
    fall back to the defaults) plus "charbonnier_eps" (default 1e-3) and
    "ssim_window" (default 11). Unknown keys raise ValueError. `scale` is the SR
    factor, forwarded to the data-consistency and residual components.
    """
    valid_keys = set(COMPONENT_DEFAULTS) | {"charbonnier_eps", "ssim_window"}
    unknown = sorted(set(loss_cfg) - valid_keys)
    if unknown:
        msg = f"Unknown loss config key(s): {unknown}; valid keys: {sorted(valid_keys)}"
        raise ValueError(msg)
    weights = {name: loss_cfg[name] for name in COMPONENT_DEFAULTS if name in loss_cfg}
    return WeightedSumLoss(
        weights,
        charbonnier_eps=loss_cfg.get("charbonnier_eps", 1e-3),
        ssim_window=loss_cfg.get("ssim_window", 11),
        scale=scale,
    )


__all__ = [
    "COMPONENT_DEFAULTS",
    "CharbonnierLoss",
    "DataConsistencyLoss",
    "FocalFrequencyLoss",
    "FourierLoss",
    "GradientLoss",
    "GradientVarianceLoss",
    "HessianStructureLoss",
    "ResidualLoss",
    "SSIMLoss",
    "WeightedSumLoss",
    "build_loss",
    "ssim_index",
]
