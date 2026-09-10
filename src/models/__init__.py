"""Model zoo: DFCAN/RCAN/NAFNet/SwinIR backbones, bicubic reference, and the config factory."""

import inspect

import torch

from src.models.bicubic import BicubicSR
from src.models.dfcan import DFCAN
from src.models.nafnet import NAFNet
from src.models.rcan import RCAN
from src.models.swinir import SwinIR

__all__ = ["BicubicSR", "DFCAN", "NAFNet", "RCAN", "SwinIR", "build_model"]

_REGISTRY: dict[str, type[torch.nn.Module]] = {
    "dfcan": DFCAN,
    "rcan": RCAN,
    "nafnet": NAFNet,
    "swinir": SwinIR,
    "bicubic": BicubicSR,
}


def build_model(model_cfg: dict) -> torch.nn.Module:
    """Instantiate a model from a config dict.

    ``model_cfg["name"]`` selects "dfcan", "rcan", "nafnet", "swinir", or "bicubic"; the remaining
    keys are forwarded to the matching constructor (e.g. "gamma" for DFCAN, "reduction" for RCAN).
    An unknown name raises ValueError; keys the constructor does not accept raise TypeError.
    """
    cfg = dict(model_cfg)
    name = cfg.pop("name", None)
    if name not in _REGISTRY:
        valid = ", ".join(repr(n) for n in sorted(_REGISTRY))
        raise ValueError(f"Unknown model name {name!r}; valid names: {valid}")
    cls = _REGISTRY[name]
    accepted = [p for p in inspect.signature(cls).parameters if p != "self"]
    kwargs: dict[str, object] = {}
    for key, value in cfg.items():
        if key not in accepted:
            raise TypeError(f"{cls.__name__} does not accept {key!r}; accepted kwargs: {accepted}")
        kwargs[key] = value
    return cls(**kwargs)
