"""Exponential moving average (EMA) of model weights for evaluation.

Keeps a shadow copy of ``model`` whose parameters follow
``ema = decay * ema + (1 - decay) * param``. The effective decay ramps from
``1/10`` towards ``decay`` as ``(1 + step) / (10 + step)`` so early updates
track the source model closely. Buffers (e.g. BatchNorm running stats) are
copied verbatim, never averaged.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys


class ModelEMA(nn.Module):
    """Shadow copy of ``model`` holding its exponential moving average.

    The averaged model is exposed as ``self.module``; it stays in eval mode
    with gradients disabled, so it can be used for validation/inference as-is.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: int = 0) -> None:
        super().__init__()
        self.module = deepcopy(model)
        self.module.requires_grad_(False)
        self.module.eval()
        self.decay = decay
        self.warmup = warmup
        self.step = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend ``model`` into the EMA; buffers are copied, not averaged."""
        self.step += 1
        decay = min(self.decay, (1 + self.step) / (10 + self.step))
        for ema_param, param in zip(self.module.parameters(), model.parameters(), strict=True):
            ema_param.mul_(decay).add_(param, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(self.module.buffers(), model.buffers(), strict=True):
            ema_buffer.copy_(buffer)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Copy the EMA parameters and buffers back into ``model``."""
        for ema_param, param in zip(self.module.parameters(), model.parameters(), strict=True):
            param.copy_(ema_param)
        for ema_buffer, buffer in zip(self.module.buffers(), model.buffers(), strict=True):
            buffer.copy_(ema_buffer)

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Torch-style state dict plus the EMA ``decay`` and ``step`` counters."""
        state: dict[str, Any] = dict(super().state_dict(*args, **kwargs))
        state["decay"] = self.decay
        state["step"] = self.step
        return state

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> _IncompatibleKeys:
        """Restore module weights plus ``decay``/``step`` from ``state_dict``."""
        state = dict(state_dict)
        self.decay = float(state.pop("decay"))
        self.step = int(state.pop("step"))
        return super().load_state_dict(state, strict=strict, assign=assign)
