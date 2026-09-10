"""Sharpness-Aware Minimization (SAM) optimizer wrapper.

Implements the two-step SAM update of Foret et al. (arXiv:2010.01412) following
the reference implementation at https://github.com/davda54/sam. ``first_step``
perturbs the weights by ``e_w`` along the (normalized) gradient direction to
reach a sharpness-maximizing point; ``second_step`` restores the original
weights and applies the base optimizer update computed at that perturbed point.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch.optim.optimizer import Optimizer, ParamsT


class SAM(Optimizer):
    """SAM wrapper around a base optimizer class (default: ``torch.optim.Adam``)."""

    def __init__(
        self,
        params: ParamsT,
        base_optimizer_cls: type[Optimizer] = torch.optim.Adam,
        rho: float = 0.05,
        **kwargs: Any,
    ) -> None:
        if rho < 0.0:
            raise ValueError(f"invalid rho, should be non-negative: {rho}")
        super().__init__(params, {"rho": rho})
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    def _grad_norm(self) -> torch.Tensor:
        """Global L2 norm of the gradients currently attached to the parameters."""
        grads = [
            p.grad.detach()
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not grads:
            return torch.zeros((), device=self.param_groups[0]["params"][0].device)
        device = grads[0].device
        return torch.norm(torch.stack([grad.norm(p=2).to(device) for grad in grads]), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Climb to the local maximum ``w + e_w`` and stash ``old_p`` for restore."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Restore the original weights, then apply the base optimizer step."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure: Callable[[], float] | None = None) -> None:
        """Unsupported: SAM requires the explicit two-step API."""
        raise NotImplementedError(
            "SAM requires the explicit two-step API: call first_step() after the first "
            "backward pass, recompute gradients, then call second_step()"
        )

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Forward gradient clearing to the wrapped base optimizer."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Return the base optimizer state (SAM itself holds no optimizer state)."""
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore the base optimizer state and re-point the shared param groups."""
        self.base_optimizer.load_state_dict(dict(state_dict))
        self.param_groups = self.base_optimizer.param_groups
