"""Optional wandb experiment tracking with a graceful no-op fallback.

wandb is an optional dependency: when the package is missing or tracking is
disabled in the config, every method silently no-ops so training never breaks.
Enable via a top-level config section:

    wandb:
      enabled: true
      entity: 2991837391-beihang-university
      project: Restore
      mode: online            # "online" on the server, "offline" for local runs
"""

from __future__ import annotations

import numpy as np

try:
    import wandb
except ImportError:  # environment-dependent; tracking degrades to a no-op
    wandb = None


def wandb_available() -> bool:
    """True when the wandb package is importable."""
    return wandb is not None


class WandbTracker:
    """Config-driven wandb wrapper: init, metrics, image panels, summary, finish."""

    def __init__(self, cfg: dict, run_dir=None, name: str | None = None) -> None:
        wb_cfg = dict(cfg.get("wandb") or {})
        self.enabled = bool(wb_cfg.get("enabled", False))
        self.run = None
        if not self.enabled:
            return
        if wandb is None:
            print(
                "[wandb] package not installed - tracking disabled "
                "(pip install wandb, then wandb login, to enable)",
                flush=True,
            )
            self.enabled = False
            return
        self.run = wandb.init(
            entity=wb_cfg.get("entity"),
            project=wb_cfg.get("project", "Restore"),
            name=name,
            mode=wb_cfg.get("mode", "online"),
            config=cfg,
            dir=str(run_dir) if run_dir is not None else None,
        )

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Log scalar metrics at `step` (None values are dropped)."""
        if self.run is None:
            return
        payload = {key: value for key, value in metrics.items() if value is not None}
        self.run.log(payload, step=step)

    def log_images(self, key: str, images: list[tuple[np.ndarray, str]], step: int) -> None:
        """Log image panels; `images` holds (H, W, C) float arrays in [0, 1] + captions."""
        if self.run is None or not images:
            return
        payload = {key: [wandb.Image(image, caption=caption) for image, caption in images]}
        self.run.log(payload, step=step)

    def update_summary(self, summary: dict) -> None:
        """Attach final metrics to the run summary (shown in the wandb table)."""
        if self.run is not None:
            self.run.summary.update(summary)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None
