"""Run directory management, checkpointing, JSONL logging, and git metadata."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import torch


def create_run_dir(base: str | Path, prefix: str) -> Path:
    """Create <base>/<YYYYmmdd-HHMMSS>_<prefix>/; append -1, -2, ... on collisions."""
    base_path = Path(base)
    base_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_path / f"{stamp}_{prefix}"
    suffix = 0
    while run_dir.exists():
        suffix += 1
        run_dir = base_path / f"{stamp}_{prefix}-{suffix}"
    run_dir.mkdir()
    return run_dir


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    cfg: dict,
    epoch: int,
    metrics: dict,
) -> None:
    """Save model state dict plus config/epoch/metrics/git hash for reproducibility."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "metrics": metrics,
            "git_hash": git_hash(),
        },
        p,
    )


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    """Load a checkpoint written by save_checkpoint."""
    return torch.load(path, map_location=map_location, weights_only=False)


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append one record as a JSON line, creating parent directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def git_hash() -> str | None:
    """Return `git rev-parse HEAD` for the cwd, or None on any failure. Never raises."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
