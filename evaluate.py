"""Evaluation entry point: load a checkpoint, score the held-out structure, save metrics."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from src.data.biosr import resolve_root

from src.models import build_model
from src.utils.config import apply_overrides, load_config
from src.utils.run import load_checkpoint
from train import build_split, evaluate_split, resolve_device


def run_evaluation(cfg: dict, checkpoint: str | Path, run_dir: Path | None = None) -> dict:
    """Evaluate `checkpoint` on cfg's LOSO test split; write and return metric aggregates.

    Results go to <run_dir>/results/test_metrics.json; when run_dir is None they
    are written next to the checkpoint (<checkpoint>/../results/).
    """
    device = resolve_device(cfg["training"]["device"])
    model = build_model(cfg["model"]).to(device)
    state = load_checkpoint(checkpoint, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    split = build_split(cfg)
    root = resolve_root(cfg["dataset"]["root"])
    results = evaluate_split(model, split, root, cfg, device)

    if run_dir is None:
        run_dir = Path(checkpoint).resolve().parent.parent
    results_path = Path(run_dir) / "results" / "test_metrics.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"held_out_structure": split.held_out, **results}
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: python evaluate.py --config <run>/config.yaml --checkpoint <run>/checkpoints/best.pt."""
    parser = argparse.ArgumentParser(description="Evaluate a trained SR checkpoint on the held-out structure")
    parser.add_argument("--config", required=True, help="path to the run's YAML config")
    parser.add_argument("--checkpoint", required=True, help="path to a checkpoint saved by training")
    parser.add_argument("overrides", nargs="*", help="dotted.key=value config overrides")
    args = parser.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.overrides)
    run_evaluation(cfg, args.checkpoint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
