"""Training entry point: BioSR leave-one-structure-out super-resolution.

A simple config-driven loop (no trainer frameworks). Discovers BioSR samples,
builds a leakage-checked LOSO split, trains with structure-balanced sampling,
tracks the best validation PSNR, and finally evaluates on the unseen structure.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.biosr import BioSRDataset, discover_samples, resolve_root
from src.data.sampling import make_balanced_sampler
from src.data.splits import Split, check_no_leakage, filter_samples, make_loso_split
from src.losses import build_loss
from src.metrics import frc_resolution, psnr, ssim, structural_precision_recall, zncc
from src.models import build_model
from src.utils.config import apply_overrides, load_config, save_config
from src.utils.inference import predict_tiled
from src.utils.run import append_jsonl, create_run_dir, save_checkpoint
from src.utils.seed import seed_worker, set_seed

REQUIRED_SECTIONS = ("dataset", "model", "training", "loss", "augmentation", "eval")
_RECORD_METRICS = ("psnr", "ssim", "zncc", "frc", "precision", "recall", "f1")
_AGGREGATE_NAMES = {"precision": "structural_precision", "recall": "structural_recall", "f1": "structural_f1"}


def validate_cfg(cfg: dict) -> None:
    """Raise on missing sections or unimplemented features."""
    missing = [section for section in REQUIRED_SECTIONS if section not in cfg]
    if missing:
        raise ValueError(f"config missing required sections: {', '.join(missing)}")
    if cfg["training"].get("val_every", 1) < 1:
        raise ValueError("training.val_every must be >= 1")
    if cfg["augmentation"].get("morphology_ood", False):
        raise NotImplementedError(
            "morphology-OOD augmentation is reserved for a later ablation; "
            "not implemented in this framework"
        )


def resolve_device(name: str) -> torch.device:
    """Map 'auto' | 'cuda' | 'cpu' to a torch.device."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name in ("cuda", "cpu"):
        return torch.device(name)
    raise ValueError(f"unsupported device {name!r}; expected 'auto', 'cuda', or 'cpu'")


def build_split(cfg: dict) -> Split:
    """Discover, filter, and LOSO-split samples exactly as cfg specifies."""
    root = resolve_root(cfg["dataset"]["root"])
    samples = discover_samples(root)
    levels = cfg["dataset"]["signal_levels"]
    filtered = filter_samples(
        samples,
        structures=cfg["dataset"]["structures"],
        levels=None if levels == "all" else list(levels),
    )
    split = make_loso_split(
        filtered,
        held_out=cfg["dataset"]["held_out_structure"],
        val_fraction=cfg["training"]["val_fraction"],
        seed=cfg["training"]["seed"],
    )
    check_no_leakage(split)
    return split


def aggregate_results(records: list[dict]) -> dict:
    """Aggregate per-image metric records into per-level and overall summaries.

    Each group gets {"n", <metric>: {"mean", "std"} (population std), "records"}.
    """
    def summarize(recs: list[dict]) -> dict:
        summary: dict = {"n": len(recs)}
        for name in _RECORD_METRICS:
            values = [record[name] for record in recs]
            summary[_AGGREGATE_NAMES.get(name, name)] = {
                "mean": statistics.fmean(values) if values else None,
                "std": statistics.pstdev(values) if values else None,
            }
        summary["records"] = recs
        return summary

    levels = sorted({record["level"] for record in records})
    per_level = {str(level): summarize([r for r in records if r["level"] == level]) for level in levels}
    return {"per_level": per_level, "overall": summarize(records)}


@torch.no_grad()
def _run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    ssim_window: int,
) -> tuple[float, float]:
    """Center-crop-patch validation; returns sample-weighted mean (psnr, ssim)."""
    model.eval()
    psnr_sum = 0.0
    ssim_sum = 0.0
    seen = 0
    for lr_img, gt in loader:
        sr = model(lr_img.to(device))
        gt = gt.to(device)
        batch = sr.shape[0]
        psnr_sum += psnr(sr, gt) * batch
        ssim_sum += ssim(sr, gt, window=ssim_window) * batch
        seen += batch
    model.train()
    if seen == 0:
        return float("nan"), float("nan")
    return psnr_sum / seen, ssim_sum / seen


def evaluate_split(
    model: torch.nn.Module,
    split: Split,
    root: str | Path,
    cfg: dict,
    device: torch.device,
) -> dict:
    """Tiled inference + metrics on split.test; returns aggregate_results() output."""
    model.eval()
    eval_cfg = cfg["eval"]
    dataset = BioSRDataset(
        split.test,
        root=root,
        patch_size=None,
        crop_mode=None,
        augment=None,
        normalization=cfg["dataset"]["normalization"],
    )
    records: list[dict] = []
    for index, sample in enumerate(split.test):
        lr_img, gt = dataset[index]
        # dataset yields (1, H, W) per sample; tiled inference and metrics expect (N, 1, H, W)
        sr = predict_tiled(
            model,
            lr_img.unsqueeze(0).to(device),
            tile=eval_cfg["tile"],
            overlap=eval_cfg["overlap"],
            scale=cfg["dataset"]["scale"],
        ).cpu()
        gt = gt.unsqueeze(0)
        spr = structural_precision_recall(
            sr,
            gt,
            tolerance_px=eval_cfg["tolerance_px"],
            percentile=eval_cfg["edge_percentile"],
        )
        records.append(
            {
                "id": sample.id,
                "level": sample.level,
                "psnr": float(psnr(sr, gt)),
                "ssim": float(ssim(sr, gt, window=eval_cfg["ssim_window"])),
                "zncc": float(zncc(sr, gt)),
                "frc": float(frc_resolution(sr, gt, threshold=eval_cfg["frc_threshold"])),
                "precision": float(spr["precision"]),
                "recall": float(spr["recall"]),
                "f1": float(spr["f1"]),
            }
        )
    return aggregate_results(records)


def run_training(cfg: dict, run_dir: Path | None = None) -> dict:
    """Train per cfg; returns summary with best epoch, val metrics, and test aggregates."""
    cfg = copy.deepcopy(cfg)
    validate_cfg(cfg)
    set_seed(cfg["training"]["seed"])
    device = resolve_device(cfg["training"]["device"])

    root = resolve_root(cfg["dataset"]["root"])
    cfg["dataset"]["root"] = str(root)
    split = build_split(cfg)

    if run_dir is None:
        run_dir = create_run_dir(
            cfg["output_dir"],
            prefix=f"{cfg['model']['name']}_{cfg['dataset']['held_out_structure']}",
        )
    else:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    split.save_json(run_dir / "split.json")

    aug = cfg["augmentation"]
    intensity_range = aug.get("intensity_range")
    train_ds = BioSRDataset(
        split.train,
        root=root,
        patch_size=cfg["dataset"]["patch_size"],
        crop_mode="random",
        augment={
            "hflip": aug["hflip"],
            "vflip": aug["vflip"],
            "rot90": aug["rot90"],
            "intensity_range": list(intensity_range) if intensity_range is not None else None,
            "noise_max_sigma": aug["noise_max_sigma"],
        },
        normalization=cfg["dataset"]["normalization"],
    )
    val_ds = BioSRDataset(
        split.val,
        root=root,
        patch_size=cfg["dataset"]["patch_size"],
        crop_mode="center",
        normalization=cfg["dataset"]["normalization"],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        sampler=make_balanced_sampler([s.structure for s in split.train], seed=cfg["training"]["seed"]),
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        worker_init_fn=seed_worker,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )

    model = build_model(cfg["model"]).to(device)
    loss_fn = build_loss(cfg["loss"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    grad_clip = cfg["training"].get("grad_clip")

    best_epoch = 0
    best_val_psnr = float("-inf")
    best_val_ssim = float("-inf")
    log_path = run_dir / "logs" / "train_log.jsonl"

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        part_sums: dict[str, float] = defaultdict(float)
        total_sum = 0.0
        seen = 0
        for lr_img, gt in train_loader:
            sr = model(lr_img.to(device))
            total, parts = loss_fn(sr, gt.to(device))
            total.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            batch = lr_img.shape[0]
            seen += batch
            total_sum += total.item() * batch
            for name, value in parts.items():
                part_sums[name] += value.item() * batch

        record: dict = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": total_sum / seen,
        }
        record.update({f"train/{name}": part_sum / seen for name, part_sum in part_sums.items()})
        val_psnr: float | None = None
        val_ssim: float | None = None
        if epoch % cfg["training"]["val_every"] == 0:
            val_psnr, val_ssim = _run_validation(model, val_loader, device, cfg["eval"]["ssim_window"])
            record["val_psnr"] = val_psnr
            record["val_ssim"] = val_ssim
            if val_psnr > best_val_psnr:
                best_epoch, best_val_psnr, best_val_ssim = epoch, val_psnr, val_ssim
                save_checkpoint(
                    run_dir / "checkpoints" / "best.pt",
                    model,
                    cfg,
                    epoch,
                    {"val_psnr": val_psnr, "val_ssim": val_ssim},
                )
        else:
            record["val_psnr"] = None
            record["val_ssim"] = None
        append_jsonl(log_path, record)
        save_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            model,
            cfg,
            epoch,
            {"val_psnr": val_psnr, "val_ssim": val_ssim},
        )

    test_results = evaluate_split(model, split, root, cfg, device)
    results_path = run_dir / "results" / "test_metrics.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"held_out_structure": split.held_out, **test_results}
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_val_psnr": best_val_psnr,
        "best_val_ssim": best_val_ssim,
        "test": test_results,
        "config": cfg,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: python train.py --config configs/dfcan.yaml [key.subkey=value ...]."""
    parser = argparse.ArgumentParser(description="Train a BioSR SR model with the LOSO protocol")
    parser.add_argument("--config", required=True, help="path to the YAML config")
    parser.add_argument("overrides", nargs="*", help="dotted.key=value config overrides")
    args = parser.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.overrides)
    summary = run_training(cfg)
    print(f"run directory: {summary['run_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
