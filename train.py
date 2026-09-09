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

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.biosr import BioSRDataset, BioSRSample, discover_samples, resolve_root
from src.data.sampling import make_balanced_sampler
from src.data.splits import Split, check_no_leakage, filter_samples, make_loso_split
from src.losses import build_loss
from src.metrics import frc_resolution, psnr, ssim, structural_precision_recall, zncc
from src.models import build_model
from src.utils.config import apply_overrides, load_config, save_config
from src.utils.inference import predict_tiled
from src.utils.run import append_jsonl, create_run_dir, save_checkpoint
from src.utils.seed import seed_worker, set_seed
from src.utils.tracking import WandbTracker

try:
    from tqdm import tqdm
except ImportError:  # optional: progress bars degrade to plain iteration
    tqdm = None

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


def _select_vis_samples(split: Split) -> list[tuple[str, BioSRSample]]:
    """Fixed visualization picks: 2 test samples per level + 1 sample per seen structure."""
    picks: list[tuple[str, BioSRSample]] = []
    by_level: dict[int, list[BioSRSample]] = {}
    for sample in split.test:
        by_level.setdefault(sample.level, []).append(sample)
    for level in sorted(by_level):
        for sample in by_level[level][:2]:
            picks.append((f"test/level_{level:02d}", sample))
    covered: set[str] = set()
    for pool in (split.val, split.train):  # val first; train only fills structures val misses
        for sample in pool:
            if sample.structure not in covered:
                covered.add(sample.structure)
                picks.append((f"seen/{sample.structure}", sample))
    return picks


def _visualize(
    model: torch.nn.Module,
    split: Split,
    root: str | Path,
    cfg: dict,
    device: torch.device,
    epoch: int,
    tracker: WandbTracker,
    run_dir: Path,
) -> None:
    """Tiled inference on fixed samples; saves (LR-up | SR | GT) TIFFs and wandb panels.

    The unseen (held-out) structure gets 2 samples per signal level; every seen
    structure gets one sample (val preferred, train as fallback).
    """
    model.eval()
    eval_cfg = cfg["eval"]
    vis_dir = run_dir / "vis" / f"epoch_{epoch:04d}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    unseen: list[tuple[np.ndarray, str]] = []
    seen: list[tuple[np.ndarray, str]] = []
    with torch.no_grad():
        for tag, sample in _select_vis_samples(split):
            dataset = BioSRDataset([sample], root=root)
            lr_img, gt_img = dataset[0]
            sr = predict_tiled(
                model,
                lr_img.unsqueeze(0).to(device),
                tile=eval_cfg["tile"],
                overlap=eval_cfg["overlap"],
                scale=cfg["dataset"]["scale"],
            ).cpu()
            lr_up = F.interpolate(
                lr_img.unsqueeze(0), size=sr.shape[-2:], mode="bicubic", align_corners=False, antialias=True
            )
            panel = torch.cat([lr_up, sr, gt_img.unsqueeze(0)], dim=1).squeeze(0).numpy()  # (3, H, W)
            panel = np.clip(panel, 0.0, 1.0)
            tifffile.imwrite(
                vis_dir / f"{sample.id.replace('/', '__')}.tif",
                panel,
                photometric="minisblack",
                planarconfig="separate",
                metadata={"axes": "CYX", "Description": f"epoch {epoch} | {tag} | {sample.id} | LR-up | SR | GT"},
            )
            image = np.transpose(panel, (1, 2, 0))
            caption = f"{tag} | {sample.id} | LR-up | SR | GT"
            (unseen if tag.startswith("test/") else seen).append((image, caption))
    tracker.log_images(f"vis/unseen_{split.held_out}", unseen, step=epoch)
    tracker.log_images("vis/seen_structures", seen, step=epoch)
    print(f"[vis] epoch {epoch}: {len(unseen) + len(seen)} panels -> {vis_dir}", flush=True)
    model.train()


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
    tracker = WandbTracker(
        cfg,
        run_dir=run_dir,
        name=f"{cfg['model']['name']}_{cfg['dataset']['held_out_structure']}",
    )

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
    trainable = [p for p in model.parameters() if p.requires_grad]
    if cfg["training"]["epochs"] > 0 and not trainable:
        raise ValueError("no trainable parameters; an epochs>0 run requires a learning-based model")
    optimizer = (
        torch.optim.Adam(
            trainable,
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        if trainable
        else None
    )
    grad_clip = cfg["training"].get("grad_clip")

    best_epoch = 0
    best_val_psnr = float("-inf")
    best_val_ssim = float("-inf")
    log_path = run_dir / "logs" / "train_log.jsonl"
    epochs_total = cfg["training"]["epochs"]
    vis_every = cfg["training"].get("vis_every", 30)
    if epochs_total > 0:
        _visualize(model, split, root, cfg, device, 0, tracker, run_dir)  # baseline before training

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        part_sums: dict[str, float] = defaultdict(float)
        total_sum = 0.0
        seen = 0
        iterator = (
            tqdm(train_loader, desc=f"epoch {epoch}/{epochs_total}", leave=False, ncols=110)
            if tqdm is not None
            else train_loader
        )
        for lr_img, gt in iterator:
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
            if tqdm is not None:
                iterator.set_postfix(loss=f"{total.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

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
        tracker.log_metrics(record, step=epoch)
        progress = f"[epoch {epoch}/{epochs_total}] train_loss={record['train_loss']:.4f}"
        if record.get("val_psnr") is not None:
            progress += f"  val_psnr={record['val_psnr']:.2f}  val_ssim={record['val_ssim']:.3f}"
        print(progress, flush=True)
        if vis_every > 0 and (epoch % vis_every == 0 or epoch == epochs_total):
            _visualize(model, split, root, cfg, device, epoch, tracker, run_dir)
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
    tracker.update_summary(
        {
            f"test/{name}": test_results["overall"][name]["mean"]
            for name in ("psnr", "ssim", "zncc", "frc", *_AGGREGATE_NAMES.values())
        }
    )
    tracker.finish()
    print(
        f"[done] best_epoch={best_epoch} best_val_psnr={best_val_psnr:.3f} "
        f"test_overall_psnr={test_results['overall']['psnr']['mean']:.3f}",
        flush=True,
    )

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
