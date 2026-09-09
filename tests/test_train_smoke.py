"""Integration smoke test for train.py (needs sibling src.* modules + conftest fixtures)."""

import json
import math

import pytest


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def _make_cfg(biosr_root, tmp_path) -> dict:
    return {
        "dataset": {
            "root": str(biosr_root),
            "scale": 2,
            "structures": ["Microtubules", "CCPs", "ER", "F-actin"],
            "held_out_structure": "F-actin",
            "signal_levels": "all",
            "patch_size": 16,
            "normalization": "per_image_minmax",
        },
        "model": {
            "name": "dfcan",
            "nf": 8,
            "num_groups": 1,
            "num_blocks": 1,
            "scale": 2,
            "gamma": 1.0,
            "residual_prediction": True,
        },
        "training": {
            "seed": 42,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "epochs": 1,
            "num_workers": 0,
            "val_fraction": 0.34,
            "val_every": 1,
            "weight_decay": 0.0,
            "grad_clip": None,
            "device": "cpu",
        },
        "loss": {
            "charbonnier": 1.0,
            "ssim": 0.1,
            "gradient": 0.1,
            "fourier": 0.0,
            "charbonnier_eps": 1e-3,
            "ssim_window": 11,
        },
        "augmentation": {
            "hflip": True,
            "vflip": True,
            "rot90": True,
            "intensity_range": [0.9, 1.1],
            "noise_max_sigma": 0.0,
            "morphology_ood": False,
        },
        "eval": {
            "tile": 16,
            "overlap": 4,
            "edge_percentile": 99.0,
            "tolerance_px": 1,
            "frc_threshold": 0.14285714285714285,
            "ssim_window": 11,
        },
        "output_dir": str(tmp_path / "runs"),
    }


def test_training_smoke(biosr_root, manifest, tmp_path):
    from train import run_training

    assert len(manifest) > 0  # fixture sanity: discovery ran
    cfg = _make_cfg(biosr_root, tmp_path)
    run_dir = tmp_path / "run"
    summary = run_training(cfg, run_dir=run_dir)

    assert math.isfinite(summary["best_val_psnr"])
    assert math.isfinite(summary["best_val_ssim"])
    assert summary["config"]["dataset"]["root"] == str(biosr_root)

    level5 = summary["test"]["per_level"]["5"]
    assert level5["n"] == 2
    for key in ("psnr", "ssim", "structural_precision", "structural_recall"):
        assert math.isfinite(level5[key]["mean"])

    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "split.json").is_file()
    assert (run_dir / "checkpoints" / "best.pt").is_file()
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert (run_dir / "logs" / "train_log.jsonl").is_file()
    assert (run_dir / "results" / "test_metrics.json").is_file()
    split_info = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    assert split_info["held_out"] == "F-actin"


def test_morphology_ood_raises(biosr_root, manifest, tmp_path):
    from train import run_training

    cfg = _make_cfg(biosr_root, tmp_path)
    cfg["augmentation"]["morphology_ood"] = True
    with pytest.raises(NotImplementedError, match="morphology-OOD"):
        run_training(cfg, run_dir=tmp_path / "run_ood")


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_run_validation_on_cuda(biosr_root, manifest, tmp_path):
    """Validation metrics must not mix devices: gt moves to the model's device."""
    import math

    import torch
    from torch.utils.data import DataLoader

    from src.data.biosr import BioSRDataset
    from train import _run_validation, build_split

    cfg = _make_cfg(biosr_root, tmp_path)
    split = build_split(cfg)
    val_ds = BioSRDataset(split.val, root=biosr_root, patch_size=16, crop_mode="center")
    loader = DataLoader(val_ds, batch_size=2, shuffle=False)
    device = torch.device("cuda")
    model = torch.nn.UpsamplingBilinear2d(scale_factor=2).to(device)
    val_psnr, val_ssim = _run_validation(model, loader, device, cfg["eval"]["ssim_window"])
    assert math.isfinite(val_psnr)
    assert math.isfinite(val_ssim)


def test_bicubic_eval_only_smoke(biosr_root, manifest, tmp_path):
    """epochs=0 + bicubic: no training, straight to the standard held-out evaluation."""
    from train import run_training

    cfg = _make_cfg(biosr_root, tmp_path)
    cfg["model"] = {"name": "bicubic", "scale": 2}
    cfg["training"]["epochs"] = 0
    run_dir = tmp_path / "run_bicubic"
    summary = run_training(cfg, run_dir=run_dir)

    level5 = summary["test"]["per_level"]["5"]
    assert level5["n"] == 2
    for key in ("psnr", "ssim", "structural_precision", "structural_recall"):
        assert math.isfinite(level5[key]["mean"])
    assert (run_dir / "results" / "test_metrics.json").is_file()


def test_visualization_smoke(biosr_root, manifest, tmp_path):
    """vis_every=1 -> grayscale panels: 2 test samples per level + 1 per seen structure."""
    import tifffile

    from train import run_training

    cfg = _make_cfg(biosr_root, tmp_path)
    cfg["training"]["vis_every"] = 1
    run_dir = tmp_path / "run_vis"
    run_training(cfg, run_dir=run_dir)
    # held_out=F-actin: test level 5 -> 2 panels; seen: MT+CCPs (val), ER (train) -> 3 panels
    for epoch in ("0000", "0001"):
        panels = list((run_dir / "vis" / f"epoch_{epoch}").glob("*.tif"))
        assert len(panels) == 5
        for panel in panels:
            stack = tifffile.imread(panel)  # 3 grayscale pages, NOT an RGB composite
            assert stack.ndim == 3 and stack.shape[0] == 3
            assert 0.0 <= float(stack.min()) and float(stack.max()) <= 1.0


def test_wandb_missing_package_is_graceful(biosr_root, manifest, tmp_path):
    """wandb enabled=true must never break training (missing package -> warn + continue;
    installed package -> offline run inside the run dir)."""
    from train import run_training

    cfg = _make_cfg(biosr_root, tmp_path)
    cfg["wandb"] = {"enabled": True, "entity": "unit", "project": "unit-test", "mode": "offline"}
    summary = run_training(cfg, run_dir=tmp_path / "run_wandb")
    assert math.isfinite(summary["best_val_psnr"])
