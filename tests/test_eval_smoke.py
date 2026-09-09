"""Integration smoke test for evaluate.py (needs sibling src.* modules + conftest fixtures)."""

import json
import math


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


def test_evaluation_smoke(biosr_root, manifest, tmp_path):
    from evaluate import run_evaluation
    from src.models import build_model
    from src.utils.run import save_checkpoint

    assert len(manifest) > 0  # fixture sanity: discovery ran
    cfg = _make_cfg(biosr_root, tmp_path)
    checkpoint = tmp_path / "ckpt" / "best.pt"
    save_checkpoint(checkpoint, build_model(cfg["model"]), cfg, epoch=0, metrics={})

    results = run_evaluation(cfg, checkpoint, run_dir=tmp_path / "evalrun")

    assert results["per_level"]["5"]["n"] == 2
    assert math.isfinite(results["overall"]["psnr"]["mean"])
    results_file = tmp_path / "evalrun" / "results" / "test_metrics.json"
    assert results_file.is_file()
    saved = json.loads(results_file.read_text(encoding="utf-8"))
    assert saved["held_out_structure"] == "F-actin"
