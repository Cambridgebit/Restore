"""Integration test: every shipped config builds via build_model and produces a 2x output."""

from pathlib import Path

import pytest
import torch
import yaml

from src.models import build_model

_CONFIGS = [
    "dfcan.yaml",
    "rcan.yaml",
    "nafnet.yaml",
    "swinir.yaml",
    "mambair.yaml",
    "wavemixsr.yaml",
    "restormer.yaml",
    "flowmatching.yaml",
    "bicubic.yaml",
]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("filename", _CONFIGS)
def test_registry_builds_and_forwards(filename):
    cfg = yaml.safe_load((_CONFIG_DIR / filename).read_text(encoding="utf-8"))
    model = build_model(cfg["model"]).eval()
    lr = torch.rand(1, 1, 16, 16)
    with torch.no_grad():
        if cfg["model"]["name"] == "flow_matching":
            sr = model.sample(lr, steps=2)
        else:
            sr = model(lr)
    assert sr.shape == (1, 1, 32, 32)
    assert torch.isfinite(sr).all()
