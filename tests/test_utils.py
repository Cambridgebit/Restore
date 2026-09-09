"""Standalone unit tests for src.utils (no sibling modules required)."""

import datetime
import json
from pathlib import Path

import pytest
import torch

from src.utils import run as run_mod
from src.utils.config import apply_overrides, load_config, save_config
from src.utils.run import append_jsonl, create_run_dir, git_hash, load_checkpoint, save_checkpoint
from src.utils.seed import set_seed


def test_apply_overrides_nested_key_creation():
    cfg = {"a": {"b": 1}}
    out = apply_overrides(cfg, ["a.c.d=2", "top=3"])
    assert out["a"]["c"]["d"] == 2
    assert out["a"]["b"] == 1
    assert out["top"] == 3


def test_apply_overrides_yaml_value_parsing():
    out = apply_overrides(
        {},
        [
            "training.learning_rate=1.0e-3",
            "augmentation.hflip=true",
            "x=[1,2]",
            "y=null",
        ],
    )
    assert out["training"]["learning_rate"] == 0.001
    assert out["augmentation"]["hflip"] is True
    assert out["x"] == [1, 2]
    assert out["y"] is None


def test_apply_overrides_does_not_mutate_input():
    cfg = {"a": {"b": 1}}
    apply_overrides(cfg, ["a.b=2", "new.section.key=9"])
    assert cfg == {"a": {"b": 1}}


def test_apply_overrides_rejects_missing_equals():
    with pytest.raises(ValueError):
        apply_overrides({}, ["not-an-override"])


def test_load_save_config_round_trip(tmp_path):
    cfg = {"training": {"seed": 42, "lr": 1.0e-4}, "dataset": {"root": None}}
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)
    assert load_config(path) == cfg
    text = path.read_text(encoding="utf-8")
    assert "training:" in text and "seed: 42" in text  # readable block-style dump


def test_load_config_empty_file_is_empty_dict(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path) == {}


def test_load_config_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(Path(tmp_path) / "missing.yaml")


def test_set_seed_determinism():
    set_seed(42)
    a = torch.rand(4)
    set_seed(42)
    b = torch.rand(4)
    assert torch.equal(a, b)


def test_create_run_dir_creates_parents(tmp_path):
    run_dir = create_run_dir(tmp_path / "nested" / "base", "dfcan_CCPs")
    assert run_dir.is_dir()
    assert run_dir.parent == tmp_path / "nested" / "base"
    assert run_dir.name.endswith("_dfcan_CCPs")


def test_create_run_dir_collision_suffixes(tmp_path, monkeypatch):
    real_datetime = datetime.datetime

    class _FrozenDatetime:
        @staticmethod
        def now() -> datetime.datetime:
            return real_datetime(2024, 1, 1, 0, 0, 0)

    monkeypatch.setattr(run_mod, "datetime", _FrozenDatetime)
    d1 = create_run_dir(tmp_path, "m")
    d2 = create_run_dir(tmp_path, "m")
    d3 = create_run_dir(tmp_path, "m")
    assert d1.name == "20240101-000000_m"
    assert d2.name == "20240101-000000_m-1"
    assert d3.name == "20240101-000000_m-2"


def test_checkpoint_round_trip(tmp_path):
    model = torch.nn.Linear(4, 2)
    cfg = {"model": {"name": "tiny"}}
    metrics = {"val_psnr": 21.5}
    path = tmp_path / "ckpts" / "best.pt"
    save_checkpoint(path, model, cfg, epoch=3, metrics=metrics)
    loaded = load_checkpoint(path)
    assert loaded["epoch"] == 3
    assert loaded["config"] == cfg
    assert loaded["metrics"] == metrics
    assert loaded["git_hash"] is None or isinstance(loaded["git_hash"], str)
    restored = torch.nn.Linear(4, 2)
    restored.load_state_dict(loaded["model"])
    assert torch.equal(restored.weight, model.weight)


def test_append_jsonl_writes_valid_json_lines(tmp_path):
    path = tmp_path / "logs" / "train_log.jsonl"
    append_jsonl(path, {"epoch": 1, "loss": 0.5})
    append_jsonl(path, {"epoch": 2, "loss": 0.25})
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line) for line in lines] == [
        {"epoch": 1, "loss": 0.5},
        {"epoch": 2, "loss": 0.25},
    ]


def test_git_hash_returns_none_or_str():
    value = git_hash()
    assert value is None or isinstance(value, str)
