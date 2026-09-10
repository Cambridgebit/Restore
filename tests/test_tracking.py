"""Tests for the optional wandb tracker (must no-op gracefully when absent/disabled)."""

import numpy as np

import src.utils.tracking as tracking
from src.utils.tracking import WandbTracker


def test_disabled_tracker_no_ops() -> None:
    tracker = WandbTracker({})
    assert tracker.enabled is False
    assert tracker.run is None
    tracker.log_metrics({"a": 1.0}, step=0)
    tracker.log_images("k", [(np.zeros((4, 4, 1), dtype=np.float32), "caption")], step=0)
    tracker.update_summary({"a": 1.0})
    tracker.finish()
    assert tracker.enabled is False


def test_enabled_tracker_without_wandb_package_degrades(monkeypatch, tmp_path) -> None:
    """enabled=true + missing package -> warning path, tracker disabled, no crash."""
    monkeypatch.setattr(tracking, "wandb", None)
    tracker = tracking.WandbTracker({"wandb": {"enabled": True}}, run_dir=tmp_path)
    assert tracker.enabled is False
    assert tracker.run is None
    tracker.log_metrics({"a": 1.0}, step=0)
    tracker.finish()


def test_enabled_tracker_uses_offline_mode_when_available(tmp_path) -> None:
    """With wandb installed, offline mode must init/finish inside run_dir (no network)."""
    if not tracking.wandb_available():
        return  # local env without wandb: covered by the degradation test above
    cfg = {"wandb": {"enabled": True, "project": "unit-test", "mode": "offline"}}
    tracker = tracking.WandbTracker(cfg, run_dir=tmp_path, name="unit")
    try:
        assert tracker.enabled is True
        assert tracker.run is not None
        tracker.log_metrics({"psnr": 30.0}, step=1)
        tracker.log_images(
            "panel", [(np.zeros((8, 8, 1), dtype=np.float32), "caption")], step=1
        )
        tracker.update_summary({"best": 1.0})
    finally:
        tracker.finish()
    assert tracker.run is None


def test_enabled_tracker_init_failure_degrades(monkeypatch, tmp_path) -> None:
    """wandb.init raising (network/auth flakiness) must disable tracking, never kill training."""

    class ExplodingWandb:
        @staticmethod
        def init(*args, **kwargs):
            raise RuntimeError("network down")

    monkeypatch.setattr(tracking, "wandb", ExplodingWandb)
    tracker = tracking.WandbTracker({"wandb": {"enabled": True}}, run_dir=tmp_path)
    assert tracker.enabled is False
    assert tracker.run is None
    tracker.log_metrics({"a": 1.0}, step=0)  # no-op, no crash
    tracker.finish()


def test_enabled_tracker_log_failure_degrades(monkeypatch, tmp_path) -> None:
    """A failing run.log degrades tracking after one warning instead of crashing training."""

    class FakeRun:
        def log(self, *args, **kwargs):
            raise RuntimeError("flaky upload")

    class FakeWandb:
        @staticmethod
        def init(*args, **kwargs):
            return FakeRun()

    monkeypatch.setattr(tracking, "wandb", FakeWandb)
    tracker = tracking.WandbTracker({"wandb": {"enabled": True}}, run_dir=tmp_path)
    assert tracker.enabled is True
    tracker.log_metrics({"a": 1.0}, step=0)  # first failure -> degrade
    assert tracker.enabled is False
    tracker.log_metrics({"b": 2.0}, step=1)  # now a no-op
    tracker.finish()
