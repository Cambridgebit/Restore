"""YAML config loading, dotted-path CLI overrides, and saving."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file; an empty file yields {}. Raises FileNotFoundError."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def apply_overrides(cfg: dict, overrides: Sequence[str]) -> dict:
    """Apply "dotted.key=value" overrides to a deep copy of cfg (input not mutated).

    Values are parsed with yaml.safe_load, so 1e-3, true, [1,2], null all work.
    Dotted paths create intermediate dicts as needed.
    """
    out = copy.deepcopy(cfg)
    for override in overrides:
        key, sep, raw = override.partition("=")
        if not sep:
            raise ValueError(f"override must be 'dotted.key=value', got {override!r}")
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def save_config(cfg: dict, path: str | Path) -> None:
    """Dump cfg as readable YAML, preserving key order."""
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
