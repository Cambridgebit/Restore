"""Leave-one-structure-out splits with cell-level leakage guarantees.

Implements AGENT.md §3 (LOSO folds) and §4 (data-leakage rules): the split is
decided per cell/FOV *before* any patching, all levels of one cell stay in the
same split, and the held-out structure never enters train/val.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from src.data.biosr import BioSRSample

# AGENT.md §3: hold out one structure, train on the other three.
LOSO_FOLDS: dict[str, str] = {
    "fold_1": "CCPs",
    "fold_2": "ER",
    "fold_3": "Microtubules",
    "fold_4": "F-actin",
}


@dataclass
class Split:
    """LOSO split: held-out structure for test, cell-disjoint train/val for the rest."""

    held_out: str
    train: list[BioSRSample]
    val: list[BioSRSample]
    test: list[BioSRSample]

    def to_dict(self) -> dict:
        """JSON-safe manifest of the full split (all sample fields included)."""
        return {
            "held_out": self.held_out,
            "train": [sample.to_dict() for sample in self.train],
            "val": [sample.to_dict() for sample in self.val],
            "test": [sample.to_dict() for sample in self.test],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Split:
        """Rebuild a split from :meth:`to_dict` output."""
        return cls(
            held_out=data["held_out"],
            train=[BioSRSample.from_dict(item) for item in data["train"]],
            val=[BioSRSample.from_dict(item) for item in data["val"]],
            test=[BioSRSample.from_dict(item) for item in data["test"]],
        )

    def save_json(self, path: str | Path) -> None:
        """Write the split manifest as indented UTF-8 JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def make_loso_split(
    samples: Sequence[BioSRSample],
    held_out: str,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Split:
    """Build a deterministic LOSO split; all samples of ``held_out`` go to test.

    Train/val are split BY CELL among the remaining structures (never by patch
    or level), so every level and realization of one cell stays in one split.
    Per structure: ``n_val = 0 if len(cells) < 2 else max(1, round(val_fraction *
    len(cells)))`` cells are drawn with a fresh ``Generator(PCG64(seed))`` so
    adding a structure does not change the other structures' assignment.
    """
    by_structure: dict[str, list[BioSRSample]] = {}
    for sample in samples:
        by_structure.setdefault(sample.structure, []).append(sample)
    if held_out not in by_structure:
        available = ", ".join(sorted(by_structure)) or "<none>"
        raise ValueError(f"held-out structure {held_out!r} has no samples; available: {available}")

    test = sorted(by_structure[held_out], key=lambda sample: sample.id)
    val_cells: dict[str, set[str]] = {}
    for structure in sorted(key for key in by_structure if key != held_out):
        cells = sorted({sample.cell for sample in by_structure[structure]})
        n_val = 0 if len(cells) < 2 else max(1, round(val_fraction * len(cells)))
        rng = np.random.Generator(np.random.PCG64(seed))
        order = rng.permutation(len(cells))
        val_cells[structure] = {cells[int(index)] for index in order[:n_val]}

    train_list: list[BioSRSample] = []
    val_list: list[BioSRSample] = []
    for structure, cell_pool in val_cells.items():
        for sample in by_structure[structure]:
            if sample.cell in cell_pool:
                val_list.append(replace(sample, split="val"))
            else:
                train_list.append(replace(sample, split="train"))

    if not val_list:
        raise ValueError(
            "validation split is empty; the structures left after holding out "
            f"{held_out!r} have too few cells for val_fraction={val_fraction}"
        )

    return Split(
        held_out=held_out,
        train=sorted(train_list, key=lambda sample: sample.id),
        val=sorted(val_list, key=lambda sample: sample.id),
        test=[replace(sample, split="test") for sample in test],
    )


def check_no_leakage(split: Split) -> None:
    """Raise ValueError on any AGENT.md §4 violation.

    (a) the held-out structure appears in train or val (by structure or by
    structure+cell); (b) any (structure, cell) in both train and val;
    (c) duplicate sample ids within one split.
    """
    for name in ("train", "val", "test"):
        ids = [sample.id for sample in getattr(split, name)]
        duplicates = sorted({sample_id for sample_id in ids if ids.count(sample_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate sample ids in {name} split: {duplicates}")

    held_out_cells = {(sample.structure, sample.cell) for sample in split.test}
    for name in ("train", "val"):
        offenders = sorted(
            sample.id
            for sample in getattr(split, name)
            if sample.structure == split.held_out or (sample.structure, sample.cell) in held_out_cells
        )
        if offenders:
            raise ValueError(f"held-out structure {split.held_out!r} leaked into {name}: {offenders}")

    train_cells = {(sample.structure, sample.cell) for sample in split.train}
    val_cells = {(sample.structure, sample.cell) for sample in split.val}
    overlap = sorted(f"{structure}/{cell}" for structure, cell in train_cells & val_cells)
    if overlap:
        raise ValueError(f"cells appear in both train and val: {overlap}")


def filter_samples(
    samples: Sequence[BioSRSample],
    structures: Iterable[str] | None = None,
    levels: Iterable[int] | None = None,
) -> list[BioSRSample]:
    """Whitelist filter on structure names and/or levels; input order preserved."""
    structure_set = set(structures) if structures is not None else None
    level_set = set(levels) if levels is not None else None
    return [
        sample
        for sample in samples
        if (structure_set is None or sample.structure in structure_set)
        and (level_set is None or sample.level in level_set)
    ]
