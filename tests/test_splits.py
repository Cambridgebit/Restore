"""Tests for LOSO splits, leakage checks, and sample filtering."""

import json

import pytest

from src.data.biosr import BioSRSample
from src.data.splits import (
    LOSO_FOLDS,
    Split,
    check_no_leakage,
    filter_samples,
    make_loso_split,
)


def _ids(samples):
    return [sample.id for sample in samples]


def _cells(samples):
    return {(sample.structure, sample.cell) for sample in samples}


def _make_sample(structure: str, cell: str, level: int = 5) -> BioSRSample:
    return BioSRSample(
        id=f"{structure}/{cell}/level_{level:02d}",
        structure=structure,
        cell=cell,
        level=level,
        layout="flat",
        input=f"{structure}/{cell}/RawSIMData_level_{level:02d}.tif",
        hr_gt=f"{structure}/{cell}/SIM_gt.tif",
        wf_gt=None,
        scale_hr=2.0,
        split=None,
    )


# --- folds and basic split properties ----------------------------------------


def test_loso_folds_match_protocol():
    assert LOSO_FOLDS == {
        "fold_1": "CCPs",
        "fold_2": "ER",
        "fold_3": "Microtubules",
        "fold_4": "F-actin",
    }


@pytest.mark.parametrize("held_out", ["CCPs", "ER", "Microtubules", "F-actin"])
def test_loso_split_all_folds_pass_leakage(manifest, held_out):
    split = make_loso_split(manifest, held_out=held_out, val_fraction=0.34, seed=42)
    check_no_leakage(split)  # must not raise for any protocol fold
    assert {s.structure for s in split.test} == {held_out}
    train_structures = {s.structure for s in split.train}
    val_structures = {s.structure for s in split.val}
    assert held_out not in train_structures
    assert held_out not in val_structures
    assert train_structures == set(s.structure for s in manifest) - {held_out}
    assert val_structures <= train_structures


def test_loso_split_test_gets_all_levels(manifest):
    split = make_loso_split(manifest, held_out="CCPs", val_fraction=0.34, seed=42)
    expected = sorted(s.id for s in manifest if s.structure == "CCPs")
    assert _ids(split.test) == expected  # levels 5 and 9, untouched


def test_loso_split_levels_stay_with_cell(manifest):
    split = make_loso_split(manifest, held_out="F-actin", val_fraction=0.34, seed=42)
    train_cells = _cells(split.train)
    val_cells = _cells(split.val)
    assert train_cells.isdisjoint(val_cells)
    for structure, cell in sorted(train_cells | val_cells):
        levels_train = sorted(
            s.level for s in split.train if (s.structure, s.cell) == (structure, cell)
        )
        levels_val = sorted(s.level for s in split.val if (s.structure, s.cell) == (structure, cell))
        combined = levels_train + levels_val
        discovered = sorted(
            s.level
            for s in manifest
            if (s.structure, s.cell) == (structure, cell) and s.split is None
        )
        assert combined == discovered  # all levels of one cell in one split


def test_loso_split_assigns_split_field(manifest):
    split = make_loso_split(manifest, held_out="ER", val_fraction=0.34, seed=42)
    assert {s.split for s in split.train} == {"train"}
    assert {s.split for s in split.val} == {"val"}
    assert {s.split for s in split.test} == {"test"}
    assert {s.split for s in manifest if s.split is not None} == set()  # input untouched


def test_loso_split_deterministic(manifest):
    split_a = make_loso_split(manifest, held_out="CCPs", val_fraction=0.34, seed=42)
    split_b = make_loso_split(manifest, held_out="CCPs", val_fraction=0.34, seed=42)
    assert _ids(split_a.train) == _ids(split_b.train)
    assert _ids(split_a.val) == _ids(split_b.val)
    assert _ids(split_a.test) == _ids(split_b.test)


def test_loso_split_json_roundtrip(manifest, tmp_path):
    split = make_loso_split(manifest, held_out="ER", val_fraction=0.34, seed=42)
    path = tmp_path / "split.json"
    split.save_json(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["held_out"] == "ER"
    restored = Split.from_dict(data)
    assert restored.held_out == split.held_out
    assert _ids(restored.train) == _ids(split.train)
    assert _ids(restored.val) == _ids(split.val)
    assert _ids(restored.test) == _ids(split.test)


def test_loso_split_unknown_held_out(manifest):
    with pytest.raises(ValueError, match="held-out"):
        make_loso_split(manifest, held_out="Nucleus")


def test_loso_split_empty_val_raises(manifest):
    single_cell = [s for s in manifest if s.cell == "Cell_001"]
    with pytest.raises(ValueError, match="validation"):
        make_loso_split(single_cell, held_out="F-actin", val_fraction=0.1, seed=42)


# --- leakage checks ----------------------------------------------------------


def test_leakage_held_out_in_train_raises():
    train = [_make_sample("A", "Cell_001"), _make_sample("C", "Cell_001")]
    split = Split(held_out="A", train=train, val=[], test=[_make_sample("A", "Cell_002")])
    with pytest.raises(ValueError, match="leaked"):
        check_no_leakage(split)


def test_leakage_cell_in_both_train_and_val_raises():
    train = [_make_sample("A", "Cell_001"), _make_sample("A", "Cell_002")]
    val = [_make_sample("A", "Cell_001")]
    split = Split(held_out="B", train=train, val=val, test=[_make_sample("B", "Cell_001")])
    with pytest.raises(ValueError, match="both train and val"):
        check_no_leakage(split)


def test_leakage_duplicate_ids_raise():
    sample = _make_sample("A", "Cell_001")
    split = Split(
        held_out="B",
        train=[sample, sample],
        val=[_make_sample("A", "Cell_002")],
        test=[_make_sample("B", "Cell_001")],
    )
    with pytest.raises(ValueError, match="duplicate"):
        check_no_leakage(split)


# --- filtering ---------------------------------------------------------------


def test_filter_samples_by_levels(manifest):
    filtered = filter_samples(manifest, levels=[5])
    assert {s.level for s in filtered} == {5}
    assert len(filtered) == 9  # 7 flat level-05 (3 MT + 2 CCPs + 2 F-actin) + ER + F-actin_Nonlinear


def test_filter_samples_by_structures(manifest):
    filtered = filter_samples(manifest, structures=["F-actin"])
    assert len(filtered) == 2
    assert {s.structure for s in filtered} == {"F-actin"}


def test_filter_samples_combined(manifest):
    filtered = filter_samples(manifest, structures=["Microtubules"], levels=[9])
    assert len(filtered) == 3
    assert all(s.structure == "Microtubules" and s.level == 9 for s in filtered)


def test_filter_samples_no_match(manifest):
    assert filter_samples(manifest, structures=["F-actin"], levels=[9]) == []
