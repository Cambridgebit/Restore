"""Tests for BioSR sample discovery, root resolution, and the dataset."""

import pytest
import torch

from src.data.biosr import (
    DEFAULT_BIOSR_ROOT,
    BioSRDataset,
    BioSRSample,
    discover_samples,
    resolve_root,
)


def _by_id(manifest, sample_id):
    return next(sample for sample in manifest if sample.id == sample_id)


# --- discovery ---------------------------------------------------------------


def test_discover_total_count(manifest):
    assert len(manifest) == 14
    counts = {}
    for sample in manifest:
        counts[sample.structure] = counts.get(sample.structure, 0) + 1
    assert counts == {"Microtubules": 6, "CCPs": 4, "F-actin": 2, "ER": 1, "F-actin_Nonlinear": 1}


def test_ids_sorted_and_formatted(manifest):
    ids = [sample.id for sample in manifest]
    assert ids == sorted(ids)
    for sample in manifest:
        assert sample.id == f"{sample.structure}/{sample.cell}/level_{sample.level:02d}"
        assert sample.split is None


def test_flat_sample_fields(manifest):
    sample = _by_id(manifest, "Microtubules/Cell_001/level_05")
    assert sample.input == "Microtubules/Cell_001/RawSIMData_level_05.tif"
    assert sample.hr_gt == "Microtubules/Cell_001/SIM_gt.tif"
    assert sample.wf_gt == "Microtubules/Cell_001/RawSIMData_gt.tif"
    assert sample.scale_hr == 2.0
    assert sample.layout == "flat"


def test_flat_sample_without_wf(manifest):
    sample = _by_id(manifest, "Microtubules/Cell_002/level_09")
    assert sample.wf_gt is None
    assert sample.hr_gt == "Microtubules/Cell_002/SIM_gt.tif"


def test_derived_files_ignored(manifest):
    cell3 = [s for s in manifest if s.id.startswith("Microtubules/Cell_003/")]
    assert len(cell3) == 2
    assert {s.level for s in cell3} == {5, 9}
    assert all("n2v2" not in s.input for s in manifest)
    assert all("checkpoint" not in s.input for s in manifest)


def test_gtless_cell_dropped(manifest):
    assert not any(sample.cell == "Cell_004" for sample in manifest)
    assert not any(sample.structure == "Myosin-IIA_MRC" for sample in manifest)


def test_er_sample(manifest):
    sample = _by_id(manifest, "ER/Cell_001/level_05")
    assert sample.layout == "er"
    assert sample.input == "ER/Cell_001/RawSIMData/RawSIMData_level_05.tif"
    assert sample.hr_gt == "ER/Cell_001/GTSIM/GTSIM_level_05.tif"
    assert sample.wf_gt == "ER/Cell_001/RawGTSIMData/RawGTSIMData_level_05.tif"
    assert sample.level == 5


def test_nonlinear_scale(manifest):
    sample = _by_id(manifest, "F-actin_Nonlinear/Cell_001/level_05")
    assert sample.scale_hr == 3.0


def test_structures_whitelist(biosr_root):
    samples = discover_samples(biosr_root, structures=["CCPs"])
    assert len(samples) == 4
    assert {sample.structure for sample in samples} == {"CCPs"}


def test_keep_incomplete_includes_gtless(biosr_root):
    samples = discover_samples(biosr_root, keep_incomplete=True)
    assert len(samples) == 16
    cell4 = next(s for s in samples if s.id == "Microtubules/Cell_004/level_05")
    assert cell4.hr_gt is None
    assert cell4.wf_gt is None
    myosin = next(s for s in samples if s.structure == "Myosin-IIA_MRC")
    assert myosin.hr_gt is None


def test_sample_roundtrip(manifest):
    sample = manifest[0]
    data = sample.to_dict()
    assert set(data) == {
        "id", "structure", "cell", "level", "layout", "input", "hr_gt", "wf_gt", "scale_hr", "split",
    }
    assert BioSRSample.from_dict(data) == sample


def test_sample_manifest_json_roundtrip(manifest, tmp_path):
    import json

    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(s.to_dict()) for s in manifest), encoding="utf-8")
    restored = [BioSRSample.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]
    assert restored == manifest


# --- root resolution ---------------------------------------------------------


def test_resolve_root_precedence(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    env_root = tmp_path / "envroot"
    env_root.mkdir()
    monkeypatch.setenv("BIOSR_ROOT", str(env_root))
    assert resolve_root(explicit) == explicit  # explicit arg wins over env
    assert resolve_root(None) == env_root


def test_resolve_root_default_used_without_env(monkeypatch):
    monkeypatch.delenv("BIOSR_ROOT", raising=False)
    try:
        resolve_root(None)
        found_default = True
    except FileNotFoundError:
        found_default = False  # default root not present on this machine
    if not found_default:
        assert DEFAULT_BIOSR_ROOT.name == "BioSR"


def test_resolve_root_missing_mentions_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("BIOSR_ROOT", raising=False)
    with pytest.raises(FileNotFoundError, match="BIOSR_ROOT"):
        resolve_root(None)
    with pytest.raises(FileNotFoundError, match="BIOSR_ROOT"):
        resolve_root(tmp_path / "nope")


# --- dataset -----------------------------------------------------------------


def test_dataset_full_image(manifest, biosr_root):
    dataset = BioSRDataset(manifest, root=biosr_root)
    assert len(dataset) == 14
    lr, gt = dataset[0]
    assert lr.shape == (1, 32, 32)
    assert gt.shape == (1, 64, 64)
    assert lr.dtype == torch.float32
    assert gt.dtype == torch.float32
    assert lr.min() >= 0.0 and lr.max() <= 1.0
    assert gt.min() >= 0.0 and gt.max() <= 1.0


def test_dataset_er_layout_shapes(manifest, biosr_root):
    dataset = BioSRDataset(manifest, root=biosr_root)
    index = next(i for i, s in enumerate(manifest) if s.layout == "er")
    lr, gt = dataset[index]
    assert lr.shape == (1, 32, 32)
    assert gt.shape == (1, 64, 64)


def test_dataset_center_crop_deterministic(manifest, biosr_root):
    dataset = BioSRDataset(manifest, root=biosr_root, patch_size=16, crop_mode="center")
    lr1, gt1 = dataset[0]
    lr2, gt2 = dataset[0]
    assert lr1.shape == (1, 16, 16)
    assert gt1.shape == (1, 32, 32)
    assert torch.equal(lr1, lr2)
    assert torch.equal(gt1, gt2)


def test_dataset_random_crop_shapes(manifest, biosr_root):
    torch.manual_seed(0)
    dataset = BioSRDataset(manifest, root=biosr_root, patch_size=16, crop_mode="random")
    lr, gt = dataset[1]
    assert lr.shape == (1, 16, 16)
    assert gt.shape == (1, 32, 32)


def test_dataset_nonlinear_scale_crop(manifest, biosr_root):
    torch.manual_seed(0)
    dataset = BioSRDataset(manifest, root=biosr_root, patch_size=16, crop_mode="random")
    index = next(i for i, s in enumerate(manifest) if s.scale_hr == 3.0)
    lr, gt = dataset[index]
    assert lr.shape == (1, 16, 16)
    assert gt.shape == (1, 48, 48)


def test_dataset_patch_too_large(manifest, biosr_root):
    dataset = BioSRDataset(manifest, root=biosr_root, patch_size=64, crop_mode="random")
    with pytest.raises(ValueError, match="patch_size"):
        dataset[0]


def test_dataset_patch_requires_crop_mode(manifest, biosr_root):
    with pytest.raises(ValueError, match="crop_mode"):
        BioSRDataset(manifest, root=biosr_root, patch_size=16)


def test_dataset_augment_keeps_range(manifest, biosr_root):
    torch.manual_seed(0)
    dataset = BioSRDataset(
        manifest,
        root=biosr_root,
        patch_size=16,
        crop_mode="random",
        augment={
            "hflip": True,
            "vflip": True,
            "rot90": True,
            "intensity_range": [0.9, 1.1],
            "noise_max_sigma": 0.05,
        },
    )
    lr, gt = dataset[0]
    assert torch.isfinite(lr).all() and torch.isfinite(gt).all()
    assert lr.min() >= 0.0 and lr.max() <= 1.0
    assert gt.min() >= 0.0 and gt.max() <= 1.0


def test_dataset_augment_unknown_key(manifest, biosr_root):
    with pytest.raises(ValueError, match="augmentation"):
        BioSRDataset(manifest, root=biosr_root, crop_mode="center", patch_size=16, augment={"zoom": True})


def test_dataset_unknown_normalization(manifest, biosr_root):
    with pytest.raises(NotImplementedError, match="normalization"):
        BioSRDataset(manifest, root=biosr_root, normalization="zscore")


def test_dataset_invalid_crop_mode(manifest, biosr_root):
    with pytest.raises(ValueError, match="crop_mode"):
        BioSRDataset(manifest, root=biosr_root, crop_mode="diagonal")


def test_dataset_rejects_incomplete_sample(biosr_root):
    samples = discover_samples(biosr_root, keep_incomplete=True)
    dataset = BioSRDataset(samples, root=biosr_root)
    index = next(i for i, s in enumerate(samples) if s.hr_gt is None)
    with pytest.raises(ValueError, match="hr_gt"):
        dataset[index]
