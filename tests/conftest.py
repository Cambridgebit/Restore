"""Shared synthetic BioSR fixtures: a small Cell_XXX tree written as float32 TIFFs.

The tree mirrors docs/data_protocol.md layouts (flat + ER) and includes the
protocol's distractor cases (n2v2 files, checkpoint dirs, GT-less cells).
All src.* imports live inside fixture bodies so sibling test modules can be
collected before src.data is importable.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile

if TYPE_CHECKING:
    from src.data.biosr import BioSRSample

LR_SIZE = 32
HR_SIZE = 64
BLOCK = HR_SIZE // LR_SIZE  # 4x4 block mean downsampling HR -> LR


def _hr_pattern(rng: np.random.Generator, size: int = HR_SIZE) -> np.ndarray:
    """Random smooth blob pattern in [0, 1] used as HR ground truth."""
    yy, xx = np.mgrid[0:size, 0:size]
    image = np.full((size, size), 0.1, dtype=np.float64)
    for _ in range(6):
        cy, cx = rng.integers(8, size - 8, size=2)
        amplitude = rng.uniform(0.4, 0.9)
        sigma = rng.uniform(3.0, 7.0)
        image += amplitude * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2))
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _block_mean(hr: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Block-mean downsample an HR image to (H // block, W // block)."""
    size = hr.shape[0] // block
    return hr.reshape(size, block, size, block).mean(axis=(1, 3)).astype(np.float32)


@pytest.fixture
def biosr_root(tmp_path: Path) -> Path:
    """Synthetic BioSR tree: 14 valid samples (+2 dropped / +2 incomplete).

    Microtubules (flat): Cell_001/002/003 x levels 05+09 with SIM_gt
    (Cell_002 without wf_gt; Cell_003 plus n2v2 + checkpoint distractors),
    Cell_004 LR-only (dropped). CCPs (flat): Cell_001/002 x levels 05+09.
    F-actin (flat): Cell_001/002, level 05 only. ER (er layout): Cell_001
    level 05 with RawGTSIMData. F-actin_Nonlinear: Cell_001 level 05 (scale 3).
    Myosin-IIA_MRC: Cell_001 LR-only (no GT -> dropped).
    """
    root = tmp_path / "BioSR"
    rng = np.random.Generator(np.random.PCG64(0))

    def write_lr(cell_dir: Path, name: str) -> None:
        tifffile.imwrite(cell_dir / name, _block_mean(_hr_pattern(rng)))

    def write_gt_pair(cell_dir: Path, level: int) -> None:
        hr = _hr_pattern(rng)
        tifffile.imwrite(cell_dir / "SIM_gt.tif", hr)
        tifffile.imwrite(cell_dir / f"RawSIMData_level_{level:02d}.tif", _block_mean(_hr_pattern(rng)))

    def write_wf(cell_dir: Path) -> None:
        tifffile.imwrite(cell_dir / "RawSIMData_gt.tif", _block_mean(_hr_pattern(rng)))

    mt = root / "Microtubules"
    for cell in ("Cell_001", "Cell_002", "Cell_003"):
        cell_dir = mt / cell
        cell_dir.mkdir(parents=True)
        for level in (5, 9):
            write_gt_pair(cell_dir, level)
        if cell != "Cell_002":
            write_wf(cell_dir)
    tifffile.imwrite(
        mt / "Cell_003" / "RawSIMData_level_05_n2v2.tif",
        np.zeros((LR_SIZE, LR_SIZE), dtype=np.float32),
    )
    checkpoint_dir = mt / "Cell_003" / "checkpoint"
    checkpoint_dir.mkdir()
    tifffile.imwrite(checkpoint_dir / "junk.tif", np.zeros((LR_SIZE, LR_SIZE), dtype=np.float32))
    cell_004 = mt / "Cell_004"
    cell_004.mkdir(parents=True)
    write_lr(cell_004, "RawSIMData_level_05.tif")

    for cell, with_wf in (("Cell_001", True), ("Cell_002", False)):
        cell_dir = root / "CCPs" / cell
        cell_dir.mkdir(parents=True)
        for level in (5, 9):
            write_gt_pair(cell_dir, level)
        if with_wf:
            write_wf(cell_dir)

    for cell in ("Cell_001", "Cell_002"):
        cell_dir = root / "F-actin" / cell
        cell_dir.mkdir(parents=True)
        write_gt_pair(cell_dir, 5)

    er_cell = root / "ER" / "Cell_001"
    (er_cell / "RawSIMData").mkdir(parents=True)
    (er_cell / "RawGTSIMData").mkdir()
    (er_cell / "GTSIM").mkdir()
    hr = _hr_pattern(rng)
    lr = _block_mean(_hr_pattern(rng))
    tifffile.imwrite(er_cell / "RawSIMData" / "RawSIMData_level_05.tif", lr)
    tifffile.imwrite(er_cell / "GTSIM" / "GTSIM_level_05.tif", hr)
    tifffile.imwrite(er_cell / "RawGTSIMData" / "RawGTSIMData_level_05.tif", _block_mean(_hr_pattern(rng)))

    nonlinear = root / "F-actin_Nonlinear" / "Cell_001"
    nonlinear.mkdir(parents=True)
    # Geometrically consistent x3 pair: HR 96x96, LR = 3x3 block mean (32x32).
    nonlinear_hr = _hr_pattern(rng, size=96)
    tifffile.imwrite(nonlinear / "SIM_gt.tif", nonlinear_hr)
    tifffile.imwrite(nonlinear / "RawSIMData_level_05.tif", _block_mean(nonlinear_hr, block=3))

    myosin = root / "Myosin-IIA_MRC" / "Cell_001"
    myosin.mkdir(parents=True)
    write_lr(myosin, "RawSIMData_level_05.tif")

    return root


@pytest.fixture
def manifest(biosr_root: Path) -> list[BioSRSample]:
    """All valid SR samples discovered from the synthetic tree (14 samples)."""
    from src.data.biosr import discover_samples

    return discover_samples(biosr_root)
