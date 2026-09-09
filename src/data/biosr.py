"""BioSR sample discovery, dataset, and data-root resolution.

Implements docs/data_protocol.md: Cell_XXX directory discovery for the flat and
ER layouts, SR sample pairing (input -> hr_gt), and the per-image min-max
normalization default. Never modifies the original TIFF files.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

# Default root from docs/data_protocol.md; override with the BIOSR_ROOT env var.
DEFAULT_BIOSR_ROOT = Path("/home/user/Documents/Code_yjq/SuperRestore/Robust_Restore/datasets/BioSR")
# Protocol §1: Myosin-IIA_MRC lacks paired high-resolution GT and is never used.
EXCLUDED_STRUCTURES = frozenset({"Myosin-IIA_MRC"})

_CELL_RE = re.compile(r"Cell_\d+")
_LEVEL_RE = re.compile(r"RawSIMData_level_(\d+)\.tif")
_GTSIM_RE = re.compile(r"GTSIM_level_(\d+)\.tif")
_RAW_GTSIM_RE = re.compile(r"RawGTSIMData_level_(\d+)\.tif")

_SAMPLE_FIELDS = ("id", "structure", "cell", "level", "layout", "input", "hr_gt", "wf_gt", "scale_hr", "split")
_AUGMENT_KEYS = ("hflip", "vflip", "rot90", "intensity_range", "noise_max_sigma")


@dataclass(frozen=True)
class BioSRSample:
    """One structure/cell/level SR sample (docs/data_protocol.md §2)."""

    id: str
    structure: str
    cell: str
    level: int
    layout: str  # "flat" | "er"
    input: str  # LR image, posix-style relative to the BioSR root
    hr_gt: str | None  # paired high-resolution SIM target
    wf_gt: str | None  # optional same-resolution reference
    scale_hr: float  # 2.0, or 3.0 for *_Nonlinear structures
    split: str | None  # "train" | "val" | "test"; assigned by src.data.splits

    def to_dict(self) -> dict:
        """JSON-safe dict with exactly the protocol field names (manifest-ready)."""
        return {
            "id": self.id,
            "structure": self.structure,
            "cell": self.cell,
            "level": self.level,
            "layout": self.layout,
            "input": self.input,
            "hr_gt": self.hr_gt,
            "wf_gt": self.wf_gt,
            "scale_hr": self.scale_hr,
            "split": self.split,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BioSRSample:
        """Rebuild a sample from :meth:`to_dict` output."""
        return cls(
            id=data["id"],
            structure=data["structure"],
            cell=data["cell"],
            level=int(data["level"]),
            layout=data["layout"],
            input=data["input"],
            hr_gt=data["hr_gt"],
            wf_gt=data["wf_gt"],
            scale_hr=float(data["scale_hr"]),
            split=data.get("split"),
        )


def resolve_root(root: str | Path | None) -> Path:
    """Resolve the BioSR root: explicit arg > $BIOSR_ROOT > protocol default.

    Raises FileNotFoundError (mentioning BIOSR_ROOT) when the resolved path is
    not an existing directory.
    """
    if root is not None:
        resolved = Path(root).expanduser()
    else:
        env = os.environ.get("BIOSR_ROOT")
        resolved = Path(env).expanduser() if env else DEFAULT_BIOSR_ROOT
    resolved = resolved.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"BioSR data root not found: {resolved} "
            f"(set the BIOSR_ROOT environment variable or pass dataset.root explicitly)"
        )
    return resolved


def discover_samples(
    root: str | Path,
    structures: Iterable[str] | None = None,
    keep_incomplete: bool = False,
) -> list[BioSRSample]:
    """Walk ``<root>/<structure>/Cell_XXX/`` and pair SR samples per the protocol.

    Layout detection is per cell (``RawSIMData`` subdirectory present -> "er").
    Flat pairing: LR = ``RawSIMData_level_NN.tif``, hr_gt = first
    ``SIM_gt*.tif`` (shared by the cell), wf_gt = optional ``RawSIMData_gt.tif``.
    ER pairing: per-level ``GTSIM/GTSIM_level_NN.tif`` (required) and optional
    ``RawGTSIMData/RawGTSIMData_level_NN.tif``.

    Ignores hidden files, ``*_n2v2.tif`` and other derived temp files (the exact
    filename regexes never match them), checkpoint directories (only ``Cell_NN``
    directories are descended into), and ``EXCLUDED_STRUCTURES`` unless
    ``keep_incomplete`` is set. Samples without ``hr_gt`` are dropped (kept with
    ``hr_gt=None`` when ``keep_incomplete``).

    Returns samples sorted by (structure, cell, level) with ``split=None``.
    """
    root_path = resolve_root(root)
    whitelist = set(structures) if structures is not None else None

    samples: list[BioSRSample] = []
    for structure_dir in _structure_dirs(root_path, whitelist, keep_incomplete):
        for cell_dir in sorted(
            path for path in structure_dir.iterdir() if path.is_dir() and _CELL_RE.fullmatch(path.name)
        ):
            samples.extend(_discover_cell(structure_dir.name, cell_dir, keep_incomplete))
    samples.sort(key=lambda sample: (sample.structure, sample.cell, sample.level))
    return samples


def _structure_dirs(root: Path, whitelist: set[str] | None, keep_incomplete: bool) -> list[Path]:
    """Candidate structure directories under root (hidden + excluded filtered)."""
    dirs = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        if not keep_incomplete and path.name in EXCLUDED_STRUCTURES:
            continue
        if whitelist is not None and path.name not in whitelist:
            continue
        dirs.append(path)
    return dirs


def _discover_cell(structure: str, cell_dir: Path, keep_incomplete: bool) -> list[BioSRSample]:
    """Discover samples for one cell, choosing the layout from its subdirectories."""
    if (cell_dir / "RawSIMData").is_dir():
        return _discover_cell_er(structure, cell_dir, keep_incomplete)
    return _discover_cell_flat(structure, cell_dir, keep_incomplete)


def _discover_cell_flat(structure: str, cell_dir: Path, keep_incomplete: bool) -> list[BioSRSample]:
    root = cell_dir.parent.parent
    sim_gts = sorted(path for path in cell_dir.glob("SIM_gt*.tif") if path.is_file())
    hr_gt = _relpath(sim_gts[0], root) if sim_gts else None
    wf_path = cell_dir / "RawSIMData_gt.tif"
    wf_gt = _relpath(wf_path, root) if wf_path.is_file() else None

    samples: list[BioSRSample] = []
    for path in sorted(cell_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        match = _LEVEL_RE.fullmatch(path.name)
        if match is None:
            continue
        _append_sample(
            samples,
            structure=structure,
            cell=cell_dir.name,
            level=int(match.group(1)),
            layout="flat",
            input_rel=_relpath(path, root),
            hr_gt=hr_gt,
            wf_gt=wf_gt,
            keep_incomplete=keep_incomplete,
        )
    return samples


def _discover_cell_er(structure: str, cell_dir: Path, keep_incomplete: bool) -> list[BioSRSample]:
    root = cell_dir.parent.parent
    gtsim_levels = _levels_in_dir(cell_dir / "GTSIM", _GTSIM_RE)
    raw_gtsim_levels = _levels_in_dir(cell_dir / "RawGTSIMData", _RAW_GTSIM_RE)

    samples: list[BioSRSample] = []
    for path in sorted((cell_dir / "RawSIMData").iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        match = _LEVEL_RE.fullmatch(path.name)
        if match is None:
            continue
        level = int(match.group(1))
        _append_sample(
            samples,
            structure=structure,
            cell=cell_dir.name,
            level=level,
            layout="er",
            input_rel=_relpath(path, root),
            hr_gt=gtsim_levels.get(level),
            wf_gt=raw_gtsim_levels.get(level),
            keep_incomplete=keep_incomplete,
        )
    return samples


def _levels_in_dir(directory: Path, pattern: re.Pattern[str]) -> dict[int, str]:
    """Map level -> root-relative path for files matching ``pattern`` in a directory.

    ``directory`` is ``<root>/<structure>/<cell>/<subdir>`` — three levels down
    from the root, one deeper than the flat-layout cell files.
    """
    levels: dict[int, str] = {}
    if not directory.is_dir():
        return levels
    root = directory.parent.parent.parent
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        match = pattern.fullmatch(path.name)
        if match is not None:
            levels[int(match.group(1))] = _relpath(path, root)
    return levels


def _append_sample(
    out: list[BioSRSample],
    structure: str,
    cell: str,
    level: int,
    layout: str,
    input_rel: str,
    hr_gt: str | None,
    wf_gt: str | None,
    keep_incomplete: bool,
) -> None:
    """Append one sample unless it lacks hr_gt and incomplete samples are dropped."""
    if hr_gt is None and not keep_incomplete:
        return
    out.append(
        BioSRSample(
            id=f"{structure}/{cell}/level_{level:02d}",
            structure=structure,
            cell=cell,
            level=level,
            layout=layout,
            input=input_rel,
            hr_gt=hr_gt,
            wf_gt=wf_gt,
            scale_hr=3.0 if structure.endswith("_Nonlinear") else 2.0,
            split=None,
        )
    )


def _relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


class BioSRDataset(Dataset):
    """Paired (LR, HR) BioSR images as float32 (1, H, W) tensors in [0, 1].

    ``patch_size`` is the LR patch size (the HR crop is ``patch_size * scale``).
    ``crop_mode``: "random" (training), "center" (validation), None (full
    images). Images are min-max normalized per full image *before* cropping
    (docs/data_protocol.md §3 default). Augmentation (crop first, then
    geometric, intensity, noise) uses the global torch RNG.
    """

    def __init__(
        self,
        samples: Sequence[BioSRSample],
        root: str | Path | None = None,
        patch_size: int | None = None,
        crop_mode: str | None = None,
        augment: dict | None = None,
        normalization: str = "per_image_minmax",
    ) -> None:
        self.root = resolve_root(root)
        self.samples = list(samples)
        self.patch_size = patch_size
        self.crop_mode = crop_mode
        if normalization != "per_image_minmax":
            raise NotImplementedError(
                f"unsupported normalization {normalization!r}; only 'per_image_minmax' is "
                "implemented (docs/data_protocol.md §3); record alternatives in the config"
            )
        if crop_mode not in (None, "random", "center"):
            raise ValueError(f"invalid crop_mode {crop_mode!r}; expected None, 'random', or 'center'")
        if patch_size is not None:
            if patch_size < 1:
                raise ValueError(f"patch_size must be >= 1, got {patch_size}")
            if crop_mode is None:
                raise ValueError("patch_size requires crop_mode 'random' or 'center'")
        if augment is not None:
            unknown = sorted(set(augment) - set(_AUGMENT_KEYS))
            if unknown:
                raise ValueError(f"unknown augmentation keys {unknown}; valid: {list(_AUGMENT_KEYS)}")
        self.augment = dict(augment) if augment is not None else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        if sample.hr_gt is None:
            raise ValueError(f"sample {sample.id} has no hr_gt; SR samples require a paired target")
        lr_img = _load_image(self.root / sample.input)
        gt_img = _load_image(self.root / sample.hr_gt)
        lr_img = _minmax(lr_img)
        gt_img = _minmax(gt_img)
        if self.patch_size is not None:
            lr_img, gt_img = _crop_pair(lr_img, gt_img, self.patch_size, int(sample.scale_hr), self.crop_mode)
        lr_tensor = torch.from_numpy(lr_img).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_img).unsqueeze(0)
        if self.augment is not None:
            lr_tensor, gt_tensor = _augment_pair(lr_tensor, gt_tensor, self.augment)
        return lr_tensor, gt_tensor


def _load_image(path: Path) -> np.ndarray:
    """Read a TIFF as float32 (H, W), squeezing a trailing single channel."""
    image = tifffile.imread(path)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    return np.asarray(image, dtype=np.float32)


def _minmax(image: np.ndarray) -> np.ndarray:
    """Min-max normalize one image to [0, 1] (protocol default; flat -> zeros)."""
    low, high = float(image.min()), float(image.max())
    if high - low < 1e-12:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _crop_pair(
    lr: np.ndarray,
    gt: np.ndarray,
    patch: int,
    scale: int,
    crop_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Aligned LR/HR crop of (patch, patch) and (patch*scale, patch*scale)."""
    height, width = lr.shape[-2:]
    if height < patch or width < patch:
        raise ValueError(f"LR image {height}x{width} is smaller than patch_size={patch}")
    hr_height, hr_width = gt.shape[-2:]
    hr_patch = patch * scale
    if hr_height < hr_patch or hr_width < hr_patch:
        raise ValueError(
            f"HR image {hr_height}x{hr_width} is smaller than patch*scale={hr_patch}; "
            f"check the pairing for this sample"
        )
    if crop_mode == "center":
        top = (height - patch) // 2
        left = (width - patch) // 2
    else:  # crop_mode == "random" (validated in __init__)
        top = int(torch.randint(0, height - patch + 1, (1,)).item())
        left = int(torch.randint(0, width - patch + 1, (1,)).item())
    lr_crop = lr[top : top + patch, left : left + patch]
    gt_crop = gt[top * scale : top * scale + hr_patch, left * scale : left * scale + hr_patch]
    return lr_crop, gt_crop


def _augment_pair(lr: torch.Tensor, gt: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Geometric (identical for LR/GT) -> intensity (shared factor) -> noise (LR only)."""
    if cfg.get("rot90", False):
        k = int(torch.randint(0, 4, (1,)).item())
        lr = torch.rot90(lr, k, dims=(-2, -1))
        gt = torch.rot90(gt, k, dims=(-2, -1))
    if cfg.get("hflip", False) and torch.rand(()) < 0.5:
        lr = torch.flip(lr, dims=(-1,))
        gt = torch.flip(gt, dims=(-1,))
    if cfg.get("vflip", False) and torch.rand(()) < 0.5:
        lr = torch.flip(lr, dims=(-2,))
        gt = torch.flip(gt, dims=(-2,))
    intensity_range = cfg.get("intensity_range")
    if intensity_range is not None:
        low, high = float(intensity_range[0]), float(intensity_range[1])
        factor = low + (high - low) * float(torch.rand(()).item())
        lr = (lr * factor).clamp(0.0, 1.0)
        gt = (gt * factor).clamp(0.0, 1.0)
    sigma_max = float(cfg.get("noise_max_sigma") or 0.0)
    if sigma_max > 0.0:
        sigma = sigma_max * float(torch.rand(()).item())
        lr = (lr + sigma * torch.randn_like(lr)).clamp(0.0, 1.0)
    return lr, gt
