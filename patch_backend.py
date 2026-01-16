"""Patch-index backend helpers for Retriever Wizard.

Goal
----
Provide a minimal, robust layer around a precomputed patch/tile index.
This module is intentionally UI-free and model-free.

It supports:
- Standardized paths under output_dir/patch_index/
- Loading patch metadata + vectors + FAISS index
- Snapping a user crop to the nearest patch (IoU-based) WITHOUT embedding live

Recommended on-disk layout
--------------------------
output_dir/
  patch_index/
    patch_meta.parquet  (preferred) or patch_meta.csv
    patch_vectors.npy   (float32, shape [N, dim]) or patch_vectors.npz
    patch.faiss
    patch_index_manifest.json (optional but recommended)

Patch-meta required columns
---------------------------
- patch_id: int
- filename_key: str
- x1, y1, x2, y2: int  (pixel coordinates in original image)
- scale or level: int  (grid size or pyramid level)
- img_w, img_h: int    (original image size)
- stable_id: str       (optional, from index.csv)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


Box = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass(frozen=True)
class PatchIndexPaths:
    base_dir: Path
    meta_parquet: Path
    meta_csv: Path
    vectors_npy: Path
    vectors_npz: Path
    faiss_index: Path
    manifest_json: Path


def get_patch_index_paths(output_dir: str | Path) -> PatchIndexPaths:
    base = Path(output_dir) / "patch_index"
    return PatchIndexPaths(
        base_dir=base,
        meta_parquet=base / "patch_meta.parquet",
        meta_csv=base / "patch_meta.csv",
        vectors_npy=base / "patch_vectors.npy",
        vectors_npz=base / "patch_vectors.npz",
        faiss_index=base / "patch.faiss",
        manifest_json=base / "patch_index_manifest.json",
    )


def load_patch_meta(paths: PatchIndexPaths) -> pd.DataFrame:
    """Load patch metadata from parquet (preferred) or csv."""

    if paths.meta_parquet.exists():
        df = pd.read_parquet(paths.meta_parquet)
    elif paths.meta_csv.exists():
        df = pd.read_csv(paths.meta_csv, low_memory=False)
    else:
        raise FileNotFoundError(f"patch_meta not found in {paths.base_dir}")

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"patch_id", "filename_key", "x1", "y1", "x2", "y2"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"patch_meta missing columns: {missing}")

    # Normalize types
    df["patch_id"] = df["patch_id"].astype(int)
    df["filename_key"] = df["filename_key"].astype(str).str.strip().str.lower()
    for c in ("x1", "y1", "x2", "y2"):
        df[c] = df[c].astype(int)

    return df


def load_patch_vectors(
    paths: PatchIndexPaths,
    mmap_mode: Literal["r", "r+", "w+", "c"] | None = "r",
) -> np.ndarray:
    """Load patch vectors.

    - .npy is preferred and supports mmap.
    - .npz is supported as a fallback (loads fully into RAM).

    Returns float32 ndarray of shape (N, dim).
    """

    if paths.vectors_npy.exists():
        X = np.load(paths.vectors_npy, mmap_mode=mmap_mode)
        X = np.asarray(X, dtype=np.float32)
        return X

    if paths.vectors_npz.exists():
        z = np.load(paths.vectors_npz)
        # convention: store under key 'X'
        if "X" not in z:
            raise ValueError("patch_vectors.npz must contain array under key 'X'")
        X = np.asarray(z["X"], dtype=np.float32)
        return X

    raise FileNotFoundError(f"patch_vectors not found in {paths.base_dir}")


def load_patch_faiss_index(paths: PatchIndexPaths) -> Any:
    """Load patch FAISS index from patch.faiss."""

    if not paths.faiss_index.exists():
        raise FileNotFoundError(f"patch.faiss not found in {paths.base_dir}")

    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("FAISS is required to load patch.faiss (pip install faiss-cpu)") from e

    return faiss.read_index(str(paths.faiss_index))


def _area(b: Box) -> int:
    x1, y1, x2, y2 = b
    return max(0, int(x2) - int(x1)) * max(0, int(y2) - int(y1))


def _intersection(a: Box, b: Box) -> Box:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(int(ax1), int(bx1))
    y1 = max(int(ay1), int(by1))
    x2 = min(int(ax2), int(bx2))
    y2 = min(int(ay2), int(by2))
    return (x1, y1, x2, y2)


def iou(a: Box, b: Box) -> float:
    """Intersection over Union for axis-aligned boxes."""

    inter = _intersection(a, b)
    inter_a = _area(inter)
    if inter_a <= 0:
        return 0.0
    ua = _area(a) + _area(b) - inter_a
    if ua <= 0:
        return 0.0
    return float(inter_a) / float(ua)


def snap_crop_to_patch(
    patch_meta_for_file: pd.DataFrame,
    crop_box: Box,
    iou_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Snap a user crop box to the 'nearest' precomputed patch.

    This DOES NOT embed anything. It only selects a patch_id based on IoU.

    Returns a dict with at least:
      - patch_id
      - iou
      - x1,y1,x2,y2
      - filename_key

    Strategy:
    1) choose patch with max IoU
    2) if max IoU < threshold, fall back to max intersection area
    3) tie-break by area similarity to crop
    """

    if patch_meta_for_file is None or patch_meta_for_file.empty:
        raise ValueError("No patches available for this filename_key")

    df = patch_meta_for_file
    if "filename_key" not in df.columns:
        # allow callers to pass an already-filtered frame without the column
        filename_key = None
    else:
        filename_key = str(df["filename_key"].iloc[0]) if len(df) else None

    crop_a = _area(crop_box)
    if crop_a <= 0:
        raise ValueError("Invalid crop_box (zero area)")

    # Vectorized-ish loop (kept simple and readable; typical patch counts per file are small).
    best = None
    best_iou = -1.0
    best_inter = -1
    best_area_delta = 10**18
    best_key = None

    for r in df.to_dict("records"):
        pb = (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))
        v = iou(crop_box, pb)
        inter_a = _area(_intersection(crop_box, pb))
        area_delta = abs(_area(pb) - crop_a)

        # Primary: max IoU
        # Secondary: max intersection
        # Tertiary: area similarity
        key = (v, inter_a, -area_delta)
        if best is None:
            best = r
            best_iou = v
            best_inter = inter_a
            best_area_delta = area_delta
            best_key = key
            continue

        if best_key is None or key > best_key:
            best = r
            best_iou = v
            best_inter = inter_a
            best_area_delta = area_delta
            best_key = key

    assert best is not None

    # If IoU is very low, allow the UI to warn user that the crop doesn't match the grid well.
    snapped = {
        "patch_id": int(best["patch_id"]),
        "filename_key": filename_key or str(best.get("filename_key", "")),
        "x1": int(best["x1"]),
        "y1": int(best["y1"]),
        "x2": int(best["x2"]),
        "y2": int(best["y2"]),
        "iou": float(best_iou),
        "intersection_area": int(best_inter),
        "crop_area": int(crop_a),
        "patch_area_delta": int(best_area_delta),
    }

    if best_iou < float(iou_threshold):
        snapped["warning"] = f"Low IoU ({best_iou:.3f}) between crop and nearest patch"

    # Pass-through optional fields if present
    for opt in ("scale", "level", "img_w", "img_h", "stable_id"):
        if opt in best:
            snapped[opt] = best[opt]

    return snapped


def patches_for_filename(meta: pd.DataFrame, filename_key: str) -> pd.DataFrame:
    """Filter patch_meta for a single filename_key."""

    if meta is None or meta.empty:
        return pd.DataFrame()
    if "filename_key" not in meta.columns:
        raise ValueError("patch_meta missing filename_key")
    fk = str(filename_key).strip().lower()
    return meta.loc[meta["filename_key"].astype(str).str.lower() == fk].copy()
