"""Build a multi-scale patch FAISS index for Retriever Wizard.

Reads image paths from an existing index.csv (must include columns: full_path, filename).
Creates grid patches at multiple scales (default: 1x1, 2x2, 3x3, 4x4), embeds each patch with SigLIP2,
and builds an inner-product (cosine) FAISS index.

Outputs (under out_dir/patch_index/):
- patch_vectors.npy          float32, shape (N, dim)
- patch_meta.parquet/csv     schema includes patch_id, filename_key, x1,y1,x2,y2, scale, img_w,img_h, stable_id(optional)
- patch.faiss                FAISS IndexFlatIP
- patch_index_manifest.json  parameters + counts + versioning

Usage (PowerShell):
    py -3.12 build_patch_index.py --index_csv .\\index.csv --out_dir .\\output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _parse_scales(s: str) -> List[int]:
    out: List[int] = []
    for part in str(s).replace(";", ",").replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    out = [x for x in out if x >= 1]
    return sorted(list(dict.fromkeys(out)))


def _grid_boxes(w: int, h: int, s: int) -> List[Tuple[int, int, int, int]]:
    # s x s grid.
    boxes: List[Tuple[int, int, int, int]] = []
    for gy in range(s):
        y0 = int(round(gy * h / s))
        y1 = int(round((gy + 1) * h / s))
        for gx in range(s):
            x0 = int(round(gx * w / s))
            x1 = int(round((gx + 1) * w / s))
            # Ensure non-empty crops.
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append((x0, y0, x1, y1))
    return boxes


def _chunked(seq: Sequence, batch_size: int):
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True, help="Path to index.csv (needs columns full_path, filename)")
    ap.add_argument("--out_dir", required=True, help="Output directory for patch index artifacts")
    ap.add_argument("--scales", default="1,2,3,4", help="Comma-separated grid scales")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for GPU/CPU embedding")
    ap.add_argument("--limit", type=int, default=0, help="Optional: limit number of images (debug)")
    args = ap.parse_args()

    index_csv = Path(args.index_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = out_dir / "patch_index"
    patch_dir.mkdir(parents=True, exist_ok=True)

    if not index_csv.exists():
        raise FileNotFoundError(f"index_csv not found: {index_csv}")

    scales = _parse_scales(args.scales)
    if not scales:
        raise ValueError("No valid scales provided")

    df = pd.read_csv(index_csv, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "full_path" not in df.columns or "filename" not in df.columns:
        raise ValueError("index.csv must contain columns: full_path, filename")

    keep_cols = ["full_path", "filename"]
    if "stable_id" in df.columns:
        keep_cols.append("stable_id")

    rows = df[keep_cols].dropna(subset=["full_path", "filename"]).copy()
    rows["full_path"] = rows["full_path"].astype(str)
    rows["filename"] = rows["filename"].astype(str).str.strip()
    rows["filename_key"] = rows["filename"].map(lambda x: Path(str(x)).name.strip().lower())
    if "stable_id" in rows.columns:
        rows["stable_id"] = rows["stable_id"].astype(str)

    if args.limit and args.limit > 0:
        rows = rows.head(int(args.limit))

    # Lazy imports (HF + PIL + faiss can be heavy)
    from PIL import Image

    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("faiss is required (pip install faiss-cpu)") from e

    from element_encoder import _embed_pil_batch  # type: ignore

    patch_meta: List[dict] = []
    patch_images: List[Image.Image] = []

    vectors: List[np.ndarray] = []

    patch_id = 0
    n_images = int(len(rows))

    def flush_batch() -> None:
        nonlocal patch_images
        if not patch_images:
            return
        vec = _embed_pil_batch(patch_images)  # (B, dim), normalized
        vectors.append(vec)
        patch_images = []

    processed_images = 0

    for i, r in enumerate(rows.to_dict("records"), start=1):
        p = Path(r["full_path"])
        filename_key = str(r["filename_key"]).strip().lower()
        stable_id = str(r.get("stable_id", "")) if "stable_id" in rows.columns else ""
        if not p.exists():
            continue

        try:
            img = Image.open(str(p)).convert("RGB")
        except Exception:
            continue

        processed_images += 1
        w, h = img.size
        for s in scales:
            for box in _grid_boxes(w, h, s):
                x1, y1, x2, y2 = box
                crop = img.crop((x1, y1, x2, y2))

                patch_meta.append({
                    "patch_id": patch_id,
                    "filename_key": filename_key,
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "scale": int(s),
                    "img_w": int(w),
                    "img_h": int(h),
                })
                if stable_id:
                    patch_meta[-1]["stable_id"] = stable_id
                patch_id += 1

                patch_images.append(crop)
                if len(patch_images) >= int(args.batch_size):
                    flush_batch()

        if i % 25 == 0:
            print(f"Scanned {i}/{n_images} rows; processed images: {processed_images:,}; patches so far: {patch_id:,}")

    flush_batch()

    if not patch_meta:
        raise RuntimeError("No patches created (check paths / image formats)")

    X = np.vstack(vectors).astype(np.float32)

    # Save vectors + metadata
    vectors_path = patch_dir / "patch_vectors.npy"
    np.save(vectors_path, X)

    meta_df = pd.DataFrame(patch_meta)
    meta_csv_path = patch_dir / "patch_meta.csv"
    meta_parquet_path = patch_dir / "patch_meta.parquet"
    wrote_parquet = False
    try:
        meta_df.to_parquet(meta_parquet_path, index=False)
        wrote_parquet = True
    except Exception:
        wrote_parquet = False
        meta_df.to_csv(meta_csv_path, index=False, encoding="utf-8")

    dim = int(X.shape[1])
    index: Any = faiss.IndexFlatIP(dim)
    # X is already normalized; normalize again for safety
    faiss.normalize_L2(X)
    index.add(np.ascontiguousarray(X, dtype=np.float32))
    faiss_path = patch_dir / "patch.faiss"
    faiss.write_index(index, str(faiss_path))

    sha = hashlib.sha256()
    try:
        with open(index_csv, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(chunk)
        index_csv_sha256 = sha.hexdigest()
    except Exception:
        index_csv_sha256 = ""

    model_id = os.environ.get("REWIZ_SIGLIP2_MODEL", "google/siglip2-giant-opt-patch16-384").strip() or "google/siglip2-giant-opt-patch16-384"
    manifest = {
        "schema_version": 1,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_id": model_id,
        "dim": int(dim),
        "scales": scales,
        "images_processed": int(processed_images),
        "patches": int(len(meta_df)),
        "index_csv": str(index_csv.resolve()),
        "index_csv_sha256": index_csv_sha256,
        "meta": {
            "format": "parquet" if wrote_parquet else "csv",
            "path": str((meta_parquet_path if wrote_parquet else meta_csv_path).resolve()),
            "columns": list(meta_df.columns),
        },
        "vectors": {
            "format": "npy",
            "path": str(vectors_path.resolve()),
            "dtype": "float32",
            "shape": [int(X.shape[0]), int(X.shape[1])],
        },
        "faiss": {
            "path": str(faiss_path.resolve()),
            "type": "IndexFlatIP",
            "normalized": True,
        },
    }
    (patch_dir / "patch_index_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {vectors_path}")
    print(f"Wrote: {meta_parquet_path if wrote_parquet else meta_csv_path}")
    print(f"Wrote: {faiss_path}")
    print(f"Wrote: {patch_dir / 'patch_index_manifest.json'}")
    print(f"Patches: {len(meta_df):,}; dim={dim}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
