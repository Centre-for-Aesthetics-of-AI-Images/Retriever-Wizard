# -*- coding: utf-8 -*-
# pyright: reportGeneralTypeIssues=false
"""
🧙🏻‍♂️ Retriever Wizard (Streamlit)

✅ Install (once):
    pip install streamlit pandas numpy faiss-cpu

(Optional for Step 9 projection)
    pip install umap-learn scikit-learn plotly altair

▶️ Run:
    streamlit run ReWiz.py
"""

import os
import re
import json
import csv
import time
import random
import hashlib
import subprocess
import sys
import logging
import importlib.util
import importlib.metadata
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import streamlit as st


_REWIZ_DEBUG = str(os.environ.get("REWIZ_DEBUG", "")).strip().lower() in {"1", "true", "yes", "y", "on"}
_LOG = logging.getLogger("rewiz")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=(logging.DEBUG if _REWIZ_DEBUG else logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )


def _dbg_exc(msg: str) -> None:
    if _REWIZ_DEBUG:
        _LOG.debug(msg, exc_info=True)


# ---------- Optional deps (patch backend) ----------
try:
    import patch_backend as _patch_backend  # type: ignore
    _PATCH_BACKEND_OK = True
except Exception:
    _patch_backend = None  # type: ignore[assignment]
    _PATCH_BACKEND_OK = False


# ---------- Optional deps (element search) ----------
try:
    from element_encoder import embed_text as _embed_text
    from element_encoder import embed_image as _embed_image
    from element_encoder import clear_model_cache as _clear_element_cache
    from element_encoder import current_model_id as _current_element_model_id
    _ELEMENT_OK = True
except Exception:
    _embed_text = None  # type: ignore[assignment]
    _embed_image = None  # type: ignore[assignment]
    _clear_element_cache = None  # type: ignore[assignment]
    _current_element_model_id = None  # type: ignore[assignment]
    _ELEMENT_OK = False


def _sidebar_env_status() -> None:
    # Diagnostics for venv mismatch issues (e.g. torch installed in venv, but Streamlit launched elsewhere)
    with st.sidebar.expander("Environment", expanded=False):
        st.caption(f"Python: {sys.executable}")
        # Avoid importing heavy packages at startup (torch/transformers import time can be seconds).
        # Use metadata/spec checks instead.
        def _pkg_status(dist_name: str, import_name: Optional[str] = None) -> str:
            try:
                v = importlib.metadata.version(dist_name)
                return str(v)
            except Exception:
                pass
            name = import_name or dist_name
            try:
                return "installed" if importlib.util.find_spec(name) is not None else "missing"
            except Exception:
                return "unknown"

        st.caption(f"torch: {_pkg_status('torch', 'torch')}")
        st.caption(f"transformers: {_pkg_status('transformers', 'transformers')}")

        st.caption(f"element encoder: {'ok' if _ELEMENT_OK else 'missing'}")
        try:
            mid = None
            if _ELEMENT_OK and _current_element_model_id is not None:
                mid = str(_current_element_model_id())
            else:
                mid = os.environ.get("REWIZ_SIGLIP2_MODEL", "")
            if mid:
                st.caption(f"SigLIP2 model: {mid}")
        except Exception:
            _dbg_exc("Env status: failed to determine SigLIP2 model id")

        st.caption(f"cropper ui: {'ok' if _CROPPER_OK else 'missing'}")
        if not _CROPPER_OK and _CROPPER_IMPORT_ERROR:
            st.caption(f"cropper detail: {_CROPPER_IMPORT_ERROR}")

_st_cropper = None  # type: ignore[assignment]
_CROPPER_OK = False
_CROPPER_IMPORT_ERROR: Optional[str] = None

try:
    # streamlit-cropperjs API names vary slightly across versions.
    import streamlit_cropperjs as _scj  # type: ignore[reportMissingImports]

    if hasattr(_scj, "st_cropperjs"):
        _st_cropper = getattr(_scj, "st_cropperjs")
        _CROPPER_OK = True
    elif hasattr(_scj, "st_cropper"):
        _st_cropper = getattr(_scj, "st_cropper")
        _CROPPER_OK = True
    else:
        _CROPPER_OK = False
        _CROPPER_IMPORT_ERROR = "streamlit_cropperjs imported, but no st_cropperjs/st_cropper found"
except Exception as e:
    _CROPPER_OK = False
    _CROPPER_IMPORT_ERROR = repr(e)


# Safe rerun helper — some Streamlit versions don't expose experimental_rerun
def safe_rerun() -> None:
    for fn in ("experimental_rerun", "script_request_rerun", "rerun", "request_rerun"):
        f = getattr(st, fn, None)
        if callable(f):
            try:
                f()
                return
            except Exception:
                continue
    # Fallback: set a flag and stop execution with a user message
    try:
        st.warning("Reload the app (refresh) to apply changes.")
        st.stop()
    except Exception:
        _dbg_exc("safe_rerun: fallback warning/stop failed")

# ---------- Optional deps ----------
try:
    import faiss  # type: ignore  # pip install faiss-cpu
    _FAISS_OK = True
except Exception:
    faiss = None  # type: ignore[assignment]
    _FAISS_OK = False

try:
    import plotly  # noqa: F401
    _PLOTLY_OK = True
except Exception:
    plotly = None  # type: ignore[assignment]
    _PLOTLY_OK = False

try:
    import altair as alt
    _ALTAIR_OK = True
except Exception:
    alt = None  # type: ignore[assignment]
    _ALTAIR_OK = False

try:
    from umap import UMAP  # type: ignore  # pip install umap-learn
    _UMAP_OK = True
except Exception:
    UMAP = None  # type: ignore[assignment]
    _UMAP_OK = False

try:
    from sklearn.manifold import TSNE  # type: ignore  # pip install scikit-learn
    _TSNE_OK = True
except Exception:
    TSNE = None  # type: ignore[assignment]
    _TSNE_OK = False


def _to_dense_2d(a: Any) -> np.ndarray:
    """Convert array-like output (incl. sparse matrices) to a dense 2D ndarray."""
    if hasattr(a, "toarray"):
        a = a.toarray()
    arr = np.asarray(a)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.asarray(arr, dtype="float32")


# ---------- App config ----------
st.set_page_config(page_title="Retriever Wizard", page_icon="🧙🏻‍♂️", layout="wide")
st.title("🧙🏻‍♂️ Retriever Wizard")
st.caption("Flow: metadata -> embeddings -> images -> check -> index -> query -> results -> stacked view -> annotate -> projection")

# Strong hint when Streamlit is launched from the wrong interpreter (common cause of missing torch/cropper deps).
try:
    exe = str(sys.executable)
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    venv_py = here / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists() and Path(exe).resolve() != venv_py.resolve():
        st.warning(
            "Streamlit is not running from this project's .venv. "
            "Install/import errors (e.g. torch, streamlit-cropperjs) are expected. "
            f"\n\nRun: `{venv_py} -m streamlit run {here / 'ReWiz.py'}`"
        )
except Exception:
    _dbg_exc("Interpreter mismatch check failed")

SUPPORTED_IMG = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
# Persist user settings in a folder inside the Retriever Wizard project directory.
# Use a hidden folder named `.retriever_wizard` located next to this script.
try:
    _HERE = Path(__file__).resolve().parent
except Exception:
    _HERE = Path.cwd()
CHECKPOINT_DIR = _HERE / ".retriever_wizard"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = CHECKPOINT_DIR / "settings.json"

# Optional repo-level defaults file so users can change default paths without editing code.
DEFAULTS_FILE = _HERE / "rewiz_default_paths.json"


# ---------- Utils ----------
def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _basename(x: str) -> str:
    x = str(x or "").strip().strip('"').strip("'")
    x = x.replace("\\", os.sep).replace("/", os.sep)
    return Path(x).name


def _fname_key(x: str) -> str:
    return _basename(x).lower()


def _ensure_filename(df: pd.DataFrame) -> pd.DataFrame:
    df = _std_cols(df)
    if "filename" in df.columns:
        df["filename"] = df["filename"].astype(str).map(_fname_key)
        return df

    # common alternatives
    for c in ["file_name", "original_filename", "image", "img", "name"]:
        if c in df.columns:
            df["filename"] = df[c].astype(str).map(_fname_key)
            return df

    for c in ["full_path", "path", "filepath", "file_path"]:
        if c in df.columns:
            df["filename"] = df[c].astype(str).map(_fname_key)
            return df

    return df


def _ensure_stable_id(df: pd.DataFrame) -> pd.DataFrame:
    df = _std_cols(df)
    if "stable_id" in df.columns:
        df["stable_id"] = df["stable_id"].astype(str).map(lambda x: str(x).strip())
    return df


def _has_key(df: pd.DataFrame, origin: str) -> None:
    cols = {str(c).strip().lower() for c in getattr(df, "columns", [])}
    if "stable_id" in cols:
        return
    if "filename" in cols:
        return
    raise ValueError(f"{origin} is missing a 'stable_id' or 'filename' column (or a recognizable alternative column).")


def _has_filename(df: pd.DataFrame, origin: str) -> None:
    if "filename" not in df.columns:
        raise ValueError(f"{origin} is missing a 'filename' column (or a recognizable alternative column).")


def parse_roots(root_field: str) -> List[str]:
    if not root_field:
        return []
    tmp = root_field
    for sep in [";", "|", ","]:
        tmp = tmp.replace(sep, "\n")
    roots = [Path(p.strip().strip('"').strip("'")).expanduser() for p in tmp.splitlines() if p.strip()]
    roots = [str(p) for p in roots if p.exists()]
    return roots


def scan_images(roots: List[str]) -> List[str]:
    out: List[str] = []
    stack = list(roots)
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    elif Path(e.name).suffix.lower() in SUPPORTED_IMG:
                        out.append(e.path)
        except Exception:
            pass
    return out


def display_image(path: str, width: int = 360, caption: Optional[str] = None) -> None:
    cap = caption or Path(path).name
    try:
        st.image(path, caption=cap, width=width)
        return
    except Exception:
        pass

    # fallback: raw bytes
    try:
        with open(path, "rb") as f:
            st.image(f.read(), caption=cap, width=width)
    except Exception as e:
        st.warning(f"Could not display image: {e}")


def _st_width_stretch() -> dict:
    """Streamlit removed `use_container_width` after 2025-12-31.

    Prefer `width='stretch'` when supported; fall back to legacy kwarg for older versions.
    """
    try:
        # New Streamlit API
        return {"width": "stretch"}
    except Exception:
        return {"use_container_width": True}


def st_dataframe_stretch(df: Any, **kwargs) -> None:
    try:
        st.dataframe(df, **_st_width_stretch(), **kwargs)
    except TypeError:
        # Fallback for older/newer signature mismatches
        try:
            st.dataframe(df, use_container_width=True, **kwargs)
        except Exception:
            st.dataframe(df, **kwargs)


def st_plotly_chart_stretch(fig: Any, **kwargs) -> None:
    try:
        st.plotly_chart(fig, **_st_width_stretch(), **kwargs)
    except TypeError:
        try:
            st.plotly_chart(fig, use_container_width=True, **kwargs)
        except Exception:
            st.plotly_chart(fig, **kwargs)


def st_vega_lite_chart_stretch(data: Any, spec: Any, **kwargs) -> None:
    try:
        st.vega_lite_chart(data, spec, **_st_width_stretch(), **kwargs)
    except TypeError:
        try:
            st.vega_lite_chart(data, spec, use_container_width=True, **kwargs)
        except Exception:
            st.vega_lite_chart(data, spec, **kwargs)


def _safe_write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _resolve_default_path(p: Any) -> Any:
    """Resolve relative filesystem paths relative to the repo root."""
    if not isinstance(p, str):
        return p
    s = p.strip().strip('"').strip("'")
    if not s:
        return s
    try:
        pp = Path(s)
        if not pp.is_absolute():
            return str((_HERE / pp).resolve())
    except Exception:
        return s
    return s


def load_repo_defaults() -> dict:
    """Load defaults from `rewiz_default_paths.json` if present; otherwise fall back to example-style defaults."""
    # Safe fallback defaults (good for the public GitHub example repo).
    fallback = {
        "meta_path": str((_HERE / "examples" / "metadata.csv").resolve()),
        "embed_path": str((_HERE / "examples" / "embeddings.csv").resolve()),
        "images_root": str((_HERE / "examples" / "images").resolve()),
        "output_dir": str((_HERE / "examples" / "_index").resolve()),
        "index_name": "index.csv",
        "auto_load_index": True,
        "index_metric": "Cosine (IP + normalization)",
        "k_neighbors": 10,
        "query_mode": "Whole-image",
        "overlay_session": "iconography_2025q4",
        "overlay_marker": "ikonografi",
    }

    p = DEFAULTS_FILE
    if not p.exists():
        return fallback

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return fallback
    except Exception as e:
        # Keep app usable even if defaults file is broken.
        st.session_state["_defaults_load_message"] = f"Failed to parse {p.name}: {e} (using built-in defaults)"
        return fallback

    out = dict(fallback)
    out.update({k: v for k, v in data.items() if k in fallback})
    # Resolve relative paths for filesystem fields.
    for k in ("meta_path", "embed_path", "images_root", "output_dir"):
        out[k] = _resolve_default_path(out.get(k))
    return out


def save_checkpoint() -> None:
    # Only persist the core paths used in Settings so checkpoints behave like a simple preset
    keep = ["meta_path", "embed_path", "images_root", "output_dir"]
    data = {k: st.session_state.get(k) for k in keep}
    _safe_write_text(CHECKPOINT, json.dumps(data, ensure_ascii=False, indent=2))


def load_checkpoint() -> bool:
    # Try several candidate locations to be robust
    candidates = [
        CHECKPOINT,
        Path.cwd() / ".retriever_wizard" / "settings.json",
        Path.home() / ".retriever_wizard" / "settings.json",
    ]
    tried = []
    for p in candidates:
        p = Path(p)
        tried.append(str(p))
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            st.session_state["_checkpoint_load_message"] = f"Found file {p} but failed to parse JSON: {e}"
            return False

        # Only keep the allowed settings (paths); sanitize string quotes if present
        loaded = {}
        for k in ("meta_path", "embed_path", "images_root", "output_dir"):
            if k in data:
                val = data.get(k)
                if isinstance(val, str):
                    val = val.strip().strip('"').strip("'")
                loaded[k] = val

        # Store loaded values in a temporary session_state key so they can be applied
        # before widget instantiation on the next rerun (avoids Streamlit API errors).
        st.session_state["_loaded_checkpoint_values"] = loaded
        st.session_state["_checkpoint_load_path"] = str(p)
        st.session_state["_checkpoint_load_message"] = f"Loaded settings from: {p} (will apply after reload)"
        return True

    st.session_state["_checkpoint_load_message"] = f"No checkpoint file found. Checked: {', '.join(tried)}"
    return False


def _is_writable_dir(d: str) -> bool:
    try:
        p = Path(d)
        p.mkdir(parents=True, exist_ok=True)
        t = p / ".write_test"
        t.write_text("ok", encoding="utf-8")
        t.unlink(missing_ok=True)
        return True
    except Exception:
        return False


# ---------- FAISS + similarity ----------
@st.cache_data(show_spinner=False)
def load_metadata(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = _ensure_stable_id(df)
    # If stable_id is present we can join on it; still try to derive filename for display.
    df = _ensure_filename(df)
    _has_key(df, "Metadata CSV")
    return df


@st.cache_data(show_spinner=False)
def load_embeddings(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = _ensure_stable_id(df)
    df = _ensure_filename(df)
    _has_key(df, "Embeddings CSV")
    vec_cols = [c for c in df.columns if str(c).strip().lower() not in ("filename", "stable_id")]
    if not vec_cols:
        raise ValueError("Embeddings CSV has no vector columns (only filename).")
    df[vec_cols] = df[vec_cols].astype("float32")
    return df


@st.cache_data(show_spinner=False)
def validate_embeddings_sample(path: str, sample_rows: int = 2000) -> Tuple[pd.DataFrame, int]:
    """Fast validation for large embeddings CSVs.

    Reads only a small sample to confirm a usable filename column and numeric vector columns.
    Returns (head_df, dim).
    """

    n = int(max(50, sample_rows))
    df = pd.read_csv(path, low_memory=False, nrows=n)
    df = _ensure_filename(df)
    _has_filename(df, "Embeddings CSV")

    vec_cols = [c for c in df.columns if str(c).strip().lower() not in ("filename", "stable_id")]
    if not vec_cols:
        raise ValueError("Embeddings CSV has no vector columns (only filename).")

    # Coerce only the sample to numeric to catch obvious format issues quickly.
    for c in vec_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[vec_cols].isna().all(axis=None):
        raise ValueError("Embeddings sample contains no numeric values in vector columns.")

    head = df.head(20).copy()
    head[vec_cols] = head[vec_cols].astype("float32")
    return head, int(len(vec_cols))


@st.cache_data(show_spinner=False)
def load_embeddings_filenames_only(path: str) -> pd.Series:
    """Load only embedding filenames (fast), used for consistency checks."""

    # Read header to decide which columns are safe to load.
    header = pd.read_csv(path, nrows=0, low_memory=False)
    cols = [str(c).strip() for c in header.columns]
    cols_l = [c.lower() for c in cols]

    want = {
        "filename",
        "file_name",
        "original_filename",
        "image",
        "img",
        "name",
        "full_path",
        "path",
        "filepath",
        "file_path",
    }
    usecols = [cols[i] for i, cl in enumerate(cols_l) if cl in want]

    if not usecols:
        # Let the existing helpers produce a clear error message.
        df = pd.read_csv(path, nrows=50, low_memory=False)
        df = _ensure_filename(df)
        _has_filename(df, "Embeddings CSV")
        return df["filename"].astype(str)

    df = pd.read_csv(path, low_memory=False, usecols=usecols)
    df = _ensure_filename(df)
    _has_filename(df, "Embeddings CSV")
    return df["filename"].astype(str)


def _l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return mat / n


def _cosine_similarity(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    qn = _l2_normalize(q)
    Mn = _l2_normalize(M)
    return (qn @ Mn.T).ravel()


def _match_label(sim: float) -> str:
    if sim >= 0.90: return "★★★★★ very close"
    if sim >= 0.80: return "★★★★ close"
    if sim >= 0.70: return "★★★ related"
    if sim >= 0.60: return "★★ loose"
    return "★ weak"


def _query_key_for_faiss(qpath: str, key_kind: str) -> Optional[str]:
    """Return the correct query key to look up the query vector inside FAISS.

    - If FAISS keys are `filename`: always use the normalized basename key.
    - If FAISS keys are `stable_id`: prefer an existing stable_id key, else compute from file stats.
    """
    try:
        p = Path(qpath)
    except Exception:
        return None
    if not (qpath and p.exists()):
        return None

    kk = str(key_kind)
    if kk != "stable_id":
        # Always use filename key for filename-keyed FAISS.
        return _fname_key(p.name)

    # stable_id-keyed FAISS.
    qk = str(st.session_state.get("query_key", "")).strip()
    # Heuristic: our stable_id strings are 16 hex chars.
    if len(qk) == 16 and all(ch in "0123456789abcdef" for ch in qk.lower()):
        return qk

    # Try cached stable_id first (if present).
    qsid = str(st.session_state.get("query_stable_id", "")).strip()
    if len(qsid) == 16 and all(ch in "0123456789abcdef" for ch in qsid.lower()):
        return qsid

    # Compute stable_id from file.
    try:
        stt = p.stat()
        return _stable_id(p, int(stt.st_size), int(stt.st_mtime))
    except Exception:
        return None


def build_faiss_index(emb_path: str, metric: str, progress_cb: Optional[Callable[[int, int], None]] = None):
    if not _FAISS_OK:
        raise RuntimeError("FAISS not installed. Run: pip install faiss-cpu")
    assert faiss is not None

    import re

    # Stream embeddings CSV in chunks to avoid OOM on large files.
    # We detect vector columns robustly:
    # 1) If columns match ^(v|emb_)\d+$ -> use only those (sorted numerically)
    # 2) Else -> include columns that parse as float on a 200-row sample,
    #           excluding common non-embedding columns (stable_id, width/height, year, etc.)
    emb_path = str(emb_path)
    header = pd.read_csv(emb_path, nrows=0, low_memory=False)
    orig_cols = [str(c) for c in header.columns]
    lower_cols = [str(c).strip().lower() for c in orig_cols]
    orig_by_lower = dict(zip(lower_cols, orig_cols))

    has_stable_id = "stable_id" in set(lower_cols)
    key_kind = "stable_id" if has_stable_id else "filename"

    # --- vec_cols detection ---
    vec_re = re.compile(r"^(?:v|emb_)(\d+)$")
    vec_regex_cols = [c for c in lower_cols if vec_re.match(c or "")]
    if vec_regex_cols:
        # Sort numerically by the suffix.
        vec_cols_lower = sorted(vec_regex_cols, key=lambda c: int(vec_re.match(c).group(1)))  # type: ignore[union-attr]
    else:
        exclude = {
            # identifiers / joins
            "stable_id",
            "filename",
            "file_name",
            "original_filename",
            "image",
            "img",
            "name",
            "full_path",
            "path",
            "filepath",
            "file_path",
            # common non-embedding numeric columns
            "width",
            "height",
            "w",
            "h",
            "size_bytes",
            "size",
            "bytes",
            "mtime",
            "timestamp",
            "year",
            "id",
            "index",
            "patch_id",
        }
        try:
            sample = pd.read_csv(emb_path, nrows=200, low_memory=False)
            sample.columns = [str(c).strip().lower() for c in sample.columns]
        except Exception as e:
            raise RuntimeError(f"Failed reading embeddings sample for vec_cols detection: {e}")

        vec_cols_lower = []
        for c in lower_cols:
            if c in exclude:
                continue
            if c not in sample.columns:
                continue
            ser = pd.to_numeric(sample[c], errors="coerce")
            non_nan = float(ser.notna().mean()) if len(ser) else 0.0
            if non_nan >= 0.90:
                vec_cols_lower.append(c)

    if not vec_cols_lower:
        raise ValueError("Embeddings CSV has no detectable vector columns (after filtering).")

    # --- columns to read ---
    want_filenameish = {
        "filename",
        "file_name",
        "original_filename",
        "image",
        "img",
        "name",
        "full_path",
        "path",
        "filepath",
        "file_path",
    }
    id_cols_lower = [c for c in lower_cols if c in want_filenameish]
    usecols_lower: List[str] = []
    if has_stable_id:
        usecols_lower.append("stable_id")
    else:
        # We need at least something to derive filename keys.
        usecols_lower.extend(id_cols_lower)

    usecols_lower.extend(vec_cols_lower)
    usecols_orig = [orig_by_lower[c] for c in usecols_lower if c in orig_by_lower]
    if not usecols_orig:
        raise ValueError("Could not determine columns to read from embeddings CSV.")

    dim = int(len(vec_cols_lower))
    if "Cosine" in metric:
        index: Any = faiss.IndexFlatIP(dim)
        do_norm = True
    else:
        index = faiss.IndexFlatL2(dim)
        do_norm = False

    # Count rows for a meaningful progress bar (streaming; no extra memory).
    total_rows = 0
    try:
        with open(emb_path, "rb") as f:
            # subtract header line
            total_rows = max(0, sum(1 for _ in f) - 1)
    except Exception:
        total_rows = 0

    keys: List[str] = []
    added = 0
    chunksize = 2000

    for chunk in pd.read_csv(emb_path, low_memory=False, usecols=usecols_orig, chunksize=chunksize):
        chunk.columns = [str(c).strip().lower() for c in chunk.columns]
        chunk = _ensure_stable_id(chunk)
        if not has_stable_id:
            chunk = _ensure_filename(chunk)
        _has_key(chunk, "Embeddings CSV")

        # Coerce to float32 robustly (handles occasional non-numeric values without crashing)
        vec_df = chunk[vec_cols_lower].copy()
        for c in vec_cols_lower:
            vec_df[c] = pd.to_numeric(vec_df[c], errors="coerce")
        X = vec_df.to_numpy(dtype="float32", copy=False)
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        X = np.ascontiguousarray(X, dtype="float32")

        if do_norm:
            faiss.normalize_L2(X)

        index.add(X)
        if has_stable_id and "stable_id" in chunk.columns:
            chunk_keys = chunk["stable_id"].astype(str).map(lambda x: str(x).strip()).tolist()
        else:
            chunk_keys = chunk["filename"].astype(str).tolist()
        keys.extend(chunk_keys)
        added += int(len(chunk_keys))

        if progress_cb is not None:
            progress_cb(added, int(total_rows or added))

    if added <= 0:
        raise RuntimeError("No vectors were added to the FAISS index (empty embeddings CSV?)")

    # Build a filename->position map (first occurrence wins). This avoids O(N) list scans at query time.
    pos_map: Dict[str, int] = {}
    try:
        for i, k in enumerate(keys):
            if k not in pos_map:
                pos_map[k] = i
    except Exception:
        pos_map = {}

    vec_cols_orig = [orig_by_lower.get(c, c) for c in vec_cols_lower]
    return index, keys, dim, pos_map, vec_cols_orig, key_kind


def _faiss_search(index, qvec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    res: Any = index.search(np.ascontiguousarray(qvec.astype("float32"), dtype="float32"), int(k))
    if isinstance(res, tuple) and len(res) >= 2:
        D, I = res[0], res[1]
    else:
        D = getattr(res, "distances", None)
        I = getattr(res, "labels", None)
        if D is None or I is None:
            raise RuntimeError("Ukendt FAISS.search() returtype.")
    D = np.asarray(D)
    I = np.asarray(I)
    if D.ndim == 1: D = D[None, :]
    if I.ndim == 1: I = I[None, :]
    return D, I


def _faiss_dim(index: Any) -> Optional[int]:
    try:
        d = getattr(index, "d", None)
        return int(d) if d is not None else None
    except Exception:
        return None


def _faiss_metric_label_from_index(index: Any) -> Optional[str]:
    """Best-effort detection of the metric implied by the FAISS index instance."""
    try:
        if not _FAISS_OK or faiss is None or index is None:
            return None
        mt = getattr(index, "metric_type", None)
        if mt is not None:
            try:
                if int(mt) == int(getattr(faiss, "METRIC_INNER_PRODUCT")):
                    return "Cosine (IP + normalization)"
                if int(mt) == int(getattr(faiss, "METRIC_L2")):
                    return "L2 (squared)"
            except Exception:
                pass
        # Fallback: inspect class name
        name = type(index).__name__.lower()
        if "flatip" in name or name.endswith("ip"):
            return "Cosine (IP + normalization)"
        if "flatl2" in name or name.endswith("l2"):
            return "L2 (squared)"
    except Exception:
        return None
    return None


def _assert_query_dim_matches_index(qvec: np.ndarray, index: Any, label: str) -> None:
    if qvec is None:
        raise RuntimeError(f"{label} produced no vector")
    if qvec.ndim == 1:
        qd = int(qvec.shape[0])
    else:
        qd = int(qvec.shape[-1])
    fd = _faiss_dim(index)
    if fd is not None and qd != int(fd):
        raise RuntimeError(
            f"{label} dim {qd} != FAISS dim {fd}. "
            "This usually means your embeddings CSV was generated with a different model than the runtime element encoder. "
            "Fix: regenerate embeddings with the same model as element search (SigLIP2), then rebuild FAISS."
        )


def _cosine_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype="float32").ravel()
    b = np.asarray(b, dtype="float32").ravel()
    an = a / max(float(np.linalg.norm(a)), 1e-12)
    bn = b / max(float(np.linalg.norm(b)), 1e-12)
    return float((an * bn).sum())


def compute_element_alignment_sample() -> Tuple[Optional[float], str]:
    """Compute cosine(element_encoder_image, FAISS.reconstruct) on a sample item.

    Returns (cosine or None, message).
    """
    if not _ELEMENT_OK or _embed_image is None:
        return None, "Element encoder not available"
    if not st.session_state.get("faiss_ready"):
        return None, "FAISS not ready"

    index = st.session_state.get("_faiss_index")
    pos_map = st.session_state.get("_faiss_pos_map") or {}
    key_kind = str(st.session_state.get("_faiss_key_kind", "filename"))
    fm = st.session_state.get("filemap", {}) or {}
    idx_df = st.session_state.get("index_df", pd.DataFrame())
    sid_map = build_sid_map(idx_df) if str(st.session_state.get("_index_key_kind", "filename")) == "stable_id" else {}

    if index is None or not pos_map:
        return None, "FAISS index/pos_map missing"

    # Try to find a key that we can resolve to an on-disk image path.
    sample_key = None
    sample_path = None
    for k in list(pos_map.keys())[:2000]:
        if key_kind == "stable_id":
            p = fm.get(str(k), "")
        else:
            p = fm.get(str(k), "")
            if not p:
                # If index is stable_id-keyed but FAISS is filename-keyed, map filename->stable_id
                sid = sid_map.get(_fname_key(str(k)), "")
                if sid:
                    p = fm.get(str(sid), "")
        if p and Path(p).exists():
            sample_key = str(k)
            sample_path = str(p)
            break

    if not sample_key or not sample_path:
        return None, "Could not resolve a sample FAISS key to a file path (check Step 4 index + filemap)"

    try:
        pos = int(pos_map[sample_key])
        v_faiss = np.asarray(index.reconstruct(pos), dtype="float32")
        v_elem = np.asarray(_embed_image(sample_path, crop_box=None), dtype="float32")
        if v_faiss.shape[-1] != v_elem.shape[-1]:
            return None, f"Dim mismatch: FAISS {v_faiss.shape[-1]} vs element {v_elem.shape[-1]}"
        cos = _cosine_1d(v_faiss, v_elem)
        return cos, f"Sample key: {sample_key}"
    except Exception as e:
        return None, f"Failed computing alignment: {e}"


def knn_filtered(
    index,
    index_filenames: List[str],
    query_key: str,
    k: int,
    allowed: Optional[Set[str]],
    metric: str,
    pos_map: Optional[Dict[str, int]] = None,
    key_kind: str = "filename",
) -> pd.DataFrame:
    """Whole-image query using an existing FAISS index.

    This avoids re-loading the full embeddings CSV (which can OOM on large files)
    by reconstructing the query vector from the FAISS index.
    """

    qkey = str(query_key).strip() if str(key_kind) == "stable_id" else _fname_key(str(query_key))
    qpos = None
    if pos_map is not None and qkey in pos_map:
        qpos = int(pos_map[qkey])
    else:
        try:
            qpos = int(index_filenames.index(qkey))
        except Exception:
            qpos = None
    if qpos is None or qpos < 0:
        raise ValueError(f"Query key '{qkey}' not found in FAISS keys.")

    try:
        qvec = np.asarray(index.reconstruct(int(qpos)), dtype="float32").reshape(1, -1)
    except Exception as e:
        raise RuntimeError("Could not reconstruct query vector from FAISS index") from e

    # Adaptive oversampling loop: start with k*50, double until k results found or cap reached
    max_index_size = len(index_filenames)
    k_search = min(max(k * 50, 200), max_index_size)
    rows = []

    if max_index_size <= 0:
        return pd.DataFrame([])

    while True:
        # Always perform at least one search (even when k_search == max_index_size)
        D, I = _faiss_search(index, qvec, k_search)
        D0, I0 = D[0], I[0]

        rows = []
        for pos in range(len(I0)):
            idx = int(I0[pos])
            if not (0 <= idx < len(index_filenames)):
                continue
            fn = str(index_filenames[idx])
            if fn == qkey:
                continue
            if allowed is not None and fn not in allowed:
                continue

            raw = float(D0[pos])
            distance = (1.0 - raw) if "Cosine" in metric else raw
            row = {"rank": len(rows) + 1, "filename": fn, "distance": distance, "raw_score": raw}
            if str(key_kind) == "stable_id":
                row["stable_id"] = fn
            rows.append(row)
            if len(rows) >= k:
                break

        if len(rows) >= k:
            break
        if k_search >= max_index_size:
            break
        k_search = min(k_search * 2, max_index_size)

    return pd.DataFrame(rows)


def add_cosine_columns_from_faiss(
    df: pd.DataFrame,
    index,
    index_filenames: List[str],
    query_key: str,
    metric: str,
    pos_map: Optional[Dict[str, int]] = None,
    key_kind: str = "filename",
) -> pd.DataFrame:
    """Add cosine similarity columns without loading embeddings CSV.

    For Cosine/IP indices, we can derive cosine similarity directly from raw scores.
    For L2 indices, we reconstruct vectors for the (small) result set and compute cosine.
    """

    if df is None or df.empty:
        return df

    df = df.copy()
    if "Cosine" in metric and "raw_score" in df.columns:
        sims_arr = np.asarray(df["raw_score"].astype(float).to_numpy(), dtype="float32")
        df["cosine_similarity"] = sims_arr
        df["cosine_distance"] = 1.0 - sims_arr
        df["score_0_100"] = np.round(((sims_arr + 1.0) / 2.0) * 100.0, 1)
        df["match_quality"] = [_match_label(float(s)) for s in sims_arr]
        return df

    qkey = str(query_key).strip() if str(key_kind) == "stable_id" else _fname_key(str(query_key))
    qpos = None
    if pos_map is not None and qkey in pos_map:
        qpos = int(pos_map[qkey])
    else:
        try:
            qpos = int(index_filenames.index(qkey))
        except Exception:
            qpos = None
    if qpos is None or qpos < 0:
        df["cosine_similarity"] = np.nan
        df["cosine_distance"] = np.nan
        df["score_0_100"] = np.nan
        df["match_quality"] = "n/a"
        return df

    qv = np.asarray(index.reconstruct(int(qpos)), dtype="float32").reshape(1, -1)

    # build a quick lookup for the small result set
    local_map: Dict[str, int] = {}
    if pos_map is not None and pos_map:
        local_map = pos_map
    else:
        # fallback: O(N) lookups per row (should only happen on small datasets)
        local_map = {}

    sims: List[float] = []
    for fn in df["filename"].astype(str).tolist():
        key = str(fn).strip() if str(key_kind) == "stable_id" else _fname_key(fn)
        ipos = None
        if local_map and key in local_map:
            ipos = int(local_map[key])
        else:
            try:
                ipos = int(index_filenames.index(key))
            except Exception:
                ipos = None
        if ipos is None:
            sims.append(float("nan"))
            continue
        vv = np.asarray(index.reconstruct(int(ipos)), dtype="float32").reshape(1, -1)
        sims.append(float(_cosine_similarity(qv, vv)[0]))

    sims_arr = np.asarray(sims, dtype="float32")
    df["cosine_similarity"] = sims_arr
    df["cosine_distance"] = 1.0 - sims_arr
    df["score_0_100"] = np.round(((sims_arr + 1.0) / 2.0) * 100.0, 1)
    df["match_quality"] = [_match_label(float(s)) if np.isfinite(s) else "n/a" for s in sims_arr]
    return df


def knn_filtered_by_vector(
    index,
    index_filenames: List[str],
    qvec: np.ndarray,
    k: int,
    allowed: Optional[Set[str]],
    metric: str,
    key_kind: str = "filename",
) -> pd.DataFrame:
    """KNN search with an explicit query vector (used for text/crop element search)."""

    if qvec.ndim == 1:
        qvec = qvec.reshape(1, -1)
    qvec = np.ascontiguousarray(qvec.astype("float32"), dtype="float32")

    if "Cosine" in metric:
        nrm = np.linalg.norm(qvec, axis=1, keepdims=True)
        nrm[nrm == 0.0] = 1e-12
        qvec = qvec / nrm

    max_index_size = len(index_filenames)
    k_search = min(max(k * 50, 200), max_index_size)
    rows = []

    if max_index_size <= 0:
        return pd.DataFrame([])

    while True:
        D, I = _faiss_search(index, qvec, k_search)
        D0, I0 = D[0], I[0]

        rows = []
        for pos in range(len(I0)):
            idx = int(I0[pos])
            if not (0 <= idx < len(index_filenames)):
                continue
            fn = str(index_filenames[idx])
            if allowed is not None and fn not in allowed:
                continue

            raw = float(D0[pos])
            distance = (1.0 - raw) if "Cosine" in metric else raw
            row = {"rank": len(rows) + 1, "filename": fn, "distance": distance, "raw_score": raw}
            if str(key_kind) == "stable_id":
                row["stable_id"] = fn
            rows.append(row)
            if len(rows) >= k:
                break

        if len(rows) >= k:
            break
        if k_search >= max_index_size:
            break
        k_search = min(k_search * 2, max_index_size)

    return pd.DataFrame(rows)


def _draw_boxes_on_image(path: str, boxes: List[Tuple[int, int, int, int]], color: str = "red"):
    from PIL import Image, ImageDraw

    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for (x0, y0, x1, y1) in boxes:
        draw.rectangle([int(x0), int(y0), int(x1), int(y1)], outline=color, width=4)
    return img


def _crop_box_ui(path: str, require_coords: bool = False) -> Optional[Tuple[int, int, int, int]]:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size

    st.caption("Select a crop region (used as query).")
    st.caption("Use crop gemmer crop’et som query. Tryk Search for at søge.")

    # IMPORTANT: Streamlit widget state persists across reruns.
    # If the user changes to an image with different dimensions, previously-stored slider values
    # can become out-of-range and break the UI. Use a per-image key prefix to isolate state.
    try:
        key_prefix = f"crop_{_fname_key(Path(path).name)}_{w}x{h}"
    except Exception:
        key_prefix = f"crop_{w}x{h}"

    # Cropper component sometimes hides its own button after a click.
    # Use a nonce in the component key so we can re-mount it on demand.
    cropper_nonce_key = f"{key_prefix}_cropper_nonce"
    st.session_state.setdefault(cropper_nonce_key, 0)
    cropper_nonce = int(st.session_state.get(cropper_nonce_key, 0) or 0)

    if require_coords:
        st.info("This mode requires crop coordinates (sliders).")
        force_sliders = True
    else:
        # Default: hide manual sliders when cropper UI is available.
        # Users can still force sliders as a fallback / advanced option.
        force_sliders = st.checkbox(
            "Manual crop sliders (advanced)",
            value=False,
            key=f"{key_prefix}_force_sliders",
            help="Enable if the cropper UI is missing or you prefer coordinate sliders.",
        )

    cropper_error = None
    cropper_used = False
    if (not force_sliders) and _CROPPER_OK and _st_cropper is not None:
        try:
            # streamlit-cropperjs has inconsistent type stubs/signatures across versions.
            # Use an Any-typed wrapper to avoid Pylance false positives.
            _cropper_any: Any = _st_cropper

            # streamlit-cropperjs expects PNG/JPG bytes and returns cropped image bytes.
            import io

            bio = io.BytesIO()
            img.save(bio, format="PNG")
            pic_bytes = bio.getvalue()

            res = None
            try:
                res = _cropper_any(pic=pic_bytes, btn_text="Use crop", size=1.0, key=f"{key_prefix}_cropper_{cropper_nonce}")
            except TypeError:
                # some versions only accept positional args
                res = _cropper_any(pic_bytes, "Use crop", 1.0, f"{key_prefix}_cropper_{cropper_nonce}")

            if isinstance(res, (bytes, bytearray)) and len(res) > 0:
                cropped_bytes = bytes(res)
                st.session_state["_cropper_last_bytes"] = cropped_bytes
                st.session_state["_cropper_last_source_path"] = str(path)
                st.session_state["_cropper_last_key_prefix"] = str(key_prefix)
                cropper_used = True
                st.image(cropped_bytes, caption="Crop preview", output_format="PNG")
        except Exception as e:
            cropper_error = e

    # If a crop is already armed for this image, offer explicit controls to change/reset.
    armed_for_this_image = False
    try:
        cb = st.session_state.get("_cropper_last_bytes")
        sp = str(st.session_state.get("_cropper_last_source_path") or "")
        armed_for_this_image = bool(cb) and bool(sp) and (Path(sp).resolve() == Path(path).resolve())
    except Exception:
        cb = st.session_state.get("_cropper_last_bytes")
        sp = str(st.session_state.get("_cropper_last_source_path") or "")
        armed_for_this_image = bool(cb) and bool(sp) and (sp == str(path))

    if (not require_coords) and _CROPPER_OK and _st_cropper is not None and armed_for_this_image:
        cA, cB = st.columns(2)
        if cA.button("✏️ Change crop", key=f"{key_prefix}_change_crop"):
            # Clear the armed crop so Search can't accidentally use the old one.
            st.session_state["_cropper_last_bytes"] = b""
            st.session_state["_cropper_last_source_path"] = ""
            st.session_state["_cropper_last_key_prefix"] = ""
            st.session_state[cropper_nonce_key] = int(st.session_state.get(cropper_nonce_key, 0) or 0) + 1
            safe_rerun()
        if cB.button("🧹 Clear crop", key=f"{key_prefix}_clear_crop"):
            st.session_state["_cropper_last_bytes"] = b""
            st.session_state["_cropper_last_source_path"] = ""
            st.session_state["_cropper_last_key_prefix"] = ""
            st.session_state[cropper_nonce_key] = int(st.session_state.get(cropper_nonce_key, 0) or 0) + 1
            safe_rerun()

    if cropper_error is not None:
        with st.expander("Cropper UI error (debug)", expanded=False):
            st.write("Cropper component failed; falling back to sliders.")
            st.code(repr(cropper_error))

    if require_coords:
        st.info("Using sliders (coordinates required).")
    elif force_sliders:
        st.info("Using sliders (cropper UI disabled).")
    elif cropper_used:
        # Cropper bytes are armed; keep UI minimal here.
        # Caller (Step 6) shows the status line.
        return None
    elif _CROPPER_OK and _st_cropper is not None:
        st.info("Cropper UI ready. Click 'Use crop' to apply.")
        # No crop armed yet; return None so caller can prompt the user.
        return None
    else:
        st.info("Cropper UI not available; using sliders (install `streamlit-cropperjs`).")

    c1, c2 = st.columns(2)
    x0 = c1.slider("x0", 0, max(0, w - 1), 0, key=f"{key_prefix}_x0")
    x1 = c1.slider("x1", 1, w, w, key=f"{key_prefix}_x1")
    y0 = c2.slider("y0", 0, max(0, h - 1), 0, key=f"{key_prefix}_y0")
    y1 = c2.slider("y1", 1, h, h, key=f"{key_prefix}_y1")
    if x1 <= x0 or y1 <= y0:
        return None
    return (int(x0), int(y0), int(x1), int(y1))


def _ensure_crop_session_id(session_id: Optional[str] = None) -> str:
    sid = str(session_id or "").strip()
    if sid:
        return sid
    sid = str(st.session_state.get("_crop_session_id") or "").strip()
    if not sid:
        # Keep it filesystem-friendly and deterministic enough for a Streamlit session.
        sid = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(time.time_ns() % 1_000_000_000):09d}"
        st.session_state["_crop_session_id"] = sid
    return sid


def _save_query_crop_image(source_path: str | Path, cropped_bytes: bytes, session_id: str) -> str:
    """Save crop bytes as a PNG under repo `/_crops/<session_id>/...` and return the path."""
    from pathlib import Path
    import time

    _HERE_LOCAL = Path(__file__).resolve().parent
    sid = _ensure_crop_session_id(session_id)
    out_dir = _HERE_LOCAL / "_crops" / str(sid)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time() * 1000)
    out_path = out_dir / f"crop_{ts}.png"

    try:
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(cropped_bytes)).convert("RGB")
        img.save(out_path, format="PNG")
    except Exception:
        out_path.write_bytes(bytes(cropped_bytes))

    st.session_state["_last_saved_crop_path"] = str(out_path)
    st.session_state["_last_saved_crop_query"] = str(source_path)
    st.session_state["_last_saved_crop_source"] = "cropper_bytes"
    return str(out_path)


def _stars_1_to_5_from_scores(scores: pd.Series, higher_is_better: bool) -> pd.Series:
    """Map numeric scores to a 1–5 star string, relative within the provided series."""
    s = pd.to_numeric(scores, errors="coerce")
    if s.dropna().empty:
        return pd.Series(["☆☆☆☆☆"] * int(len(scores)), index=scores.index)

    vmin = float(np.nanmin(s.to_numpy(dtype="float32")))
    vmax = float(np.nanmax(s.to_numpy(dtype="float32")))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) <= 1e-12:
        stars = pd.Series([3] * int(len(scores)), index=scores.index)
    else:
        # Normalize to [0,1] within this result set.
        if higher_is_better:
            p = (s.astype(float) - vmin) / (vmax - vmin)
        else:
            p = (vmax - s.astype(float)) / (vmax - vmin)
        p = p.clip(0.0, 1.0)
        stars = (np.floor(p * 5.0).astype(int) + 1).clip(1, 5)

    return stars.map(lambda n: ("★" * int(n)) + ("☆" * int(5 - int(n))))


# ---------- Index (filename -> full_path + stable_id) ----------
def _stable_id(pp: Path, size: int, mtime: int) -> str:
    # New v2 stable_id strategy: prefer resolved path (lowercased) when possible.
    # Input prefixed with 'v2|' to distinguish from older strategy.
    try:
        pstr = str(pp.resolve()).lower()
    except Exception:
        pstr = str(pp).lower()
    base = f"v2|{pstr}|{int(size)}|{int(mtime)}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


def _compute_v2_stable_id_from_row(full_path: str, size: int, mtime: int) -> str:
    try:
        p = Path(full_path)
        try:
            pstr = str(p.resolve()).lower()
        except Exception:
            pstr = str(p).lower()
    except Exception:
        pstr = str(full_path).lower()
    base = f"v2|{pstr}|{int(size)}|{int(mtime)}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


def build_sid_map(idx_df: pd.DataFrame) -> Dict[str, str]:
    """Build a mapping filename_key -> stable_id from an index dataframe.

    Returns an empty dict if idx_df is None/empty or if stable_id isn't present.
    """
    if idx_df is None or idx_df.empty:
        return {}
    df = _std_cols(idx_df.copy())
    if "filename" not in df.columns:
        if "full_path" in df.columns:
            df["filename"] = df["full_path"].astype(str).map(_fname_key)
        else:
            return {}
    else:
        df["filename"] = df["filename"].astype(str).map(_fname_key)

    if "stable_id" not in df.columns:
        return {}

    return dict(zip(df["filename"].astype(str), df["stable_id"].astype(str)))


def build_files_index(
    roots_field: str,
    output_dir: str,
    index_name: str,
    progress_cb=None,
) -> Tuple[pd.DataFrame, int, Optional[str], Optional[pd.DataFrame]]:
    roots = parse_roots(roots_field)
    if not roots:
        return pd.DataFrame(), 0, None, pd.DataFrame()

    files = scan_images(roots)
    total = len(files)

    out_path = None
    rows = []
    header = ["full_path", "filename", "parent", "size_bytes", "mtime", "stable_id"]

    writable = bool(output_dir) and _is_writable_dir(output_dir)
    if writable:
        out_path = str(Path(output_dir) / index_name)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i, p in enumerate(files, 1):
                pp = Path(p)
                try:
                    stt = pp.stat()
                    sid = _stable_id(pp, stt.st_size, int(stt.st_mtime))
                    row = [str(pp), pp.name, pp.parent.name, int(stt.st_size), int(stt.st_mtime), sid]
                except Exception:
                    sid = hashlib.sha1(str(pp).encode("utf-8")).hexdigest()[:16]
                    row = [str(pp), pp.name, pp.parent.name, -1, 0, sid]
                w.writerow(row)
                if progress_cb and (i % 500 == 0 or i == total):
                    progress_cb(i, total)
        preview = pd.read_csv(out_path, nrows=20, low_memory=False)
        return preview, total, out_path, None

    # Hosted / no write permissions -> in-memory
    for i, p in enumerate(files, 1):
        pp = Path(p)
        try:
            stt = pp.stat()
            sid = _stable_id(pp, stt.st_size, int(stt.st_mtime))
            row = [str(pp), pp.name, pp.parent.name, int(stt.st_size), int(stt.st_mtime), sid]
        except Exception:
            sid = hashlib.sha1(str(pp).encode("utf-8")).hexdigest()[:16]
            row = [str(pp), pp.name, pp.parent.name, -1, 0, sid]
        rows.append(row)
        if progress_cb and (i % 500 == 0 or i == total):
            progress_cb(i, total)

    df = pd.DataFrame(rows, columns=header)
    return df.head(20), total, None, df


@st.cache_data(show_spinner=False)
def load_index_df(index_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(index_csv_path, low_memory=False)
    df = _std_cols(df)
    return df


def filemap_from_index(df_idx: pd.DataFrame) -> Dict[str, str]:
    df = _std_cols(df_idx)
    if "full_path" not in df.columns:
        raise ValueError("Index CSV missing 'full_path'.")
    df = _ensure_stable_id(df)
    if "filename" not in df.columns:
        df["filename"] = df["full_path"].astype(str).map(_fname_key)
    else:
        df["filename"] = df["filename"].astype(str).map(_fname_key)

    use_stable = "stable_id" in df.columns and df["stable_id"].astype(str).str.strip().ne("").any()
    if use_stable:
        st.session_state["_index_key_kind"] = "stable_id"
        try:
            st.session_state["_sid_to_filename"] = dict(zip(df["stable_id"].astype(str), df["filename"].astype(str)))
        except Exception:
            st.session_state["_sid_to_filename"] = {}
    else:
        st.session_state["_index_key_kind"] = "filename"
        st.session_state["_sid_to_filename"] = {}
        st.warning("⚠️ Index has no stable_id column; using filename basenames as keys (risk of collisions).")

    # Detect basename collisions only in filename-key mode.
    try:
        if not use_stable and len(df) > 0:
            vc = df["filename"].astype(str).value_counts()
            dups = vc[vc > 1]
            st.session_state["_index_has_duplicates"] = bool(len(dups) > 0)
            st.session_state["_index_duplicate_examples"] = list(dups.head(10).index.astype(str))
        else:
            st.session_state["_index_has_duplicates"] = False
            st.session_state["_index_duplicate_examples"] = []
    except Exception:
        st.session_state["_index_has_duplicates"] = False
        st.session_state["_index_duplicate_examples"] = []
        _dbg_exc("Index duplicate detection failed")

    # Use vectorized zip to build the mapping (faster than iterrows)
    keys = df["stable_id"].astype(str).tolist() if use_stable else df["filename"].astype(str).tolist()
    vals = df["full_path"].astype(str).tolist()
    return dict(zip(keys, vals))


def attach_paths_from_index(df: pd.DataFrame, idx_df: pd.DataFrame, index_key_kind: str) -> pd.DataFrame:
    """Attach a `full_path` column to a results dataframe using the Step 4 index.

    This prevents thumbnail rendering from losing the original file paths.
    """
    if df is None or df.empty:
        return df
    if idx_df is None or idx_df.empty:
        return df

    try:
        idx = _std_cols(idx_df.copy())
        if "full_path" not in idx.columns:
            return df
        idx = _ensure_stable_id(_ensure_filename(idx))
        idx["filename"] = idx["filename"].astype(str).map(_fname_key)
    except Exception:
        return df

    out = df.copy()
    try:
        out = _std_cols(out)
    except Exception:
        pass

    kk = str(index_key_kind)
    if kk == "stable_id" and "stable_id" in idx.columns:
        # Ensure df has stable_id for the join.
        if "stable_id" not in out.columns:
            if "filename" in out.columns:
                sid_map = build_sid_map(idx)
                out["stable_id"] = out["filename"].astype(str).map(lambda x: sid_map.get(_fname_key(str(x)), ""))
        # Attach full_path.
        keep = idx[["stable_id", "full_path"]].copy()
        keep["stable_id"] = keep["stable_id"].astype(str).str.strip()
        out["stable_id"] = out.get("stable_id", "").astype(str).str.strip()
        out = out.merge(keep, on="stable_id", how="left")
    else:
        # filename-keyed join.
        if "filename" not in out.columns:
            return out
        keep = idx[["filename", "full_path"]].copy()
        keep["filename"] = keep["filename"].astype(str).map(_fname_key)
        out["filename"] = out["filename"].astype(str).map(_fname_key)
        out = out.merge(keep, on="filename", how="left")

    return out


def try_load_existing_index(output_dir: str, index_name: str) -> Tuple[bool, Optional[str], Dict[str, str], pd.DataFrame]:
    p = Path(output_dir) / index_name
    if not p.exists():
        return False, None, {}, pd.DataFrame()
    try:
        df = load_index_df(str(p))
        fm = filemap_from_index(df)
        # Check whether the stable_id strategy appears to match the current v2 strategy.
        # If not, set a session_state flag so the UI can warn the user to rebuild the index.
        try:
            changed = False
            msg = None
            if "stable_id" not in df.columns:
                changed = True
                msg = "stable_id missing in index; rebuild index recommended"
            else:
                # sample up to 200 rows to detect mismatches without too much work
                sample = df.head(200) if len(df) > 200 else df
                for _, row in sample.iterrows():
                    full = row.get("full_path") if "full_path" in row else None
                    size = row.get("size_bytes") if "size_bytes" in row else None
                    mtime = row.get("mtime") if "mtime" in row else None
                    if full is None or size is None or mtime is None:
                        # cannot compute reliably for this row — skip
                        continue
                    try:
                        computed = _compute_v2_stable_id_from_row(full, int(size), int(mtime))
                    except Exception:
                        continue
                    existing = str(row.get("stable_id", ""))
                    if existing != computed:
                        changed = True
                        msg = "stable_id strategy changed; rebuild index recommended"
                        break
            st.session_state["stable_id_strategy_changed"] = bool(changed)
            if changed:
                st.session_state["stable_id_strategy_message"] = msg
            else:
                st.session_state.pop("stable_id_strategy_message", None)
        except Exception:
            # don't let detection errors prevent loading
            pass
        return True, str(p), fm, df
    except Exception:
        return False, None, {}, pd.DataFrame()


# ---------- Filters ----------
def build_allowed_set(meta_df: pd.DataFrame, key_kind: str = "filename") -> Optional[Set[str]]:
    include_rules: Dict[str, object] = st.session_state.get("filter_include", {}) or {}
    exclude_rules: Dict[str, object] = st.session_state.get("filter_exclude", {}) or {}
    query_text = (st.session_state.get("filter_query_text", "") or "").strip()

    if not include_rules and not exclude_rules and not query_text:
        return None

    df = meta_df.copy()
    df = _ensure_stable_id(df)
    df = _ensure_filename(df)
    _has_key(df, "Metadata DF")

    # include
    for col, vals in include_rules.items():
        if col not in df.columns:
            continue
        if isinstance(vals, (list, tuple, set)):
            df = df[df[col].astype(str).isin([str(v) for v in vals])]
        else:
            df = df[df[col].astype(str) == str(vals)]

    # exclude
    for col, vals in exclude_rules.items():
        if col not in df.columns:
            continue
        if isinstance(vals, (list, tuple, set)):
            df = df[~df[col].astype(str).isin([str(v) for v in vals])]
        else:
            df = df[df[col].astype(str) != str(vals)]

    # pandas query
    if query_text:
        try:
            df = df.query(query_text, engine="python")
        except Exception as e:
            st.warning(f"⚠️ Ignoring query (error): {e}")

    if str(key_kind) == "stable_id":
        if "stable_id" not in df.columns:
            st.warning("⚠️ Filters requested stable_id, but metadata has no stable_id column; falling back to filename.")
            allowed = {str(x) for x in df["filename"].astype(str).tolist() if str(x).strip()}
        else:
            allowed = {str(x).strip() for x in df["stable_id"].astype(str).tolist() if str(x).strip()}
    else:
        allowed = {str(x) for x in df["filename"].astype(str).tolist() if str(x).strip()}
    return allowed


# ---------- Overlay (annotations) ----------
def overlay_dir(output_dir: str, session: str) -> Path:
    base = Path(output_dir or ".") / "_overlay" / str(session)
    base.mkdir(parents=True, exist_ok=True)
    return base


def overlay_current_path(output_dir: str, session: str) -> Path:
    return overlay_dir(output_dir, session) / "overlay_current.csv"


def overlay_log_path(output_dir: str, session: str) -> Path:
    return overlay_dir(output_dir, session) / "overlay_log.csv"


def overlay_load_current(output_dir: str, session: str) -> pd.DataFrame:
    p = overlay_current_path(output_dir, session)
    if p.exists():
        try:
            df = pd.read_csv(p, low_memory=False)
            return _std_cols(df)
        except Exception:
            _dbg_exc(f"Overlay load failed: {p}")
    return pd.DataFrame(columns=["stable_id", "filename", "marker", "value", "timestamp"])


def overlay_values_map(output_dir: str, session: str, marker: str, key_kind: str = "filename") -> Dict[str, Set[str]]:
    cur = overlay_load_current(output_dir, session)
    if cur.empty:
        return {}
    need = {"filename", "marker", "value"}
    if not need.issubset(set(cur.columns)):
        return {}
    cur = cur[cur["marker"].astype(str) == str(marker)].copy()
    out: Dict[str, Set[str]] = {}
    if str(key_kind) == "stable_id" and "stable_id" in cur.columns:
        cur["stable_id"] = cur["stable_id"].astype(str).str.strip()
        for sid, grp in cur.groupby("stable_id")["value"]:
            if str(sid).strip():
                out[str(sid).strip()] = {str(v) for v in grp.tolist() if str(v).strip()}
    else:
        # Normalize filename keys so lookups match _fname_key usage elsewhere.
        cur["filename"] = cur["filename"].astype(str).map(_fname_key)
        for fn, grp in cur.groupby("filename")["value"]:
            out[str(fn)] = {str(v) for v in grp.tolist() if str(v).strip()}
    return out


def overlay_append_log(output_dir: str, session: str, action: str, row: dict) -> None:
    lp = overlay_log_path(output_dir, session)
    write_header = not lp.exists()
    lp.parent.mkdir(parents=True, exist_ok=True)
    with open(lp, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "action", "stable_id", "filename", "marker", "value"])
        if write_header:
            w.writeheader()
        w.writerow(
            {
                "timestamp": row.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                "action": action,
                "stable_id": row.get("stable_id", ""),
                "filename": row.get("filename", ""),
                "marker": row.get("marker", ""),
                "value": row.get("value", ""),
            }
        )


def overlay_add(output_dir: str, session: str, stable_id: str, filename: str, marker: str, value: str) -> None:
    cur = overlay_load_current(output_dir, session)
    cur = _std_cols(cur)
    row = {
        "stable_id": str(stable_id),
        "filename": _fname_key(filename),
        "marker": str(marker),
        "value": str(value),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if cur.empty:
        cur = pd.DataFrame([row])
        cur.to_csv(overlay_current_path(output_dir, session), index=False, encoding="utf-8")
        overlay_append_log(output_dir, session, "add", row)
        return

    mask = (
        (cur["stable_id"].astype(str) == row["stable_id"])
        & (cur["marker"].astype(str) == row["marker"])
        & (cur["value"].astype(str) == row["value"])
    )
    if not mask.any():
        cur = pd.concat([cur, pd.DataFrame([row])], ignore_index=True)
        cur.to_csv(overlay_current_path(output_dir, session), index=False, encoding="utf-8")
        overlay_append_log(output_dir, session, "add", row)


def overlay_remove(output_dir: str, session: str, stable_id: str, marker: str, value: str) -> None:
    cur = overlay_load_current(output_dir, session)
    if cur.empty:
        return
    cur = _std_cols(cur)
    if not {"stable_id", "marker", "value"}.issubset(set(cur.columns)):
        return

    mask = (
        (cur["stable_id"].astype(str) == str(stable_id))
        & (cur["marker"].astype(str) == str(marker))
        & (cur["value"].astype(str) == str(value))
    )
    if mask.any():
        row = cur.loc[mask].iloc[0].to_dict()
        cur = cur.loc[~mask].copy()
        cur.to_csv(overlay_current_path(output_dir, session), index=False, encoding="utf-8")
        overlay_append_log(output_dir, session, "remove", row)


# ---------- Overlay schema helpers ----------
def overlay_schema_path(output_dir: str, session: str) -> Path:
    return overlay_dir(output_dir, session) / "schema.json"


def _field_key_from_title(title: str) -> str:
    k = _safe_colname(title).lower()
    if not k:
        k = "field"
    if k[0].isdigit():
        k = f"f_{k}"
    return k


def overlay_load_schema(output_dir: str, session: str) -> dict:
    p = overlay_schema_path(output_dir, session)
    if not p.exists():
        return {"fields": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"fields": {}}
        if "fields" not in data or not isinstance(data.get("fields"), dict):
            data["fields"] = {}
        return data
    except Exception:
        return {"fields": {}}


def overlay_save_schema(output_dir: str, session: str, schema: dict) -> None:
    p = overlay_schema_path(output_dir, session)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")


def overlay_clear_field(output_dir: str, session: str, stable_id: str, marker: str, filename: Optional[str] = None) -> None:
    cur = overlay_load_current(output_dir, session)
    if cur.empty:
        return
    cur = _std_cols(cur)
    if filename is not None:
        mask = (
            (cur["marker"].astype(str) == str(marker)) & (cur["filename"].astype(str) == _fname_key(filename))
        )
    else:
        mask = (
            (cur["marker"].astype(str) == str(marker)) & (cur["stable_id"].astype(str) == str(stable_id))
        )
    if not mask.any():
        return
    rows = cur.loc[mask].to_dict("records")
    cur = cur.loc[~mask].copy()
    cur.to_csv(overlay_current_path(output_dir, session), index=False, encoding="utf-8")
    for r in rows:
        overlay_append_log(output_dir, session, "remove", r)



# ---------- Export overlay merged ----------
def _safe_colname(s: str) -> str:
    s = str(s)
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s[:80] or "val")


def export_overlay_merged(
    meta_path: str,
    index_csv_path: Optional[str],
    output_dir: str,
    session: str,
    marker: str,
    style: str,
    include_extras: bool,
    field_spec: Optional[dict] = None,
) -> pd.DataFrame:
    meta = load_metadata(meta_path)

    idx_df = pd.DataFrame()
    if index_csv_path and Path(index_csv_path).exists():
        idx_df = load_index_df(index_csv_path)

    cur = overlay_load_current(output_dir, session)
    cur = _std_cols(cur)
    if cur.empty or "marker" not in cur.columns:
        raise ValueError("No overlay_current.csv (or it is empty).")

    cur = cur[cur["marker"].astype(str) == str(marker)].copy()
    if cur.empty:
        raise ValueError(f"No overlay entries for marker='{marker}' in session='{session}'.")

    cur["filename"] = cur["filename"].astype(str).map(_fname_key)
    cur["stable_id"] = cur["stable_id"].astype(str)

    # Optional: stable_id -> filename via index (if index has stable_id + filename)
    if not idx_df.empty and {"stable_id", "filename"}.issubset(set(idx_df.columns)):
        tmp = idx_df[["stable_id", "filename"]].copy()
        tmp["stable_id"] = tmp["stable_id"].astype(str)
        tmp["filename"] = tmp["filename"].astype(str).map(_fname_key)
        sid_to_fn = dict(zip(tmp["stable_id"], tmp["filename"]))
        cur["filename"] = cur["stable_id"].map(sid_to_fn).fillna(cur["filename"])

    # helpers
    def _spec_type(spec: Optional[dict]) -> str:
        if not isinstance(spec, dict):
            return "category"
        t = str(spec.get("type") or "category").strip().lower()
        return t if t in {"category", "number"} else "category"

    def _spec_kind(spec: Optional[dict]) -> str:
        if not isinstance(spec, dict):
            return "multi"
        k = str(spec.get("kind") or "multi").strip().lower()
        return k if k in {"single", "multi"} else "multi"

    def _safe_unique_col(base: str) -> str:
        base = str(base)
        if base in meta.columns:
            return f"{base}_overlay"
        return base

    def uniq_sorted(vals):
        return sorted({str(v).strip() for v in vals if str(v).strip()})

    field_type = _spec_type(field_spec)
    field_kind = _spec_kind(field_spec)
    num_kind = None
    if isinstance(field_spec, dict):
        nk = str(field_spec.get("num_kind") or "").strip().lower()
        num_kind = nk if nk in {"int", "float"} else None

    style = str(style or "").strip()
    style_l = style.lower()
    is_smart = style_l.startswith("smart")
    is_wide = style_l.startswith("wide")
    is_binary = style_l.startswith("binary")

    # Export rules (goal: predictable columns for end-users):
    # - number fields -> one numeric column named marker
    # - single category fields -> one string column named marker
    # - multi category fields -> Wide/Smart => marker_list, Binary => one-hot columns
    if field_type == "number":
        # numeric column (one value per filename; use latest timestamp if present)
        dfv = cur[["filename", "value"]].copy()
        if "timestamp" in cur.columns:
            tmp = cur[["filename", "value", "timestamp"]].copy()
            tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce")
            tmp = tmp.sort_values("timestamp")
            dfv = tmp.groupby("filename", as_index=False).tail(1)[["filename", "value"]].copy()
        else:
            dfv = dfv.groupby("filename", as_index=False).tail(1)

        col = _safe_unique_col(str(marker))
        ser = dfv["value"].astype(str).str.strip()
        ser = ser.replace("", np.nan)
        ser = pd.to_numeric(ser, errors="coerce")
        if num_kind == "int":
            ser = ser.round().astype("Int64")
        dfv[col] = ser
        dfv = dfv.drop(columns=["value"], errors="ignore")

        merged = meta.merge(dfv, on="filename", how="left")
        overlay_keys = set(dfv["filename"].astype(str))
        overlay_df = dfv

    elif field_kind == "single":
        # single category column (one value per filename; use latest timestamp if present)
        dfv = cur[["filename", "value"]].copy()
        if "timestamp" in cur.columns:
            tmp = cur[["filename", "value", "timestamp"]].copy()
            tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce")
            tmp = tmp.sort_values("timestamp")
            dfv = tmp.groupby("filename", as_index=False).tail(1)[["filename", "value"]].copy()
        else:
            dfv = dfv.groupby("filename", as_index=False).tail(1)
        col = _safe_unique_col(str(marker))
        dfv[col] = dfv["value"].astype(str).str.strip().replace("", np.nan)
        dfv = dfv.drop(columns=["value"], errors="ignore")

        merged = meta.merge(dfv, on="filename", how="left")
        overlay_keys = set(dfv["filename"].astype(str))
        overlay_df = dfv

    else:
        # multi category export
        grp = cur.groupby("filename")["value"].agg(uniq_sorted)

        if (not is_binary) or is_wide or is_smart:
            col = f"{marker}_list"
            col = _safe_unique_col(col)
            wide = pd.DataFrame({"filename": grp.index, col: [";".join(vs) for vs in grp.values]})
            merged = meta.merge(wide, on="filename", how="left")
            overlay_keys = set(wide["filename"].astype(str))
            overlay_df = wide
        else:
            # binary columns
            all_vals = sorted({v for vs in grp.values for v in vs})
            bin_df = pd.DataFrame({"filename": grp.index})
            used = set(meta.columns) | {"filename"}
            for v in all_vals:
                c = f"{marker}_{_safe_colname(v)}"
                if c in used:
                    k = 2
                    while f"{c}_{k}" in used:
                        k += 1
                    c = f"{c}_{k}"
                used.add(c)
                bin_df[c] = [int(v in set(vs)) for vs in grp.values]
            merged = meta.merge(bin_df, on="filename", how="left")
            overlay_keys = set(bin_df["filename"].astype(str))
            overlay_df = bin_df

    if include_extras:
        meta_keys = set(meta["filename"].astype(str))
        extra_keys = sorted(list(overlay_keys - meta_keys))
        if extra_keys:
            extra_base = pd.DataFrame({c: [np.nan] * len(extra_keys) for c in meta.columns})
            extra_base["filename"] = extra_keys
            extra_rows = extra_base.merge(overlay_df, on="filename", how="left")
            merged = pd.concat([merged, extra_rows], ignore_index=True)

    return merged


def export_overlay_merged_all_fields(
    meta_path: str,
    index_csv_path: Optional[str],
    output_dir: str,
    session: str,
    style: str,
    include_extras: bool,
) -> pd.DataFrame:
    # Load schema to know which markers to include
    schema = overlay_load_schema(output_dir, session)
    fields = schema.get("fields", {}) if isinstance(schema, dict) else {}
    if not fields:
        # nothing defined -> behave like empty merged metadata
        return load_metadata(meta_path)

    meta = load_metadata(meta_path)
    idx_df = pd.DataFrame()
    if index_csv_path and Path(index_csv_path).exists():
        idx_df = load_index_df(index_csv_path)

    overlay_keys_all = set()
    merged = meta.copy()

    # For each field, compute a merged df and join progressively
    for key, spec in fields.items():
        try:
            m = export_overlay_merged(
                meta_path,
                index_csv_path,
                output_dir,
                session,
                key,
                style,
                include_extras,
                field_spec=spec,
            )
        except Exception:
            # if field has no entries, m may raise; skip
            continue
        # merge m into merged on filename; only add columns that are not already present
        if "filename" not in m.columns:
            # nothing sensible to merge
            continue
        # determine new columns to add from m (keep filename + any column not already in merged)
        new_cols = [c for c in m.columns if c == "filename" or c not in merged.columns]
        if len(new_cols) <= 1:
            # only filename present — nothing new
            continue
        merged = merged.merge(m[new_cols], on="filename", how="left")

    return merged


# ---------- Session state ----------
def _ensure_state():
    ss = st.session_state
    ss.setdefault("step", 1)

    # Defaults come from repo file `rewiz_default_paths.json` (or built-in example defaults).
    d = load_repo_defaults()
    ss.setdefault("images_root", str(d.get("images_root", "")))
    ss.setdefault("output_dir", str(d.get("output_dir", "")))
    ss.setdefault("meta_path", str(d.get("meta_path", "")))
    ss.setdefault("embed_path", str(d.get("embed_path", "")))
    ss.setdefault("index_name", str(d.get("index_name", "index.csv")))
    ss.setdefault("auto_load_index", bool(d.get("auto_load_index", True)))

    ss.setdefault("index_metric", str(d.get("index_metric", "Cosine (IP + normalization)")))
    ss.setdefault("k_neighbors", int(d.get("k_neighbors", 10)))

    ss.setdefault("meta_ok", False)
    ss.setdefault("embed_ok", False)
    ss.setdefault("images_ok", False)
    ss.setdefault("meta_head", pd.DataFrame())
    ss.setdefault("embed_head", pd.DataFrame())
    ss.setdefault("images_count", 0)

    ss.setdefault("index_csv_path", None)
    ss.setdefault("filemap", {})           # filename_key -> full_path
    ss.setdefault("index_df", pd.DataFrame())

    ss.setdefault("faiss_ready", False)
    ss.setdefault("_faiss_index", None)
    ss.setdefault("faiss_dim", None)
    ss.setdefault("embed_filenames", [])
    ss.setdefault("auto_load_faiss", True)
    ss.setdefault("auto_save_faiss", True)
    ss.setdefault("_faiss_cache_id", "")
    ss.setdefault("_faiss_saved_path", "")

    ss.setdefault("query_image_path", "")
    ss.setdefault("query_key", "")

    ss.setdefault("query_mode", str(d.get("query_mode", "Whole-image")))
    ss.setdefault("text_query", "")

    # Element encoder model selection (optional)
    ss.setdefault("siglip2_model", os.environ.get("REWIZ_SIGLIP2_MODEL", "siglip2_giant384"))

    ss.setdefault("patch_index_ready", False)
    ss.setdefault("_patch_faiss_index", None)
    ss.setdefault("patch_meta_path", "")
    ss.setdefault("patch_index_path", "")

    ss.setdefault("filter_include", {})
    ss.setdefault("filter_exclude", {})
    ss.setdefault("filter_query_text", "")

    ss.setdefault("result_meta_cols", [])
    ss.setdefault("last_results_df", pd.DataFrame())

    ss.setdefault("overlay_session", str(d.get("overlay_session", "iconography_2025q4")))
    ss.setdefault("overlay_marker", str(d.get("overlay_marker", "ikonografi")))
    # do NOT auto-load saved settings on startup by default; app should start with defaults
    ss.setdefault("auto_load_settings", False)
    ss.setdefault("_did_autoload_settings", False)


_ensure_state()


# Auto-load saved settings on startup if enabled (run once per session).
# This runs after defaults are set but before any sidebar widgets are instantiated.
if st.session_state.get("auto_load_settings", False) and not st.session_state.get("_did_autoload_settings", False):
    st.session_state["_did_autoload_settings"] = True
    # Only attempt to load if a checkpoint file exists (load_checkpoint searches a few candidates)
    if CHECKPOINT.exists() or (Path.cwd() / ".retriever_wizard" / "settings.json").exists() or (Path.home() / ".retriever_wizard" / "settings.json").exists():
        ok = load_checkpoint()
        # If load_checkpoint stored values, reload to apply them before widgets are created
        if ok:
            safe_rerun()


# If a checkpoint was loaded in the previous run, apply those values now
# before any widgets are instantiated to avoid Streamlit session_state errors.
if "_loaded_checkpoint_values" in st.session_state:
    try:
        _vals = st.session_state.pop("_loaded_checkpoint_values") or {}
        for _k, _v in _vals.items():
            # only apply allowed keys
            if _k in ("meta_path", "embed_path", "images_root", "output_dir"):
                st.session_state[_k] = _v
        st.session_state["_checkpoint_load_message"] = (
            st.session_state.get("_checkpoint_load_message")
            or f"Applied saved settings from {st.session_state.get('_checkpoint_load_path', '')}"
        )
    except Exception:
        pass


def reset_to_defaults() -> None:
    """Clear session state and re-apply defaults from _ensure_state then rerun."""
    # remove all keys from session state
    for k in list(st.session_state.keys()):
        try:
            del st.session_state[k]
        except Exception:
            pass
    # reinitialize defaults and refresh UI
    _ensure_state()
    safe_rerun()


# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    _sidebar_env_status()

    with st.expander("Element search model", expanded=False):
        st.caption("Used by Text and Crop (light). You can use the alias `siglip2_giant384`.")
        st.text_input("REWIZ_SIGLIP2_MODEL", key="siglip2_model")
        # Apply into environment so element_encoder picks it up.
        try:
            os.environ["REWIZ_SIGLIP2_MODEL"] = str(st.session_state.get("siglip2_model", "")).strip()
        except Exception:
            pass

        cA, cB = st.columns(2)
        if cA.button("Use SigLIP2 Giant 384", key="siglip2_set_giant384"):
            st.session_state["siglip2_model"] = "siglip2_giant384"
            try:
                os.environ["REWIZ_SIGLIP2_MODEL"] = "siglip2_giant384"
            except Exception:
                pass
            if _ELEMENT_OK and _clear_element_cache is not None:
                _clear_element_cache()
            safe_rerun()

        if cB.button("Clear model cache", key="siglip2_clear_cache"):
            if _ELEMENT_OK and _clear_element_cache is not None:
                _clear_element_cache()
                st.info("Cleared in-process model cache. Next embed will reload the model.")
            else:
                st.info("Element encoder not available.")
    st.text_input("Metadata CSV", key="meta_path")
    st.text_input("Embeddings CSV", key="embed_path")
    st.text_area("Image root folder(s)", key="images_root", height=80, help="Multiple roots: ; , | or newline")
    st.text_input("Output folder (for index + overlay)", key="output_dir")
    st.text_input("Index filename", key="index_name")
    st.checkbox("Auto-load existing index if present", key="auto_load_index")

    st.markdown("---")
    st.subheader("📐 FAISS metric")
    st.session_state["index_metric"] = st.radio(
        "Metric",
        ["Cosine (IP + normalization)", "L2 (squared)"],
        index=0 if "Cosine" in st.session_state["index_metric"] else 1,
    )

    with st.expander("FAISS / embeddings debug", expanded=False):
        st.caption(f"FAISS ready: {bool(st.session_state.get('faiss_ready'))}")
        if st.session_state.get("faiss_ready"):
            st.caption(f"Key kind: {st.session_state.get('_faiss_key_kind', 'unknown')}")
            st.caption(f"Dim: {st.session_state.get('faiss_dim')}")
            cols = st.session_state.get("_faiss_vec_cols") or []
            if cols:
                show = [str(c) for c in cols[:20]]
                st.caption(f"vec_cols (first {len(show)}):")
                st.code("\n".join(show))
            else:
                st.caption("vec_cols preview not available (build FAISS in Step 4).")
        else:
            st.caption("Build FAISS in Step 4 to see detected vec_cols/dim.")

    st.checkbox("Auto-load saved FAISS if present", key="auto_load_faiss")
    st.checkbox("Auto-save FAISS after building", key="auto_save_faiss")

    st.markdown("---")
    c1, c2 = st.columns(2)
    st.checkbox("Auto-load saved settings on startup", key="auto_load_settings")
    st.write(f"Settings file: `{str(CHECKPOINT)}`")
    if c1.button("💾 Save checkpoint", key="save_checkpoint"):
        save_checkpoint()
        st.success("Saved.")
    if c2.button("📥 Load checkpoint", key="load_checkpoint"):
        ok = load_checkpoint()
        msg = st.session_state.get("_checkpoint_load_message")
        if ok:
            st.success(msg or f"Loaded from {st.session_state.get('_checkpoint_load_path')}")
            # Refresh UI so new values are visible immediately
            safe_rerun()
        else:
            st.error(msg or "No checkpoint found.")

    # Reset button: restore default values (does not delete saved settings file)
    if st.button("↺ Reset to defaults", key="reset_defaults"):
        reset_to_defaults()

    st.markdown("---")
    if st.button("🧹 Clear cached data (reload CSVs)", key="clear_cache"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        safe_rerun()


# ---------- Auto-load index ----------
if st.session_state.get("auto_load_index", True):
    if not st.session_state.get("filemap"):
        ok, idx_path, fm, df_idx = try_load_existing_index(st.session_state["output_dir"], st.session_state["index_name"])
        if ok:
            st.session_state["index_csv_path"] = idx_path
            st.session_state["filemap"] = fm
            st.session_state["index_df"] = df_idx
            st.info(f"📌 Loaded existing index: {idx_path} ({len(fm):,} entries)")
            if st.session_state.get("_index_has_duplicates"):
                ex = st.session_state.get("_index_duplicate_examples", []) or []
                hint = f" Examples: {', '.join(ex[:5])}" if ex else ""
                st.warning("⚠️ Duplicate basenames detected in index; results may map to the wrong file." + hint)
            if st.session_state.get("stable_id_strategy_changed"):
                st.warning(st.session_state.get("stable_id_strategy_message", "stable_id strategy changed; rebuild index recommended"))


# ---------- FAISS persistence helpers ----------
def _faiss_cache_id(embed_path: str, metric: str) -> str:
    try:
        p = Path(str(embed_path)).expanduser()
        try:
            p = p.resolve()
        except Exception:
            pass
        payload = f"{str(p)}|{str(metric)}"
    except Exception:
        payload = f"{str(embed_path)}|{str(metric)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _faiss_cache_dir(output_dir: str) -> Path:
    base = Path(output_dir or ".")
    return base / "faiss"


def _faiss_cache_paths(output_dir: str, embed_path: str, metric: str) -> Dict[str, Path]:
    cid = _faiss_cache_id(embed_path, metric)
    d = _faiss_cache_dir(output_dir)
    return {
        "dir": d,
        "index": d / f"faiss_{cid}.faiss",
        "keys": d / f"faiss_{cid}.keys.txt",
        "manifest": d / f"faiss_{cid}.manifest.json",
    }


def _save_faiss_cache(
    output_dir: str,
    embed_path: str,
    metric: str,
    index: Any,
    keys: List[str],
    dim: int,
    vec_cols: List[str],
    key_kind: str,
) -> Tuple[bool, str]:
    if not _FAISS_OK:
        return False, "FAISS not installed"
    if not output_dir:
        return False, "No output_dir set"
    if not _is_writable_dir(output_dir):
        return False, f"Output dir not writable: {output_dir}"

    try:
        assert faiss is not None
        paths = _faiss_cache_paths(output_dir, embed_path, metric)
        paths["dir"].mkdir(parents=True, exist_ok=True)

        # Write FAISS index
        faiss.write_index(index, str(paths["index"]))

        # Write keys (one per line)
        with open(paths["keys"], "w", encoding="utf-8", newline="\n") as f:
            for k in keys:
                s = str(k).replace("\n", " ").strip()
                f.write(s + "\n")

        # Manifest for mismatch warnings
        emb_p = Path(str(embed_path)).expanduser()
        try:
            emb_p_res = emb_p.resolve()
        except Exception:
            emb_p_res = emb_p
        try:
            stt = emb_p_res.stat()
            emb_size = int(stt.st_size)
            emb_mtime = int(stt.st_mtime)
        except Exception:
            emb_size = None
            emb_mtime = None

        manifest = {
            "schema_version": 1,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "embed_path": str(emb_p_res),
            "embed_size": emb_size,
            "embed_mtime": emb_mtime,
            "metric": str(metric),
            "key_kind": str(key_kind),
            "dim": int(dim),
            "count": int(len(keys)),
            "vec_cols": list(vec_cols),
            "faiss_index_path": str(paths["index"].resolve()),
            "keys_path": str(paths["keys"].resolve()),
        }
        paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        st.session_state["_faiss_cache_id"] = _faiss_cache_id(embed_path, metric)
        st.session_state["_faiss_saved_path"] = str(paths["index"])  # user-friendly
        return True, f"Saved FAISS cache ({len(keys):,} vectors)"
    except Exception as e:
        return False, f"Failed saving FAISS cache: {e}"


def _load_faiss_cache_to_state(output_dir: str, embed_path: str, metric: str) -> Tuple[bool, str]:
    if not _FAISS_OK:
        return False, "FAISS not installed"
    if not output_dir:
        return False, "No output_dir set"

    paths = _faiss_cache_paths(output_dir, embed_path, metric)
    if not paths["index"].exists() or not paths["keys"].exists():
        return False, "No saved FAISS cache for current embeddings/metric"

    try:
        assert faiss is not None
        idx = faiss.read_index(str(paths["index"]))
        with open(paths["keys"], "r", encoding="utf-8") as f:
            keys = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

        # Manifest is optional but preferred
        key_kind = "filename"
        vec_cols: List[str] = []
        metric_built = str(metric)
        try:
            if paths["manifest"].exists():
                m = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                if isinstance(m, dict):
                    key_kind = str(m.get("key_kind", key_kind))
                    metric_built = str(m.get("metric", metric_built))
                    vc = m.get("vec_cols", [])
                    if isinstance(vc, list):
                        vec_cols = [str(x) for x in vc]
        except Exception:
            pass

        # Build pos_map (first occurrence wins)
        pos_map: Dict[str, int] = {}
        for i, k in enumerate(keys):
            if k not in pos_map:
                pos_map[k] = i

        dim = _faiss_dim(idx)

        st.session_state["_faiss_index"] = idx
        st.session_state["embed_filenames"] = keys
        st.session_state["_faiss_pos_map"] = pos_map
        st.session_state["faiss_dim"] = int(dim) if dim is not None else None
        st.session_state["_faiss_vec_cols"] = list(vec_cols)[:200]
        st.session_state["_faiss_key_kind"] = str(key_kind)
        st.session_state["_faiss_metric_built"] = str(metric_built)
        st.session_state["faiss_ready"] = True
        st.session_state["_faiss_cache_id"] = _faiss_cache_id(embed_path, metric)
        st.session_state["_faiss_saved_path"] = str(paths["index"])
        return True, f"Loaded saved FAISS cache: {paths['index'].name}"
    except Exception as e:
        return False, f"Failed loading FAISS cache: {e}"


def _clear_faiss_from_state() -> None:
    st.session_state["faiss_ready"] = False
    st.session_state["_faiss_index"] = None
    st.session_state["_faiss_pos_map"] = {}
    st.session_state["faiss_dim"] = None
    st.session_state["embed_filenames"] = []
    st.session_state["_faiss_vec_cols"] = []
    st.session_state["_faiss_key_kind"] = ""
    st.session_state["_faiss_metric_built"] = ""


# ---------- Auto-load FAISS cache ----------
if (
    st.session_state.get("auto_load_faiss", True)
    and not st.session_state.get("faiss_ready")
    and int(st.session_state.get("step", 1)) >= 4
):
    try:
        ok, msg = _load_faiss_cache_to_state(
            st.session_state.get("output_dir", ""),
            st.session_state.get("embed_path", ""),
            st.session_state.get("index_metric", "Cosine (IP + normalization)"),
        )
        if ok:
            st.info(f"📌 {msg}")
    except Exception:
        pass


# ---------- Step helpers (reduce redundant checks) ----------
def _validate_metadata_to_state(meta_path: str) -> None:
    try:
        df = load_metadata(meta_path)
        st.session_state["meta_ok"] = True
        st.session_state["meta_head"] = df.head(20)
    except Exception:
        st.session_state["meta_ok"] = False
        st.session_state["meta_head"] = pd.DataFrame()
        raise


def _validate_embeddings_to_state(embed_path: str) -> int:
    """Validate embeddings quickly and store a small preview in session_state.

    Returns detected embedding dim.
    """
    try:
        head, dim = validate_embeddings_sample(embed_path)
        st.session_state["embed_ok"] = True
        st.session_state["embed_head"] = head
        return int(dim)
    except Exception:
        st.session_state["embed_ok"] = False
        st.session_state["embed_head"] = pd.DataFrame()
        raise


def _scan_images_to_state(roots_field: str) -> Tuple[List[str], List[str]]:
    """Parse image roots, scan files, and update session_state.

    Returns (roots, files).
    """
    roots = parse_roots(roots_field)
    if not roots:
        st.session_state["images_ok"] = False
        st.session_state["images_count"] = 0
        return [], []
    files = scan_images(roots)
    st.session_state["images_ok"] = len(files) > 0
    st.session_state["images_count"] = int(len(files))
    return roots, files


def _run_consistency_check(meta_path: str, embed_path: str, roots_field: str) -> Dict[str, Any]:
    """Run consistency check and return a structured result.

    Checks:
    - metadata vs embeddings vs disk
    - (if available) file index vs disk
    - (if available) FAISS keys vs embeddings / file index
    """

    meta = load_metadata(meta_path)
    emb_names = load_embeddings_filenames_only(embed_path)
    # Disk scan can be very slow on large corpora. If we already have a Step 4 file index loaded,
    # use that as the "disk" view (good enough for join-consistency).
    idx_df = st.session_state.get("index_df", pd.DataFrame())
    disk_set: Set[str] = set()
    idx_filename_set: Set[str] = set()
    idx_stable_id_set: Set[str] = set()
    try:
        if isinstance(idx_df, pd.DataFrame) and not idx_df.empty:
            idx_df = _std_cols(idx_df)
            if "filename" in idx_df.columns:
                idx_filename_set = {str(x).strip().lower() for x in idx_df["filename"].astype(str).tolist() if str(x).strip()}
                disk_set = set(idx_filename_set)
            elif "full_path" in idx_df.columns:
                disk_set = {Path(str(x)).name.lower() for x in idx_df["full_path"].astype(str).tolist() if str(x).strip()}

            if "stable_id" in idx_df.columns:
                idx_stable_id_set = {str(x).strip() for x in idx_df["stable_id"].astype(str).tolist() if str(x).strip()}
    except Exception:
        disk_set = set()
        idx_filename_set = set()
        idx_stable_id_set = set()

    if not disk_set:
        roots = parse_roots(roots_field)
        files = scan_images(roots)
        disk_set = {Path(p).name.lower() for p in files}

    meta_set = set(meta["filename"].astype(str))
    emb_set = set(emb_names.astype(str))

    # FAISS (in-memory or saved cache)
    faiss_saved_exists = False
    faiss_keys_count: Optional[int] = None
    faiss_key_kind = str(st.session_state.get("_faiss_key_kind", ""))
    faiss_keys_set: Set[str] = set()
    faiss_sample_keys: List[str] = []

    try:
        out_dir = str(st.session_state.get("output_dir", "") or "")
        metric = str(st.session_state.get("index_metric", "Cosine (IP + normalization)"))
        paths = _faiss_cache_paths(out_dir, embed_path, metric)
        faiss_saved_exists = bool(paths["index"].exists() and paths["keys"].exists())
    except Exception:
        faiss_saved_exists = False

    if bool(st.session_state.get("faiss_ready")):
        try:
            faiss_key_kind = str(st.session_state.get("_faiss_key_kind", faiss_key_kind or "filename"))
            keys = list(st.session_state.get("embed_filenames") or [])
            faiss_keys_count = int(len(keys))
            # Keep sets as strings; stable_id keys are case-sensitive-ish but are already normalized.
            if faiss_key_kind == "stable_id":
                faiss_keys_set = {str(k).strip() for k in keys if str(k).strip()}
            else:
                faiss_keys_set = {str(k).strip().lower() for k in keys if str(k).strip()}
            faiss_sample_keys = [str(k) for k in keys[:10]]
        except Exception:
            faiss_keys_count = None
            faiss_keys_set = set()
            faiss_sample_keys = []
    elif faiss_saved_exists:
        # Avoid loading the FAISS index itself; read keys file for a lightweight check.
        try:
            out_dir = str(st.session_state.get("output_dir", "") or "")
            metric = str(st.session_state.get("index_metric", "Cosine (IP + normalization)"))
            paths = _faiss_cache_paths(out_dir, embed_path, metric)
            kf = paths.get("keys")
            if kf and Path(kf).exists():
                cnt = 0
                with open(kf, "r", encoding="utf-8") as f:
                    for ln in f:
                        s = str(ln).strip()
                        if not s:
                            continue
                        cnt += 1
                        if len(faiss_sample_keys) < 10:
                            faiss_sample_keys.append(s)
                faiss_keys_count = int(cnt)
        except Exception:
            faiss_keys_count = None

    # Compare FAISS keys when we have them in memory.
    miss_faiss_in_emb: List[str] = []
    miss_faiss_in_index: List[str] = []
    if faiss_keys_set:
        if faiss_key_kind == "stable_id":
            if idx_stable_id_set:
                miss_faiss_in_index = sorted(list(faiss_keys_set - idx_stable_id_set))[:10]
        else:
            miss_faiss_in_emb = sorted(list(faiss_keys_set - emb_set))[:10]
            if idx_filename_set:
                miss_faiss_in_index = sorted(list(faiss_keys_set - idx_filename_set))[:10]

    return {
        "counts": {
            "meta": int(len(meta_set)),
            "emb": int(len(emb_set)),
            "disk": int(len(disk_set)),
            "file_index": int(len(idx_filename_set)) if idx_filename_set else 0,
            "file_index_stable_id": int(len(idx_stable_id_set)) if idx_stable_id_set else 0,
            "faiss_keys": int(faiss_keys_count or 0) if (faiss_keys_count is not None) else 0,
        },
        "miss_meta_on_disk": sorted(list(meta_set - disk_set))[:10],
        "miss_emb_on_disk": sorted(list(emb_set - disk_set))[:10],
        "miss_meta_in_emb": sorted(list(meta_set - emb_set))[:10],
        "miss_emb_in_meta": sorted(list(emb_set - meta_set))[:10],
        "miss_index_on_disk": sorted(list(idx_filename_set - disk_set))[:10] if idx_filename_set else [],
        "miss_disk_in_index": sorted(list(disk_set - idx_filename_set))[:10] if idx_filename_set else [],
        "faiss": {
            "in_memory": bool(st.session_state.get("faiss_ready")),
            "saved_exists": bool(faiss_saved_exists),
            "key_kind": str(faiss_key_kind or ""),
            "sample_keys": list(faiss_sample_keys),
        },
        "miss_faiss_in_emb": miss_faiss_in_emb,
        "miss_faiss_in_index": miss_faiss_in_index,
    }


def _apply_loaded_index(idx_path: Optional[str], fm: Dict[str, str], df_idx: pd.DataFrame, *, announce: Optional[str] = None) -> None:
    st.session_state["index_csv_path"] = idx_path
    st.session_state["filemap"] = fm
    st.session_state["index_df"] = df_idx
    if announce:
        st.info(announce)

    if st.session_state.get("_index_has_duplicates"):
        ex = st.session_state.get("_index_duplicate_examples", []) or []
        hint = f" Examples: {', '.join(ex[:5])}" if ex else ""
        st.warning("⚠️ Duplicate basenames detected in index; results may map to the wrong file." + hint)
    if st.session_state.get("stable_id_strategy_changed"):
        st.warning(st.session_state.get("stable_id_strategy_message", "stable_id strategy changed; rebuild index recommended"))


def _load_or_build_index_to_state(
    output_dir: str,
    index_name: str,
    roots_field: str,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Tuple[bool, str]:
    """Try loading an existing Step 4 index; if missing, build it.

    Returns (ok, message).
    """

    ok, idx_path, fm, df_idx = try_load_existing_index(output_dir, index_name)
    if ok:
        _apply_loaded_index(idx_path, fm, df_idx)
        return True, f"Loaded existing index: {idx_path}"

    preview, n, out_path, df_full = build_files_index(roots_field, output_dir, index_name, progress_cb=progress_cb)
    # Ensure session_state gets a full index df + filemap
    if out_path:
        df_idx = load_index_df(out_path)
        fm = filemap_from_index(df_idx)
        _apply_loaded_index(out_path, fm, df_idx)
        return True, f"Wrote index: {out_path} ({n:,} rows)"

    if df_full is None:
        raise RuntimeError("Index build returned no dataframe.")
    fm = filemap_from_index(df_full)
    _apply_loaded_index(None, fm, df_full)
    return True, f"Built index in-memory ({n:,} rows)"


# ---------- Step UI ----------
st.markdown("---")
st.subheader(f"Step {st.session_state['step']} of 9")


# ========== Step 1: Metadata ==========
if st.session_state["step"] == 1:
    st.markdown("**Step 1: Validate metadata (CSV)**")
    st.info(
        "Load your metadata table (a spreadsheet about your images). "
        "The important link is a filename column so the tool can connect rows to image files. "
        "If your file uses a different column name (like path or full_path), the tool will try to derive filenames automatically."
    )

    # Plain-language intro (Step 1 only)
    with st.expander("What is this tool?", expanded=False):
        st.write(
            "This is a guided workflow for exploring an image collection with the help of embeddings (a numeric representation of images). "
            "You can search for visually similar images, review results, add human annotations (an overlay that never overwrites your source data), "
            "and optionally make a 2D map (projection) to explore patterns."
        )
        st.markdown(
            "**Glossary**\n"
            "- **Metadata**: A spreadsheet about the images (titles, dates, authors, etc.).\n"
            "- **Embeddings**: Numbers that summarize visual features; used for similarity search.\n"
            "- **File index**: A fast lookup table connecting filenames to full file paths (and stable IDs).\n"
            "- **FAISS index**: A vector index built from embeddings; used to find nearest neighbors quickly.\n"
            "- **Nearest neighbors**: The most visually similar images to your chosen query image.\n"
            "- **Overlay / annotation**: Your added labels saved separately from the original metadata.\n"
            "- **Projection (UMAP/t-SNE)**: A 2D map where nearby points look more similar (approx.)."
        )

    if st.button("🔍 Validate metadata", key="validate_metadata"):
        with st.status("Validating metadata…", expanded=True) as s:
            try:
                _validate_metadata_to_state(st.session_state["meta_path"])
                s.update(label="✅ Metadata OK", state="complete")
            except Exception as e:
                st.error(str(e))
                s.update(label="❌ Metadata NOT OK", state="error")

    st_dataframe_stretch(st.session_state["meta_head"])
    st.button("➡️ Next", disabled=not st.session_state["meta_ok"], on_click=lambda: st.session_state.update(step=2), key="next_step_1")

    if st.button("⚡ Auto-validate Steps 1–5 and go to Step 6", key="auto_validate_1_5"):
        with st.status("Auto-validating steps 1–5…", expanded=True) as s:
            # Step 1: metadata
            try:
                _validate_metadata_to_state(st.session_state["meta_path"])
                s.update(label="✅ Metadata OK", state="running")
            except Exception as e:
                s.update(label=f"❌ Metadata failed: {e}", state="error")
                st.error(f"Metadata validation failed: {e}")

            # Step 2: embeddings (fast sample)
            try:
                dim = _validate_embeddings_to_state(st.session_state["embed_path"])
                s.update(label=f"✅ Embeddings OK (dim={dim})", state="running")
            except Exception as e:
                s.update(label=f"❌ Embeddings failed: {e}", state="error")
                st.error(f"Embeddings validation failed: {e}")

            # Step 3: images
            try:
                roots, files = _scan_images_to_state(st.session_state["images_root"])
                if roots and files:
                    s.update(label=f"✅ Images OK ({len(files):,} files)", state="running")
                else:
                    s.update(label="❌ Images not found (no valid roots or no files)", state="error")
                    st.error("No valid image roots configured (or no images found).")
            except Exception as e:
                st.session_state["images_ok"] = False
                st.session_state["images_count"] = 0
                s.update(label=f"❌ Images failed: {e}", state="error")
                st.error(f"Image scan failed: {e}")

            # Step 4: load or build file index
            try:
                bar = st.empty()

                def progress_cb(done: int, total: int) -> None:
                    pct = int(100 * done / max(1, total))
                    bar.progress(pct, text=f"Indexing files: {done}/{total}")

                with st.status("Building/loading index…", expanded=True) as s2:
                    ok, msg = _load_or_build_index_to_state(
                        st.session_state["output_dir"],
                        st.session_state["index_name"],
                        st.session_state["images_root"],
                        progress_cb=progress_cb,
                    )
                    if ok:
                        s2.update(label=f"✅ {msg}", state="complete")
            except Exception as e:
                s.update(label=f"❌ Index step failed: {e}", state="error")
                st.error(f"Index build/load failed: {e}")

            # Step 5: consistency check (now includes index + FAISS presence)
            try:
                res = _run_consistency_check(
                    st.session_state["meta_path"],
                    st.session_state["embed_path"],
                    st.session_state["images_root"],
                )
                st.write("**Auto-check mismatches (top 10)**")
                st.write("In metadata but missing on disk:", res["miss_meta_on_disk"] or "—")
                st.write("In embeddings but missing on disk:", res["miss_emb_on_disk"] or "—")
                st.write("In metadata but not in embeddings:", res["miss_meta_in_emb"] or "—")
                st.write("In embeddings but not in metadata:", res["miss_emb_in_meta"] or "—")
                if res.get("miss_index_on_disk"):
                    st.write("In file index but missing on disk:", res["miss_index_on_disk"] or "—")
                s.update(label="✅ Consistency checks done", state="running")
            except Exception as e:
                s.update(label=f"❌ Consistency checks failed: {e}", state="error")
                st.error(f"Consistency check failed: {e}")

            st.session_state.update(step=6)
            st.success("Auto-validate complete — moving to Step 6")


# ========== Step 2: Embeddings ==========
elif st.session_state["step"] == 2:
    st.markdown("**Step 2: Validate embeddings (CSV)**")
    st.info(
        "Load the embeddings table. Embeddings are numbers that describe visual features of each image. "
        "They enable similarity search (finding images that look alike)."
    )

    st.caption("This validation reads a small sample for speed. Full vectors are loaded only when building FAISS or running projections.")

    if st.button("🔍 Validate embeddings", key="validate_embeddings"):
        with st.status("Validating embeddings…", expanded=True) as s:
            try:
                dim = _validate_embeddings_to_state(st.session_state["embed_path"])
                st.write(f"Vector dim: **{dim}**")
                s.update(label="✅ Embeddings OK", state="complete")
            except Exception as e:
                st.error(str(e))
                s.update(label="❌ Embeddings NOT OK", state="error")

    st_dataframe_stretch(st.session_state["embed_head"])
    cL, cR = st.columns(2)
    cL.button("⬅️ Back", on_click=lambda: st.session_state.update(step=1), key="back_step_2")
    cR.button("➡️ Next", disabled=not st.session_state["embed_ok"], on_click=lambda: st.session_state.update(step=3), key="next_step_2")


# ========== Step 3: Images ==========
elif st.session_state["step"] == 3:
    st.markdown("**Step 3: Validate image folder(s)**")
    st.info(
        "Select one or more folders that contain the image files. "
        "The tool scans these folders (including subfolders) to find images and build an index for fast lookup."
    )

    # Warn about missing roots (parse_roots intentionally drops non-existent paths)
    raw_roots = []
    try:
        tmp = st.session_state.get("images_root", "") or ""
        for sep in [";", "|", ","]:
            tmp = tmp.replace(sep, "\n")
        raw_roots = [p.strip().strip('"').strip("'") for p in tmp.splitlines() if p.strip()]
    except Exception:
        raw_roots = []
    if raw_roots:
        missing = [p for p in raw_roots if p and not Path(p).expanduser().exists()]
        if missing:
            st.warning("Some image roots do not exist and will be ignored:\n" + "\n".join([f"- {m}" for m in missing]))

    if st.button("🔍 Check images", key="check_images"):
        with st.status("Scanning image folder(s)…", expanded=True) as s:
            try:
                roots, files = _scan_images_to_state(st.session_state["images_root"])
                if not roots:
                    st.error("No valid roots. Check the paths.")
                    s.update(label="❌ Images NOT OK", state="error")
                else:
                    st.write(f"Found **{len(files):,}** images across **{len(roots)}** root(s).")
                    s.update(label="✅ Images OK" if files else "❌ Images NOT OK", state="complete" if files else "error")
            except Exception as e:
                st.session_state["images_ok"] = False
                st.session_state["images_count"] = 0
                st.error(f"Image scan failed: {e}")
                s.update(label="❌ Images NOT OK", state="error")

    st.metric("Images found", f"{st.session_state['images_count']:,}")
    cL, cR = st.columns(2)
    cL.button("⬅️ Back", on_click=lambda: st.session_state.update(step=2), key="back_step_3")
    cR.button("➡️ Next", disabled=not st.session_state["images_ok"], on_click=lambda: st.session_state.update(step=4), key="next_step_3")


# ========== Step 4: Indexing ==========
elif st.session_state["step"] == 4:
    st.markdown("**Step 4: Indexing (file paths + FAISS)**")
    st.info(
        "This step prepares two different indexes. They solve different problems, so you typically need both."
    )

    # Flash messages for Step 4 actions (so users see immediate feedback after a rerun).
    try:
        _flash = str(st.session_state.pop("_step4_flash", "") or "").strip()
        if _flash:
            st.info(_flash)
    except Exception:
        pass

    # Quick visual status indicators (so users know if they can press Next).
    file_index_ready = False
    try:
        fm_now = st.session_state.get("filemap", {}) or {}
        idx_df_now = st.session_state.get("index_df", pd.DataFrame())
        file_index_ready = bool(fm_now) or (isinstance(idx_df_now, pd.DataFrame) and not idx_df_now.empty)
    except Exception:
        file_index_ready = False

    faiss_ready_now = bool(st.session_state.get("faiss_ready"))
    sA, sB, sC = st.columns([1, 1, 2])
    sA.markdown(f"**File index**: {'🟢 loaded' if file_index_ready else '🔴 missing'}")
    sB.markdown(f"**FAISS**: {'🟢 loaded' if faiss_ready_now else '🔴 not loaded'}")
    if file_index_ready and faiss_ready_now:
        sC.success("Ready — you can press Next")
    elif file_index_ready and not faiss_ready_now:
        sC.info("File index is ready. Load/build FAISS if you want similarity search in Step 6.")
    else:
        sC.warning("Build/load the file index first (section A)")

    st.markdown(
        "**Why are there two indexes?**\n"
        "- **File index** (`index.csv`): maps an image **filename** to its **full_path** (and optional **stable_id**). This is for loading/previewing images fast and reliably.\n"
        "- **FAISS index** (`output_dir/faiss/...`): built from your **embeddings vectors**. This is what makes **nearest-neighbor search** fast.\n\n"
        "In Step 6, FAISS returns *IDs/keys* for nearest neighbors; the file index is how the app turns those keys back into real files on disk (thumbnails + open image)."
    )

    st.markdown("---")
    st.markdown("**A) File index (filename → full_path + stable_id)**")
    st.caption(
        "Needed for: displaying query/results images and avoiding ambiguity when multiple images share the same basename. "
        "This index does not contain vectors and is not used for similarity math."
    )

    try:
        idx_path_now = st.session_state.get("index_csv_path")
        fm_now = st.session_state.get("filemap", {}) or {}
        kind_now = str(st.session_state.get("_index_key_kind", "filename"))
        status_bits = []
        if idx_path_now:
            status_bits.append(f"source: {idx_path_now}")
        else:
            status_bits.append("source: (in-memory)")
        status_bits.append(f"entries: {len(fm_now):,}")
        status_bits.append(f"key kind: {kind_now}")
        st.caption(" | ".join(status_bits))
    except Exception:
        pass

    cA, cB = st.columns(2)
    if cA.button("📖 Load existing index", key="load_index"):
        ok, idx_path, fm, df_idx = try_load_existing_index(st.session_state["output_dir"], st.session_state["index_name"])
        if ok:
            _apply_loaded_index(idx_path, fm, df_idx)
            st.success(f"Loaded index: {idx_path} ({len(fm):,})")
            st_dataframe_stretch(df_idx.head(20))
        else:
            st.warning("No existing index found.")

    if cB.button("🏗️ Build index file", key="build_index_file"):
        bar = st.empty()

        def prog(done, total):
            pct = int(100 * done / max(1, total))
            bar.progress(pct, text=f"Indexing files: {done}/{total}")

        with st.status("Building index…", expanded=True) as s:
            ok, msg = _load_or_build_index_to_state(
                st.session_state["output_dir"],
                st.session_state["index_name"],
                st.session_state["images_root"],
                progress_cb=prog,
            )

            # show a preview and offer download if we built in-memory
            df_idx = st.session_state.get("index_df", pd.DataFrame())
            try:
                st_dataframe_stretch(df_idx.head(20))
            except Exception:
                pass

            if st.session_state.get("index_csv_path"):
                st.success(f"✅ {msg}")
            else:
                st.success(f"✅ {msg}")
                if df_idx is not None and not getattr(df_idx, "empty", True):
                    st.download_button(
                        "⬇️ Download index.csv",
                        data=df_idx.to_csv(index=False).encode("utf-8"),
                        file_name=st.session_state["index_name"],
                        mime="text/csv",
                    )
            s.update(label="✅ Index ready", state="complete")

    st.markdown("---")
    st.markdown("**B) FAISS vector index (built from embeddings)**")
    st.caption(
        "Needed for: fast nearest-neighbor search. You only need to build this once per (embeddings file + metric). "
        "If a saved FAISS cache exists, loading it is instant."
    )

    if not _FAISS_OK:
        st.error("❌ FAISS not installed. Run: pip install faiss-cpu")
    else:
        out_dir = str(st.session_state.get("output_dir", "") or "")
        emb_path = str(st.session_state.get("embed_path", "") or "")
        metric = str(st.session_state.get("index_metric", "Cosine (IP + normalization)"))
        paths = _faiss_cache_paths(out_dir, emb_path, metric)

        try:
            saved_exists = bool(paths["index"].exists() and paths["keys"].exists())
        except Exception:
            saved_exists = False

        st.caption(
            " | ".join(
                [
                    f"metric: {metric}",
                    f"cache id: {_faiss_cache_id(emb_path, metric)}",
                    f"saved cache: {'yes' if saved_exists else 'no'}",
                    f"in memory: {'yes' if bool(st.session_state.get('faiss_ready')) else 'no'}",
                ]
            )
        )
        st.caption(f"Cache dir: {paths['dir']}")

        c1, c2, c3, c4 = st.columns(4)
        if c1.button("📥 Load saved FAISS", key="faiss_load_saved_step5"):
            ok, msg = _load_faiss_cache_to_state(out_dir, emb_path, metric)
            if ok:
                st.session_state["_step4_flash"] = msg
                safe_rerun()
            else:
                st.warning(msg)

        if c2.button("🏗️ Build FAISS index", key="build_faiss_index_step5"):
            bar = st.empty()

            def prog(done, total):
                pct = int(100 * done / max(1, total))
                bar.progress(pct, text=f"Adding vectors: {done}/{total}")

            with st.status("Building FAISS…", expanded=True) as s:
                try:
                    index, keys, dim, pos_map, vec_cols, key_kind = build_faiss_index(
                        st.session_state["embed_path"],
                        st.session_state["index_metric"],
                        progress_cb=prog,
                    )
                    st.session_state["_faiss_index"] = index
                    st.session_state["embed_filenames"] = keys
                    st.session_state["_faiss_pos_map"] = pos_map
                    st.session_state["faiss_dim"] = dim
                    st.session_state["_faiss_vec_cols"] = list(vec_cols)[:200]
                    st.session_state["_faiss_key_kind"] = str(key_kind)
                    st.session_state["_faiss_metric_built"] = str(st.session_state.get("index_metric"))
                    st.session_state["faiss_ready"] = True
                    s.update(label=f"✅ FAISS ready ({len(keys):,} vectors, dim={dim}, key={key_kind})", state="complete")

                    # Save alongside other index artifacts under output_dir
                    if st.session_state.get("auto_save_faiss", True):
                        ok2, msg2 = _save_faiss_cache(
                            str(st.session_state.get("output_dir", "")),
                            str(st.session_state.get("embed_path", "")),
                            str(st.session_state.get("index_metric", "Cosine (IP + normalization)")),
                            index,
                            keys,
                            int(dim),
                            list(vec_cols),
                            str(key_kind),
                        )
                        if ok2:
                            st.info(f"💾 {msg2}")
                        else:
                            st.caption(f"FAISS not saved: {msg2}")

                    # Strong warning: filename keys are collision-prone.
                    if str(key_kind) != "stable_id" and str(st.session_state.get("_index_key_kind", "filename")) == "stable_id":
                        st.warning(
                            "⚠️ Embeddings CSV has no stable_id column, so FAISS keys fall back to filename. "
                            "This can produce incorrect matches if basenames collide. Recommended: regenerate embeddings with stable_id and rebuild FAISS."
                        )
                except Exception as e:
                    st.session_state["faiss_ready"] = False
                    st.error(str(e))
                    s.update(label="❌ FAISS build failed", state="error")

        if c3.button("🧹 Clear FAISS (memory)", key="faiss_clear_memory_step5"):
            _clear_faiss_from_state()
            st.session_state["_step4_flash"] = "Cleared FAISS from memory."
            safe_rerun()

        if c4.button("💾 Save FAISS now", key="faiss_save_now_step5"):
            if not st.session_state.get("faiss_ready"):
                st.warning("Build or load FAISS first.")
            else:
                ok, msg = _save_faiss_cache(
                    out_dir,
                    emb_path,
                    metric,
                    st.session_state.get("_faiss_index"),
                    list(st.session_state.get("embed_filenames") or []),
                    int(st.session_state.get("faiss_dim") or 0),
                    list(st.session_state.get("_faiss_vec_cols") or []),
                    str(st.session_state.get("_faiss_key_kind", "filename")),
                )
                if ok:
                    st.success(msg)
                else:
                    st.warning(msg)

    cL, cR = st.columns(2)
    cL.button("⬅️ Back", on_click=lambda: st.session_state.update(step=3), key="back_step_4")
    cR.button("➡️ Next", disabled=not file_index_ready, on_click=lambda: st.session_state.update(step=5), key="next_step_4")


# ========== Step 5: Consistency check ==========
elif st.session_state["step"] == 5:
    st.markdown("**Step 5: Consistency check (metadata vs embeddings vs disk + indexes)**")
    st.info(
        "This step checks whether your metadata, embeddings, disk images, and (if available) the two index artifacts agree. "
        "It helps catch missing files and mismatched exports before searching."
    )

    # Status indicators: help users see if they should go back to Step 4 or can just run the check.
    file_index_ready = False
    try:
        fm_now = st.session_state.get("filemap", {}) or {}
        idx_df_now = st.session_state.get("index_df", pd.DataFrame())
        file_index_ready = bool(fm_now) or (isinstance(idx_df_now, pd.DataFrame) and not idx_df_now.empty)
    except Exception:
        file_index_ready = False

    faiss_ready_now = bool(st.session_state.get("faiss_ready"))
    faiss_saved_exists = False
    try:
        out_dir = str(st.session_state.get("output_dir", "") or "")
        emb_path = str(st.session_state.get("embed_path", "") or "")
        metric = str(st.session_state.get("index_metric", "Cosine (IP + normalization)"))
        paths = _faiss_cache_paths(out_dir, emb_path, metric)
        faiss_saved_exists = bool(paths["index"].exists() and paths["keys"].exists())
    except Exception:
        faiss_saved_exists = False

    sA, sB, sC, sD = st.columns([1, 1, 1, 2])
    sA.markdown(f"**File index**: {'🟢 ready' if file_index_ready else '🔴 missing'}")
    sB.markdown(f"**FAISS (memory)**: {'🟢 loaded' if faiss_ready_now else '🔴 not loaded'}")
    sC.markdown(f"**FAISS (saved)**: {'🟢 exists' if faiss_saved_exists else '🔴 none'}")
    if not file_index_ready:
        sD.warning("Go back to Step 4 and build/load the file index.")
    elif not faiss_ready_now and faiss_saved_exists:
        sD.info("Optional: go back to Step 4 and click ‘Load saved FAISS’ for faster Step 6.")
    elif not faiss_ready_now and not faiss_saved_exists:
        sD.info("Optional: go back to Step 4 and build FAISS to enable similarity search.")
    else:
        sD.success("Looks good — run the check or press Next")

    if st.button("🧪 Run check", key="run_check"):
        with st.status("Running checks…", expanded=True) as s:
            try:
                res = _run_consistency_check(
                    st.session_state["meta_path"],
                    st.session_state["embed_path"],
                    st.session_state["images_root"],
                )

                st.write(f"Metadata: **{res['counts']['meta']:,}**")
                st.write(f"Embeddings (unique filenames): **{res['counts']['emb']:,}**")
                st.write(f"Disk images: **{res['counts']['disk']:,}**")
                if int(res["counts"].get("file_index", 0)):
                    st.write(f"File index (unique filenames): **{int(res['counts']['file_index']):,}**")
                if int(res["counts"].get("file_index_stable_id", 0)):
                    st.write(f"File index (stable_id): **{int(res['counts']['file_index_stable_id']):,}**")
                fa = res.get("faiss", {}) if isinstance(res.get("faiss"), dict) else {}
                if fa:
                    st.write(
                        f"FAISS cache: **{'in memory' if fa.get('in_memory') else ('saved' if fa.get('saved_exists') else 'missing')}**"
                        + (f" (key={fa.get('key_kind')})" if fa.get("key_kind") else "")
                    )

                st.markdown("**Top 10 mismatches**")
                st.write("In metadata but missing on disk:", res["miss_meta_on_disk"] or "—")
                st.write("In embeddings but missing on disk:", res["miss_emb_on_disk"] or "—")
                st.write("In metadata but not in embeddings:", res["miss_meta_in_emb"] or "—")
                st.write("In embeddings but not in metadata:", res["miss_emb_in_meta"] or "—")
                if res.get("miss_index_on_disk") is not None:
                    st.write("In file index but missing on disk:", res.get("miss_index_on_disk") or "—")
                if res.get("miss_faiss_in_emb"):
                    st.write("In FAISS but not in embeddings:", res.get("miss_faiss_in_emb") or "—")
                if res.get("miss_faiss_in_index"):
                    st.write("In FAISS but not in file index:", res.get("miss_faiss_in_index") or "—")

                s.update(label="✅ Checks complete", state="complete")
            except Exception as e:
                st.error(str(e))
                s.update(label="❌ Checks failed", state="error")

    cL, cR = st.columns(2)
    cL.button("⬅️ Back", on_click=lambda: st.session_state.update(step=4), key="back_step_5")
    cR.button("➡️ Next", on_click=lambda: st.session_state.update(step=6), key="next_step_5")


# ========== Step 6: Query + Filters + KNN ==========
elif st.session_state["step"] == 6:
    st.markdown("**Step 6: Pick image -> filters -> find K nearest**")

    st.info(
        "Choose a query image and find visually similar images (nearest neighbors). "
        "You can optionally filter by metadata (for example: period, artist, collection) before searching."
    )

    if not _FAISS_OK:
        st.error("❌ FAISS not installed. Run: pip install faiss-cpu")
    elif not st.session_state.get("faiss_ready"):
        st.warning("FAISS is not loaded yet. Go back to Step 4 to load/build it.")
        if st.button("⬅️ Go to Step 4 (FAISS)", key="go_to_step4_for_faiss"):
            st.session_state.update(step=4)
            safe_rerun()

    st.markdown("**Query mode**")
    mode = st.selectbox(
        "Mode",
        ["Whole-image", "Text", "Crop (light)", "Crop (patch index)"],
        index=["Whole-image", "Text", "Crop (light)", "Crop (patch index)"].index(st.session_state.get("query_mode", "Whole-image")),
        key="query_mode",
    )

    # Element-search alignment status (helps explain why Text/Crop(light) can be wrong).
    if mode in ("Text", "Crop (light)"):
        with st.expander("Element search: alignment check", expanded=False):
            if st.button("Check alignment (SigLIP2 vs FAISS)", key="check_element_alignment"):
                cos, msg = compute_element_alignment_sample()
                st.session_state["_element_alignment_cos"] = cos
                st.session_state["_element_alignment_msg"] = msg

            cos = st.session_state.get("_element_alignment_cos")
            msg = st.session_state.get("_element_alignment_msg")
            if msg:
                st.caption(str(msg))
            if cos is None:
                st.info("No alignment score computed yet.")
            else:
                st.write(f"Cosine alignment (sample): **{float(cos):.4f}**")
                if float(cos) < 0.90:
                    st.warning(
                        "Low alignment (<0.90). Text/Crop(light) search will look random because your embeddings/FAISS are not in SigLIP2 space. "
                        "Fix: generate embeddings CSV using SigLIP2 image embeddings (same model as element encoder), then rebuild FAISS."
                    )

    if mode == "Text" and not _ELEMENT_OK:
        st.error("Element search requires Hugging Face deps. Install: `pip install transformers torch`")

    st.markdown("**Choose query image**")
    fm = st.session_state.get("filemap", {}) or {}
    idx_df = st.session_state.get("index_df", pd.DataFrame())
    index_key_kind = str(st.session_state.get("_index_key_kind", "filename"))
    sid_to_filename = st.session_state.get("_sid_to_filename", {}) or {}

    if mode != "Text":
        cU, cP = st.columns(2)
        uploaded = cU.file_uploader("Upload image", type=[e.strip(".") for e in sorted(SUPPORTED_IMG)])
        if uploaded is not None:
            name = Path(uploaded.name).name
            fn_key = _fname_key(name)
            st.session_state["query_filename_key"] = fn_key

            # Prefer stable_id lookup when index provides it.
            matched_key = None
            try:
                if index_key_kind == "stable_id" and not idx_df.empty and "stable_id" in idx_df.columns:
                    tmp = _std_cols(idx_df.copy())
                    tmp = _ensure_stable_id(tmp)
                    tmp = _ensure_filename(tmp)
                    hit = tmp.loc[tmp.get("filename").astype(str) == fn_key]
                    if not hit.empty:
                        matched_key = str(hit.iloc[0].get("stable_id", "")).strip() or None
            except Exception:
                matched_key = None

            if matched_key is None:
                matched_key = fn_key

            if matched_key in fm and Path(fm[matched_key]).exists():
                st.session_state["query_key"] = matched_key
                st.session_state["query_image_path"] = fm[matched_key]
                # also keep stable_id if we can (useful for alignment/debug)
                try:
                    pp = Path(fm[matched_key])
                    stt = pp.stat()
                    st.session_state["query_stable_id"] = _stable_id(pp, int(stt.st_size), int(stt.st_mtime))
                except Exception:
                    st.session_state["query_stable_id"] = ""
                st.success(f"Matched index file: {name}")
            else:
                tmp_dir = Path(st.session_state["output_dir"] or ".")
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = tmp_dir / name
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                st.session_state["query_image_path"] = str(tmp_path)
                # Always keep the filename key for FAISS filename-keyed searches.
                st.session_state["query_key"] = fn_key
                # Also compute stable_id for alignment/debug (may not exist in embeddings).
                try:
                    pp = Path(tmp_path)
                    stt = pp.stat()
                    st.session_state["query_stable_id"] = _stable_id(pp, int(stt.st_size), int(stt.st_mtime))
                except Exception:
                    st.session_state["query_stable_id"] = ""
                st.info(f"Saved temp: {tmp_path}")

        if fm:
            filt = cP.text_input("Filter filenames (substring)", "")
            all_keys = sorted(list(fm.keys()))
            if filt.strip():
                q = filt.lower()
                show = [
                    k
                    for k in all_keys
                    if (q in str(k).lower()) or (q in str(sid_to_filename.get(k, "")).lower())
                ]
            else:
                show = all_keys[:1000]

            def _fmt_key(k: str) -> str:
                if index_key_kind == "stable_id":
                    nm = sid_to_filename.get(k, "")
                    return f"{nm} ({k})" if nm else str(k)
                return str(k)

            sel = cP.selectbox("Pick from index", options=["(none)"] + show, format_func=lambda x: "(none)" if x == "(none)" else _fmt_key(str(x)))
            if sel != "(none)" and cP.button("Use selected", key="use_selected_filename"):
                p = fm.get(str(sel), "")
                if p and Path(p).exists():
                    st.session_state["query_key"] = str(sel)
                    st.session_state["query_image_path"] = p
                    st.session_state["query_filename_key"] = _fname_key(Path(p).name)
                    try:
                        pp = Path(p)
                        stt = pp.stat()
                        st.session_state["query_stable_id"] = _stable_id(pp, int(stt.st_size), int(stt.st_mtime))
                    except Exception:
                        st.session_state["query_stable_id"] = ""
                    st.success(f"Selected: {Path(p).name}")
                else:
                    st.error("Path not found for selected key.")
    else:
        st.session_state.setdefault("text_query", "")
        st.text_area(
            "Text query (one prompt per line)",
            key="text_query",
            height=90,
            help="Tip: add multiple phrasings on separate lines; we’ll embed each and average for more robust retrieval.",
        )

        # Prompting can materially change SigLIP2 text->image retrieval quality.
        st.session_state.setdefault("text_prompt_template", "{text}")
        st.session_state.setdefault("text_prompt_ensemble", True)
        tpl = st.selectbox(
            "Prompt template",
            options=[
                "{text}",
                "This is a photo of {text}.",
                "This is an image of {text}.",
                "a photo of {text}",
                "an image of {text}",
                "a painting of {text}",
                "a drawing of {text}",
                "an illustration of {text}",
                "a poster showing {text}",
                "a wall chart showing {text}",
            ],
            index=[
                "{text}",
                "This is a photo of {text}.",
                "This is an image of {text}.",
                "a photo of {text}",
                "an image of {text}",
                "a painting of {text}",
                "a drawing of {text}",
                "an illustration of {text}",
                "a poster showing {text}",
                "a wall chart showing {text}",
            ].index(str(st.session_state.get("text_prompt_template", "{text}")) if str(st.session_state.get("text_prompt_template", "{text}")) in {
                "{text}",
                "This is a photo of {text}.",
                "This is an image of {text}.",
                "a photo of {text}",
                "an image of {text}",
                "a painting of {text}",
                "a drawing of {text}",
                "an illustration of {text}",
                "a poster showing {text}",
                "a wall chart showing {text}",
            } else "{text}"),
            key="text_prompt_template",
            help="If Text retrieval feels weak, try different templates or enable the ensemble below. (SigLIP docs often use phrasing like 'This is a photo of …'.)",
        )
        st.session_state.setdefault("text_prompt_template", tpl)
        st.checkbox(
            "Use prompt ensemble (recommended)",
            key="text_prompt_ensemble",
            help="Embeds several prompt variants and averages them; often more robust than a single phrasing.",
        )

    qpath = st.session_state.get("query_image_path", "")
    if mode != "Text":
        st.write("Selected path:")
        st.code(qpath or "(none)")
        if qpath and Path(qpath).exists():
            display_image(qpath, width=360)

        # Ensure we have a query_key in session_state.
        if not st.session_state.get("query_key") and qpath and Path(qpath).exists():
            try:
                pp = Path(qpath)
                stt = pp.stat()
                st.session_state["query_filename_key"] = _fname_key(pp.name)
                st.session_state["query_stable_id"] = _stable_id(pp, int(stt.st_size), int(stt.st_mtime))
                st.session_state["query_key"] = st.session_state["query_stable_id"] if index_key_kind == "stable_id" else st.session_state["query_filename_key"]
            except Exception:
                st.session_state["query_key"] = _fname_key(Path(qpath).name)

        # Debug: element_encoder alignment (whole-image embedding vs FAISS reconstruct)
        # Important: embedding via transformers/torch can be slow, so do NOT run automatically on every rerun.
        with st.expander("Debug: element_encoder alignment", expanded=False):
            st.caption("Compares FAISS.reconstruct(query) vs SigLIP2(embed_image(query)). Can be slow; run on demand.")
            if not _ELEMENT_OK or _embed_image is None:
                st.info("Element encoder not available (install transformers+torch).")
            elif not st.session_state.get("faiss_ready"):
                st.info("Build/load the FAISS index first.")
            elif not (qpath and Path(qpath).exists()):
                st.info("Select a query image first.")
            else:
                faiss_cache_id = str(st.session_state.get("_faiss_cache_id") or "")
                try:
                    model_id = str(_current_element_model_id()) if (_ELEMENT_OK and _current_element_model_id is not None) else str(os.environ.get("REWIZ_SIGLIP2_MODEL", ""))
                except Exception:
                    model_id = str(os.environ.get("REWIZ_SIGLIP2_MODEL", ""))

                st.session_state.setdefault("_query_align_result", {})
                res = st.session_state.get("_query_align_result") or {}
                cached_ok = bool(res) and (
                    str(res.get("qpath")) == str(qpath)
                    and str(res.get("faiss_cache_id")) == faiss_cache_id
                    and str(res.get("model_id")) == model_id
                )

                cA, cB = st.columns(2)
                run_now = cA.button("Compute alignment", key="compute_query_alignment")
                if cB.button("Clear cached result", key="clear_query_alignment"):
                    st.session_state["_query_align_result"] = {}
                    cached_ok = False

                if run_now or cached_ok:
                    if run_now:
                        try:
                            key_kind = str(st.session_state.get("_faiss_key_kind", "filename"))
                            pos_map = st.session_state.get("_faiss_pos_map") or {}
                            index = st.session_state.get("_faiss_index")
                            if index is None:
                                raise RuntimeError("FAISS index missing in session.")

                            qkey = _query_key_for_faiss(qpath, key_kind)
                            if not qkey:
                                raise RuntimeError("No query key available")
                            if qkey not in pos_map:
                                st.session_state["_query_align_result"] = {
                                    "qpath": str(qpath),
                                    "faiss_cache_id": faiss_cache_id,
                                    "model_id": model_id,
                                    "qkey": str(qkey),
                                    "found": False,
                                    "message": "Query key not found in FAISS (expected for uploaded temp images not in embeddings).",
                                }
                            else:
                                pos = int(pos_map[qkey])
                                v_faiss = np.asarray(index.reconstruct(pos), dtype="float32").reshape(1, -1)
                                v_elem = np.asarray(_embed_image(qpath, crop_box=None), dtype="float32").reshape(1, -1)

                                an = v_faiss.ravel() / max(float(np.linalg.norm(v_faiss)), 1e-12)
                                bn = v_elem.ravel() / max(float(np.linalg.norm(v_elem)), 1e-12)
                                cos = float((an * bn).sum())

                                st.session_state["_query_align_result"] = {
                                    "qpath": str(qpath),
                                    "faiss_cache_id": faiss_cache_id,
                                    "model_id": model_id,
                                    "qkey": str(qkey),
                                    "found": True,
                                    "faiss_dim": int(v_faiss.shape[1]),
                                    "elem_dim": int(v_elem.shape[1]),
                                    "cos": float(cos),
                                }
                        except Exception as e:
                            st.session_state["_query_align_result"] = {
                                "qpath": str(qpath),
                                "faiss_cache_id": faiss_cache_id,
                                "model_id": model_id,
                                "found": False,
                                "error": str(e),
                            }

                    res2 = st.session_state.get("_query_align_result") or {}
                    if res2.get("error"):
                        st.error(f"Alignment debug failed: {res2.get('error')}")
                    elif not bool(res2.get("found")):
                        st.warning(str(res2.get("message") or "No alignment result"))
                        if res2.get("qkey"):
                            st.caption(f"qkey: {res2.get('qkey')}")
                    else:
                        st.write(f"FAISS dim: {int(res2.get('faiss_dim') or 0)} | element_encoder dim: {int(res2.get('elem_dim') or 0)}")
                        st.write(f"Cosine(reconstruct, element_encoder): **{float(res2.get('cos') or 0.0):.4f}**")
                        if int(res2.get("faiss_dim") or 0) != int(res2.get("elem_dim") or 0):
                            st.error("Dim mismatch: embeddings/FAISS and element_encoder are not aligned.")
                        elif float(res2.get("cos") or 0.0) < 0.90:
                            st.warning("Low alignment (cos < 0.90). This usually means different models/pipelines were used to generate the embeddings CSV vs the runtime element encoder.")
                        else:
                            st.success("Alignment looks good (cos ≥ 0.90).")

    crop_box: Optional[Tuple[int, int, int, int]] = None
    armed_crop_matches_current = False
    if mode in ("Crop (light)", "Crop (patch index)"):
        if not _ELEMENT_OK:
            st.error("Crop search requires Hugging Face deps. Install: `pip install transformers torch`")
        elif not (qpath and Path(qpath).exists()):
            st.warning("Select or upload an image to crop.")
        else:
            with st.expander("Crop selection", expanded=True):
                require_coords = (mode == "Crop (patch index)")
                if require_coords:
                    st.info("Patch-index crop search needs crop coordinates (sliders). The cropper component returns cropped image bytes but not coordinates.")

                crop_box = _crop_box_ui(qpath, require_coords=require_coords)

                if mode == "Crop (light)":
                    cb = st.session_state.get("_cropper_last_bytes")
                    sp = str(st.session_state.get("_cropper_last_source_path") or "")
                    try:
                        armed_crop_matches_current = bool(cb) and bool(sp) and (Path(sp).resolve() == Path(qpath).resolve())
                    except Exception:
                        armed_crop_matches_current = bool(cb) and bool(sp) and (sp == str(qpath))

                    if armed_crop_matches_current:
                        st.success("Crop is armed — press Search.")
                    elif crop_box is None:
                        st.info("No armed crop for the current image yet. Click the cropper’s 'Use crop' button (or enable manual sliders).")

                    if crop_box is not None:
                        st.write(f"Crop box: {crop_box}")

                    st.session_state.setdefault("crop_light_run_sanity", False)
                    st.checkbox(
                        "Run match sanity (also embed full image)",
                        key="crop_light_run_sanity",
                        help="Computes cosine(full_image, crop) so you can verify the crop embedding differs from the full-image embedding.",
                    )
                    if bool(st.session_state.get("crop_light_run_sanity", False)):
                        st.caption("Sanity runs when you press Search. Result shows below (and in Crop(light) debug).")
                        if st.session_state.get("_crop_light_last_sanity_cos") is not None:
                            try:
                                st.caption(
                                    f"Last sanity cosine(full, crop): {float(st.session_state.get('_crop_light_last_sanity_cos')):0.4f}"
                                )
                            except Exception:
                                pass

                else:
                    # Patch-index mode always needs coordinates
                    if crop_box is None:
                        st.warning("Invalid crop box.")
                    else:
                        st.write(f"Crop box: {crop_box}")

    st.markdown("---")
    st.markdown("**Filters (limits candidates via metadata)**")

    meta_df = None
    try:
        meta_df = load_metadata(st.session_state["meta_path"]) if st.session_state.get("meta_ok") else None
    except Exception:
        meta_df = None

    if meta_df is None:
        st.info("No metadata loaded -> no metadata filters (all embeddings can be used).")
    else:
        # choose columns to show in results
        opts = [c for c in meta_df.columns if c != "filename"]
        st.session_state["result_meta_cols"] = st.multiselect(
            "Metadata columns to show in results",
            options=opts,
            default=st.session_state.get("result_meta_cols") or opts[:6],
        )

        col = st.selectbox("Column to filter on", options=list(meta_df.columns))
        vals = meta_df[col].dropna().astype(str).unique().tolist()
        vals = sorted(vals)
        search = st.text_input("Search values (substring)", "")
        vals_show = [v for v in vals if search.lower() in v.lower()] if search.strip() else vals[:400]

        c1, c2 = st.columns(2)
        # Assume stored filters are lists; default to empty list if missing
        saved_inc = st.session_state.get("filter_include", {}).get(col, []) or []
        saved_exc = st.session_state.get("filter_exclude", {}).get(col, []) or []

        inc = c1.multiselect("➕ Include", options=vals_show, default=[v for v in (saved_inc) if v in vals_show])
        exc = c2.multiselect("➖ Exclude", options=vals_show, default=[v for v in (saved_exc) if v in vals_show])

        c3, c4, c5 = st.columns(3)
        if c3.button("Save include", key="save_include_col"):
            # Always store lists (possibly empty)
            st.session_state.setdefault("filter_include", {})
            st.session_state["filter_include"][col] = list(inc)
        if c4.button("Save exclude", key="save_exclude_col"):
            st.session_state.setdefault("filter_exclude", {})
            st.session_state["filter_exclude"][col] = list(exc)
        if c5.button("Clear for column", key="clear_filter_col"):
            st.session_state["filter_include"].pop(col, None)
            st.session_state["filter_exclude"].pop(col, None)

        st.write("Active include:", st.session_state["filter_include"] or "—")
        st.write("Active exclude:", st.session_state["filter_exclude"] or "—")

        st.session_state["filter_query_text"] = st.text_input(
            "Optional pandas.query() (advanced)",
            value=st.session_state.get("filter_query_text", ""),
            help="Example: year >= 1900 and year < 1950",
        )

    st.markdown("---")
    # Patch-index management (for Crop (patch index) mode)
    patch_dir = Path(st.session_state.get("output_dir") or ".")
    default_index_csv = None
    if st.session_state.get("index_csv_path") and Path(str(st.session_state.get("index_csv_path"))).exists():
        default_index_csv = str(st.session_state.get("index_csv_path"))
    else:
        candidate = patch_dir / (st.session_state.get("index_name") or "index.csv")
        if candidate.exists():
            default_index_csv = str(candidate)

    # New layout (preferred): output_dir/patch_index/...
    if _PATCH_BACKEND_OK and _patch_backend is not None:
        pb_paths = _patch_backend.get_patch_index_paths(patch_dir)
        patch_index_path = pb_paths.faiss_index
        patch_meta_parquet = pb_paths.meta_parquet
        patch_meta_csv = pb_paths.meta_csv
        patch_vectors_npy = pb_paths.vectors_npy
        patch_vectors_npz = pb_paths.vectors_npz
    else:
        pb_paths = None
        patch_index_path = patch_dir / "patch.faiss"
        patch_meta_parquet = patch_dir / "patch_meta.parquet"
        patch_meta_csv = patch_dir / "patch_meta.csv"
        patch_vectors_npy = patch_dir / "patch_vectors.npy"
        patch_vectors_npz = patch_dir / "patch_vectors.npz"

    # Backward-compat (old layout): files directly in output_dir
    legacy_patch_index_path = patch_dir / "patch.faiss"
    legacy_patch_meta_path = patch_dir / "patch_meta.csv"

    if mode == "Crop (patch index)":
        st.markdown("**Patch index**")
        st.write("Expected files under:")
        st.code(str((patch_dir / 'patch_index').resolve()) if (patch_dir / 'patch_index').exists() else str(patch_dir.resolve()))
        st.write(f"FAISS: {patch_index_path.name}")
        st.write(f"Meta: {patch_meta_parquet.name} (or {patch_meta_csv.name})")
        st.caption("Vectors file is optional at query time; FAISS reconstruct() is used when possible.")

        cA, cB, cC = st.columns(3)
        if cA.button("📥 Load patch index", key="load_patch_index"):
            try:
                assert faiss is not None

                # Prefer new layout; fall back to legacy layout.
                idx_path = patch_index_path
                if not idx_path.exists() and legacy_patch_index_path.exists():
                    idx_path = legacy_patch_index_path

                if not idx_path.exists():
                    raise FileNotFoundError("patch.faiss not found (build patch index first)")

                st.session_state["_patch_faiss_index"] = faiss.read_index(str(idx_path))

                # Load metadata
                meta_path = None
                meta_df = None
                if _PATCH_BACKEND_OK and _patch_backend is not None and pb_paths is not None:
                    meta_df = _patch_backend.load_patch_meta(pb_paths)
                    # store a preferred meta path for debugging
                    meta_path = str((pb_paths.meta_parquet if pb_paths.meta_parquet.exists() else pb_paths.meta_csv).resolve())
                else:
                    # legacy: CSV with columns patch_id, filename, x0,y0,x1,y1
                    if legacy_patch_meta_path.exists():
                        meta_path = str(legacy_patch_meta_path)
                        meta_df = pd.read_csv(meta_path, low_memory=False)
                        meta_df.columns = [str(c).strip().lower() for c in meta_df.columns]
                    elif patch_meta_csv.exists():
                        meta_path = str(patch_meta_csv)
                        meta_df = pd.read_csv(meta_path, low_memory=False)
                        meta_df.columns = [str(c).strip().lower() for c in meta_df.columns]

                if meta_df is None or meta_df.empty:
                    raise RuntimeError("Failed to load patch metadata")

                st.session_state["_patch_meta"] = meta_df
                # Keep for backwards compatibility, but do not load huge mmap arrays by default.
                # Windows can throw OSError 1455 (pagefile too small) when mapping very large .npy files.
                st.session_state["_patch_vectors"] = None

                st.session_state["patch_index_ready"] = True
                st.session_state["patch_index_path"] = str(idx_path)
                st.session_state["patch_meta_path"] = str(meta_path or "")
                st.success("Patch index loaded.")
            except Exception as e:
                st.session_state["patch_index_ready"] = False
                st.error(repr(e) if isinstance(e, OSError) else str(e))

        if cB.button("🏗️ Build patch index", key="build_patch_index"):
            if default_index_csv is None:
                st.error("No index.csv found (build Step 4 index first, or set output_dir/index_name).")
            else:
                cmd = [
                    sys.executable,
                    str((_HERE / "build_patch_index.py").resolve()),
                    "--index_csv",
                    str(Path(default_index_csv).resolve()),
                    "--out_dir",
                    str(patch_dir.resolve()),
                ]
                with st.status("Building patch index…", expanded=True) as s:
                    try:
                        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                        st.code(" ".join(cmd))
                        if proc.stdout:
                            st.text(proc.stdout)
                        if proc.stderr:
                            st.text(proc.stderr)
                        if proc.returncode != 0:
                            raise RuntimeError(f"Patch index build failed (exit {proc.returncode})")
                        s.update(label="✅ Patch index built", state="complete")
                    except Exception as e:
                        s.update(label="❌ Patch index build failed", state="error")
                        st.error(str(e))

        if cC.button("↻ Clear patch index", key="clear_patch_index"):
            st.session_state["_patch_faiss_index"] = None
            st.session_state["_patch_meta"] = None
            st.session_state["_patch_vectors"] = None
            st.session_state["patch_index_ready"] = False
            st.session_state["patch_meta_path"] = ""
            st.session_state["patch_index_path"] = ""

        st.write("Patch index ready:", bool(st.session_state.get("patch_index_ready")))

    # Search readiness depends on mode.
    if mode == "Whole-image":
        can_search = bool(st.session_state.get("faiss_ready")) and bool(qpath)
    elif mode == "Text":
        can_search = bool(st.session_state.get("faiss_ready")) and bool(st.session_state.get("text_query", "").strip())
    elif mode == "Crop (light)":
        can_search = bool(st.session_state.get("faiss_ready")) and bool(qpath) and (armed_crop_matches_current or (crop_box is not None))
    elif mode == "Crop (patch index)":
        can_search = bool(st.session_state.get("patch_index_ready")) and bool(qpath) and crop_box is not None
    else:
        can_search = False

    form_key = {
        "Whole-image": "knn_form_whole",
        "Text": "knn_form_text",
        "Crop (light)": "knn_form_crop_light",
        "Crop (patch index)": "knn_form_crop_patch",
    }.get(mode, "knn_form")

    with st.form(form_key):
        k = st.number_input("K neighbors", min_value=1, max_value=200, value=int(st.session_state["k_neighbors"]))
        img_w = st.slider("Image width (px)", 160, 1200, 320, step=20)
        do = st.form_submit_button("🔎 Search", disabled=not can_search)

    if do:
        st.session_state["k_neighbors"] = int(k)
        st.session_state["_last_search_ok"] = False
        st.session_state["_last_search_mode"] = str(mode)

        # Persist “what query was used” so Step 7 can display the correct query context
        # (text prompts vs crop vs full image), even if query_image_path is left over from earlier.
        st.session_state["_last_query_mode"] = str(mode)
        if mode != "Text":
            st.session_state["_last_query_image_path"] = str(st.session_state.get("query_image_path") or "")

        key_kind = str(st.session_state.get("_faiss_key_kind", "filename"))
        allowed = build_allowed_set(meta_df, key_kind=key_kind) if meta_df is not None else None

        # Use the metric that matches the built FAISS index (important: changing the UI metric
        # after building FAISS will otherwise mis-normalize queries and misinterpret scores).
        metric_ui = str(st.session_state.get("index_metric", ""))
        metric_built = str(st.session_state.get("_faiss_metric_built", "") or "").strip() or None
        metric_infer = _faiss_metric_label_from_index(st.session_state.get("_faiss_index")) if st.session_state.get("faiss_ready") else None
        metric_used = metric_built or metric_infer or metric_ui
        if metric_used and metric_ui and (metric_used != metric_ui):
            st.warning(f"FAISS was built for '{metric_used}', but UI is set to '{metric_ui}'. Using '{metric_used}' for this search.")

        with st.status("Searching…", expanded=True) as s:
            try:
                if mode == "Whole-image":
                    qkey = _query_key_for_faiss(qpath, key_kind) or (st.session_state.get("query_key") or Path(qpath).name)
                    df = knn_filtered(
                        st.session_state["_faiss_index"],
                        st.session_state["embed_filenames"],
                        str(qkey),
                        int(k),
                        allowed=allowed,
                        metric=metric_used,
                        pos_map=st.session_state.get("_faiss_pos_map"),
                        key_kind=key_kind,
                    )
                    df = add_cosine_columns_from_faiss(
                        df,
                        st.session_state["_faiss_index"],
                        st.session_state["embed_filenames"],
                        str(qkey),
                        metric=metric_used,
                        pos_map=st.session_state.get("_faiss_pos_map"),
                        key_kind=key_kind,
                    )

                    # Query display context for Step 7
                    st.session_state["_last_query_display_kind"] = "image"
                    st.session_state["_last_query_display_path"] = str(qpath or "")

                elif mode == "Text":
                    if not _ELEMENT_OK or _embed_text is None:
                        raise RuntimeError("Element encoder not available")
                    qtext_raw_all = str(st.session_state.get("text_query", ""))
                    lines = [ln.strip() for ln in qtext_raw_all.splitlines() if str(ln).strip()]
                    if not lines:
                        raise ValueError("Text must be non-empty")
                    tpl = str(st.session_state.get("text_prompt_template", "{text}"))
                    use_ensemble = bool(st.session_state.get("text_prompt_ensemble", True))

                    def _format_prompt(template: str, text: str) -> str:
                        template = str(template)
                        if "{text}" in template:
                            return template.format(text=text)
                        return (template + " " + text).strip()

                    templates = [
                        tpl,
                        "This is a photo of {text}.",
                        "This is an image of {text}.",
                        "a photo of {text}",
                        "an image of {text}",
                        "a painting of {text}",
                        "a depiction of {text}",
                    ]

                    prompts_used: list[str] = []
                    seen: set[str] = set()
                    # Build prompt list from all lines, optionally with a small ensemble per line.
                    for ln in lines:
                        if use_ensemble:
                            for t in templates:
                                p = _format_prompt(t, ln)
                                if p and p not in seen:
                                    prompts_used.append(p)
                                    seen.add(p)
                        else:
                            p = _format_prompt(tpl, ln)
                            if p and p not in seen:
                                prompts_used.append(p)
                                seen.add(p)

                    # Cap to avoid excessive embedding work.
                    prompts_used = prompts_used[:8]

                    vecs = [_embed_text(p) for p in prompts_used]
                    mat = np.concatenate([np.asarray(v, dtype="float32") for v in vecs], axis=0)
                    qvec = np.mean(mat, axis=0, keepdims=True).astype("float32")
                    # Normalize only for cosine/IP indices (knn_filtered_by_vector also normalizes,
                    # but we keep qvec consistent for debug + diagnostics).
                    if "Cosine" in str(metric_used):
                        n = float(np.linalg.norm(qvec))
                        if n > 0:
                            qvec = qvec / n
                    _assert_query_dim_matches_index(qvec, st.session_state["_faiss_index"], "Text encoder")

                    # Persist the exact query vector + prompt(s) so post-search diagnostics can run on reruns
                    # (e.g., when the user clicks a diagnostic button).
                    try:
                        qv_store = np.asarray(qvec, dtype="float32")
                        if qv_store.ndim == 2:
                            qv_store = qv_store.reshape(-1)
                        st.session_state["_last_search_mode"] = "Text"
                        st.session_state["_last_text_query"] = "\n".join(lines)
                        st.session_state["_last_text_prompts_used"] = list(prompts_used)
                        st.session_state["_last_text_metric_used"] = str(metric_used)
                        st.session_state["_last_text_qvec"] = qv_store.tolist()
                    except Exception:
                        pass

                    # Query display context for Step 7
                    st.session_state["_last_query_display_kind"] = "text"
                    st.session_state["_last_query_display_path"] = ""

                    # Quick visibility into what we're actually searching with.
                    with st.expander("Text debug", expanded=False):
                        try:
                            mid = str(_current_element_model_id()) if (_ELEMENT_OK and _current_element_model_id is not None) else ""
                        except Exception:
                            mid = ""
                        st.write(f"model: {mid or st.session_state.get('siglip2_model','')}")
                        st.write(f"metric_used: {metric_used}")
                        qv = np.asarray(qvec, dtype="float32")
                        if qv.ndim == 1:
                            qv = qv.reshape(1, -1)
                        st.write(f"text_lines: {len(lines)}")
                        st.write(lines)
                        st.write(f"prompt_ensemble: {bool(use_ensemble)}")
                        if len(prompts_used) == 1:
                            st.write(f"prompt: {prompts_used[0]}")
                        else:
                            st.write(f"prompts_used: {len(prompts_used)}")
                            st.write(prompts_used)
                        st.write(f"qvec_dim: {int(qv.shape[1])}")
                        st.write(f"qvec_l2_norm: {float(np.linalg.norm(qv)):0.6f}")

                    # Guard against model-space mismatch (dimension might match but semantics won't).
                    cos = st.session_state.get("_element_alignment_cos")
                    if cos is None:
                        cos2, msg2 = compute_element_alignment_sample()
                        st.session_state["_element_alignment_cos"] = cos2
                        st.session_state["_element_alignment_msg"] = msg2
                        cos = cos2
                    if cos is None:
                        raise RuntimeError(
                            "Could not verify element-search alignment against this FAISS index. "
                            "Text results may be meaningless unless the embeddings CSV was generated with SigLIP2 image embeddings. "
                            "Use 'Element search: alignment check' to debug, or rebuild embeddings/FAISS in SigLIP2 space."
                        )
                    if float(cos) < 0.90:
                        raise RuntimeError(
                            f"Element search is not aligned with this FAISS index (sample cosine={float(cos):.3f} < 0.90). "
                            "Text results will be unreliable. Rebuild embeddings/FAISS using SigLIP2 image embeddings."
                        )
                    df = knn_filtered_by_vector(
                        st.session_state["_faiss_index"],
                        st.session_state["embed_filenames"],
                        qvec,
                        int(k),
                        allowed=allowed,
                        metric=metric_used,
                        key_kind=key_kind,
                    )
                    if not df.empty:
                        # Mode-specific scoring fields for display.
                        if "Cosine" in str(metric_used):
                            df["text_metric"] = "cosine"
                            # NOTE: `raw_score` is already cosine similarity for Cosine/IP indices.
                            df["text_score"] = df["raw_score"].astype(float)
                            df["text_angle_deg"] = np.degrees(np.arccos(np.clip(df["text_score"].astype(float), -1.0, 1.0)))
                            df["text_gap_to_next"] = df["text_score"].astype(float) - df["text_score"].astype(float).shift(-1)
                            df["text_stars"] = _stars_1_to_5_from_scores(df["text_score"], higher_is_better=True)
                        else:
                            # For L2 (and other distance-like metrics), smaller is better.
                            df["text_metric"] = "l2"
                            # NOTE: `distance` is already the L2 distance for L2 indices.
                            df["text_score"] = df["distance"].astype(float)
                            df["text_gap_to_next"] = df["text_score"].astype(float).shift(-1) - df["text_score"].astype(float)
                            df["text_stars"] = _stars_1_to_5_from_scores(df["text_score"], higher_is_better=False)

                    with st.expander("Text results debug", expanded=False):
                        st.write(f"results: {int(len(df))}")
                        st.write(
                            {
                                "filters_active": bool(allowed is not None),
                                "allowed_count": (None if allowed is None else int(len(allowed))),
                            }
                        )
                        if not df.empty:
                            scores = df["raw_score"].astype(float)
                            st.write(
                                {
                                    "raw_score_max": float(scores.max()),
                                    "raw_score_min": float(scores.min()),
                                    "raw_score_mean": float(scores.mean()),
                                }
                            )

                            # If Cosine/IP is used, vectors in FAISS should be ~unit length.
                            try:
                                idx = st.session_state.get("_faiss_index")
                                fns = st.session_state.get("embed_filenames") or []
                                if idx is not None and hasattr(idx, "reconstruct") and len(fns) > 0:
                                    top = df.head(5).to_dict("records")
                                    norms = []
                                    for r in top:
                                        fn = str(r.get("filename", ""))
                                        if not fn:
                                            continue
                                        try:
                                            ipos = int(fns.index(fn))
                                        except Exception:
                                            continue
                                        v = np.asarray(idx.reconstruct(ipos), dtype="float32")
                                        norms.append(float(np.linalg.norm(v)))
                                    if norms:
                                        st.write({"faiss_top_vector_norm_min": float(min(norms)), "faiss_top_vector_norm_max": float(max(norms))})
                            except Exception:
                                pass

                            # Score distribution sanity: sample random vectors and compare.
                            try:
                                idx = st.session_state.get("_faiss_index")
                                nt = int(getattr(idx, "ntotal", 0) or 0) if idx is not None else 0
                                st.write({"faiss_ntotal": nt})
                            except Exception:
                                pass


                elif mode == "Crop (light)":
                    if not _ELEMENT_OK or _embed_image is None:
                        raise RuntimeError("Element encoder not available")

                    # Prefer using an "armed" crop (cropper bytes) if it matches the current query image.
                    src_path = str(qpath)
                    has_crop = (
                        isinstance(st.session_state.get("_cropper_last_bytes"), (bytes, bytearray))
                        and len(st.session_state.get("_cropper_last_bytes")) > 0
                        and str(st.session_state.get("_cropper_last_source_path", "")) == src_path
                    )
                    try:
                        if not has_crop:
                            # If paths differ only by normalization, try resolved-path equivalence.
                            sp = str(st.session_state.get("_cropper_last_source_path", "") or "")
                            if sp:
                                has_crop = Path(sp).resolve() == Path(src_path).resolve()
                    except Exception:
                        pass

                    saved_crop_path = ""
                    if has_crop:
                        session_id = _ensure_crop_session_id()
                        try:
                            saved_crop_path = _save_query_crop_image(
                                source_path=qpath,
                                cropped_bytes=bytes(st.session_state.get("_cropper_last_bytes")),
                                session_id=session_id,
                            )
                            st.image(saved_crop_path, caption=f"Embedded crop: {Path(saved_crop_path).name}")
                        except Exception as e:
                            st.error(f"Failed to save armed crop: {e}")
                            s.update(label="❌ Search failed", state="error")
                            st.stop()

                    # Embed the saved crop if available; otherwise fall back to slider crop (or whole image).
                    try:
                        if saved_crop_path:
                            qvec = _embed_image(saved_crop_path, crop_box=None)
                        elif crop_box is not None:
                            qvec = _embed_image(qpath, crop_box=crop_box)
                        else:
                            raise RuntimeError("No crop selected: click the cropper's 'Use crop' (or enable manual sliders).")
                    except Exception as e:
                        st.error(f"Crop(light) embedding failed: {e}")
                        s.update(label="❌ Search failed", state="error")
                        st.stop()

                    # Query display context for Step 7
                    st.session_state["_last_query_display_kind"] = "crop_light"
                    st.session_state["_last_crop_light_source_path"] = str(qpath or "")
                    st.session_state["_last_crop_light_crop_box"] = (None if crop_box is None else tuple(int(v) for v in crop_box))
                    st.session_state["_last_crop_light_saved_crop_path"] = str(saved_crop_path or "")
                    st.session_state["_last_query_display_path"] = str(saved_crop_path or "")

                    qvec = np.asarray(qvec, dtype="float32")
                    if qvec.ndim == 1:
                        qvec = qvec.reshape(1, -1)

                    # 4) Dim check (must show st.error, not just raise)
                    faiss_index = st.session_state.get("_faiss_index")
                    if faiss_index is None:
                        raise RuntimeError("FAISS index missing in session")
                    try:
                        faiss_dim = int(getattr(faiss_index, "d"))
                    except Exception:
                        faiss_dim = int(qvec.shape[1])
                    qdim = int(qvec.shape[1])
                    if qdim != faiss_dim:
                        st.error(f"Crop(light) embedding dim mismatch: qvec has dim={qdim} but FAISS index expects dim={faiss_dim}.")
                        raise RuntimeError("Crop(light) qvec dim != FAISS dim")

                    # Optional sanity: compute cosine(full_image, crop_query) and show it prominently.
                    if bool(st.session_state.get("crop_light_run_sanity", False)):
                        try:
                            v_full = _embed_image(qpath, crop_box=None)
                            v_full = np.asarray(v_full, dtype="float32")
                            if v_full.ndim == 1:
                                v_full = v_full.reshape(1, -1)

                            a = v_full.astype(np.float32)
                            b = np.asarray(qvec, dtype=np.float32)
                            if b.ndim == 1:
                                b = b.reshape(1, -1)
                            na = np.linalg.norm(a, axis=1, keepdims=True)
                            nb = np.linalg.norm(b, axis=1, keepdims=True)
                            na[na == 0.0] = 1e-12
                            nb[nb == 0.0] = 1e-12
                            a = a / na
                            b = b / nb
                            cos_fc = float((a * b).sum())
                            st.session_state["_crop_light_last_sanity_cos"] = float(cos_fc)
                            st.info(f"Sanity cosine(full, crop): {cos_fc:0.4f}")
                        except Exception as e:
                            st.session_state["_crop_light_last_sanity_cos"] = None
                            st.warning(f"Sanity check failed: {e}")

                    # 5) Debug/verification expander
                    with st.expander("Crop(light) debug", expanded=False):
                        if crop_box is not None:
                            st.write(f"crop_box: {tuple(int(v) for v in crop_box)}")
                        else:
                            st.write("crop_box: (none; using cropper bytes)")
                        st.write(f"saved_crop_path: {str(saved_crop_path) if saved_crop_path else '(none; used slider crop)'}")
                        st.write(f"saved_crop_source: {str(st.session_state.get('_last_saved_crop_source') or 'n/a')}")
                        st.write(f"qvec_dim: {qdim}")
                        st.write(f"qvec_l2_norm: {float(np.linalg.norm(qvec)):0.6f}")

                        if bool(st.session_state.get("crop_light_run_sanity", False)):
                            if st.session_state.get("_crop_light_last_sanity_cos") is not None:
                                try:
                                    st.write(
                                        f"cosine(full, crop): **{float(st.session_state.get('_crop_light_last_sanity_cos')):0.4f}**"
                                    )
                                except Exception:
                                    pass

                    cos = st.session_state.get("_element_alignment_cos")
                    if cos is None:
                        cos2, msg2 = compute_element_alignment_sample()
                        st.session_state["_element_alignment_cos"] = cos2
                        st.session_state["_element_alignment_msg"] = msg2
                        cos = cos2
                    if cos is None:
                        raise RuntimeError(
                            "Could not verify element-search alignment against this FAISS index. "
                            "Crop(light) results may be meaningless unless the embeddings CSV was generated with SigLIP2 image embeddings. "
                            "Use 'Element search: alignment check' to debug, or rebuild embeddings/FAISS in SigLIP2 space."
                        )
                    if float(cos) < 0.90:
                        raise RuntimeError(
                            f"Element search is not aligned with this FAISS index (sample cosine={float(cos):.3f} < 0.90). "
                            "Crop(light) results will be unreliable. Rebuild embeddings/FAISS using SigLIP2 image embeddings."
                        )
                    df = knn_filtered_by_vector(
                        st.session_state["_faiss_index"],
                        st.session_state["embed_filenames"],
                        qvec,
                        int(k),
                        allowed=allowed,
                        metric=metric_used,
                        key_kind=key_kind,
                    )
                    if "Cosine" in metric_used and not df.empty:
                        df["cosine_similarity"] = df["raw_score"].astype(float)
                        df["cosine_distance"] = df["distance"].astype(float)
                        df["score_0_100"] = (100.0 * df["cosine_similarity"]).clip(0, 100)
                        df["match_quality"] = df["cosine_similarity"].map(lambda v: _match_label(float(v)))

                elif mode == "Crop (patch index)":
                    if not st.session_state.get("patch_index_ready"):
                        raise RuntimeError("Patch index not loaded")
                    assert crop_box is not None

                    patch_index = st.session_state.get("_patch_faiss_index")
                    if patch_index is None:
                        raise RuntimeError("Patch index missing in session")

                    meta_df = st.session_state.get("_patch_meta")
                    if meta_df is None or (hasattr(meta_df, "empty") and meta_df.empty):
                        raise RuntimeError("Patch meta missing in session")

                    # Choose how to form the query vector.
                    # - Snap (fast): use nearest precomputed patch vector for this query image.
                    # - Live (accurate): embed the crop with SigLIP2 at query time.
                    crop_query_mode = st.session_state.get("_crop_patch_query_mode", "Snap to precomputed patch (fast)")
                    crop_query_mode = st.radio(
                        "Patch query vector",
                        ["Snap to precomputed patch (fast)", "Embed crop live with SigLIP2 (slow)"] ,
                        index=0 if str(crop_query_mode).startswith("Snap") else 1,
                        horizontal=True,
                        key="_crop_patch_query_mode",
                        help="Live embedding matches your expectation (crop -> SigLIP2 -> search). Snap mode is faster and uses the nearest precomputed patch embedding.",
                    )

                    qvec = None
                    q_patch_id = None
                    if str(crop_query_mode).startswith("Embed"):
                        if not _ELEMENT_OK or _embed_image is None:
                            raise RuntimeError("Live crop embedding requires the element encoder (transformers+torch).")
                        qvec = _embed_image(qpath, crop_box=crop_box)
                        _assert_query_dim_matches_index(qvec, patch_index, "Crop image encoder")
                        # normalize for cosine/IP
                        if "Cosine" in str(metric_used):
                            qvec = np.asarray(qvec, dtype="float32")
                            if qvec.ndim == 1:
                                qvec = qvec.reshape(1, -1)
                            nrm = np.linalg.norm(qvec, axis=1, keepdims=True)
                            nrm[nrm == 0.0] = 1e-12
                            qvec = qvec / nrm
                        st.caption("Using live SigLIP2 embedding for the crop.")
                    else:
                        # Snap crop to nearest precomputed patch for THIS query image.
                        qkey = _fname_key(Path(qpath).name)
                        if _PATCH_BACKEND_OK and _patch_backend is not None and "filename_key" in getattr(meta_df, "columns", []):
                            pm = _patch_backend.patches_for_filename(meta_df, qkey)
                            snapped = _patch_backend.snap_crop_to_patch(pm, crop_box)
                            q_patch_id = int(snapped["patch_id"])
                        else:
                            # legacy meta schema: filename + x0,y0,x1,y1
                            m = meta_df
                            if "filename" not in m.columns:
                                raise RuntimeError("Legacy patch_meta.csv missing filename")
                            m = m.loc[m["filename"].astype(str).str.lower() == qkey].copy()
                            if m.empty:
                                raise RuntimeError("No patches found for query filename in legacy patch_meta")
                            # treat (x0,y0,x1,y1) as (x1,y1,x2,y2)
                            best_pid = None
                            best_iou = -1.0
                            for r in m.to_dict("records"):
                                pb = (int(r.get("x0", 0)), int(r.get("y0", 0)), int(r.get("x1", 0)), int(r.get("y1", 0)))
                                # compute IoU inline
                                ix1 = max(int(crop_box[0]), int(pb[0])); iy1 = max(int(crop_box[1]), int(pb[1]))
                                ix2 = min(int(crop_box[2]), int(pb[2])); iy2 = min(int(crop_box[3]), int(pb[3]))
                                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                                if inter <= 0:
                                    v = 0.0
                                else:
                                    a = max(0, crop_box[2]-crop_box[0]) * max(0, crop_box[3]-crop_box[1])
                                    b = max(0, pb[2]-pb[0]) * max(0, pb[3]-pb[1])
                                    u = a + b - inter
                                    v = float(inter) / float(u) if u > 0 else 0.0
                                if v > best_iou:
                                    best_iou = v
                                    best_pid = int(r.get("patch_id"))
                            if best_pid is None:
                                raise RuntimeError("Failed to snap crop to a legacy patch")
                            q_patch_id = int(best_pid)

                        # Query vector: prefer FAISS reconstruct() to avoid loading/mmaping huge patch_vectors arrays.
                        vectors = st.session_state.get("_patch_vectors")
                        if vectors is not None:
                            qvec = np.ascontiguousarray(
                                np.asarray(vectors[q_patch_id : q_patch_id + 1], dtype=np.float32),
                                dtype=np.float32,
                            )
                        else:
                            try:
                                # IndexFlat* supports reconstruct(); should work for our patch indices.
                                if hasattr(patch_index, "ntotal") and int(q_patch_id) >= int(patch_index.ntotal):
                                    raise IndexError(f"patch_id {q_patch_id} out of range (ntotal={int(patch_index.ntotal)})")
                                qvec = np.asarray(patch_index.reconstruct(int(q_patch_id)), dtype="float32").reshape(1, -1)
                            except OSError as e:
                                # Windows: 1455 = pagefile too small for this operation.
                                raise RuntimeError(
                                    "Windows ran out of virtual memory while reconstructing patch vectors. "
                                    "Try increasing the Windows paging file (pagefile), reducing patch index size (fewer scales/tiles), "
                                    "or rebuilding a smaller patch index."
                                ) from e
                            except Exception as e:
                                raise RuntimeError(
                                    "Could not obtain patch query vector (no patch_vectors loaded and FAISS reconstruct failed)."
                                ) from e
                        st.caption(f"Snapped to patch_id={q_patch_id} (precomputed embedding).")

                    # Query display context for Step 7
                    st.session_state["_last_query_display_kind"] = "crop_patch"
                    st.session_state["_last_crop_patch_source_path"] = str(qpath or "")
                    st.session_state["_last_crop_patch_crop_box"] = tuple(int(v) for v in crop_box)
                    st.session_state["_last_crop_patch_id"] = (None if q_patch_id is None else int(q_patch_id))
                    st.session_state["_last_query_display_path"] = ""

                    assert qvec is not None

                    # Search more patches, then group by filename_key (or legacy filename)
                    pmeta = meta_df
                    pmeta.columns = [str(c).strip().lower() for c in pmeta.columns]
                    if "patch_id" in pmeta.columns:
                        pmeta = pmeta.set_index("patch_id", drop=False)

                    Dp, Ip = _faiss_search(patch_index, qvec.astype("float32"), min(int(k) * 50, 5000))
                    D0, I0 = Dp[0], Ip[0]

                    best_by_file: Dict[str, Dict[str, Any]] = {}
                    for pos in range(len(I0)):
                        pid = int(I0[pos])
                        if pid < 0:
                            continue
                        if pid not in pmeta.index:
                            continue
                        row = pmeta.loc[pid].to_dict()
                        fn = str(row.get("filename_key", row.get("filename", "")))
                        if not fn:
                            continue
                        if allowed is not None and fn not in allowed:
                            continue

                        raw = float(D0[pos])
                        if (fn not in best_by_file) or (raw > float(best_by_file[fn]["raw_score"])):
                            best_by_file[fn] = {
                                "filename": fn,
                                "raw_score": raw,
                                "x0": int(row.get("x1", row.get("x0", 0))),
                                "y0": int(row.get("y1", row.get("y0", 0))),
                                "x1": int(row.get("x2", row.get("x1", 0))),
                                "y1": int(row.get("y2", row.get("y1", 0))),
                                "patch_id": int(pid),
                            }
                        if len(best_by_file) >= int(k) and pos > int(k) * 10:
                            # early stop once we have enough unique files and are deep into results
                            pass

                    rows = list(best_by_file.values())
                    rows.sort(key=lambda r: float(r.get("raw_score", -1e9)), reverse=True)
                    rows = rows[: int(k)]
                    out = []
                    for i, r in enumerate(rows, start=1):
                        raw = float(r.get("raw_score"))
                        distance = (1.0 - raw) if "Cosine" in st.session_state["index_metric"] else raw
                        out.append({
                            "rank": i,
                            "filename": r.get("filename"),
                            "distance": distance,
                            "raw_score": raw,
                            "patch_id": int(r.get("patch_id")),
                            "x0": int(r.get("x0")),
                            "y0": int(r.get("y0")),
                            "x1": int(r.get("x1")),
                            "y1": int(r.get("y1")),
                        })
                    df = pd.DataFrame(out)
                    if "Cosine" in st.session_state["index_metric"] and not df.empty:
                        df["cosine_similarity"] = df["raw_score"].astype(float)
                        df["cosine_distance"] = df["distance"].astype(float)
                        df["score_0_100"] = (100.0 * df["cosine_similarity"]).clip(0, 100)
                        df["match_quality"] = df["cosine_similarity"].map(lambda v: _match_label(float(v)))

                else:
                    raise RuntimeError(f"Unknown query mode: {mode}")

                if meta_df is not None and st.session_state.get("result_meta_cols") and not df.empty:
                    meta_df = _ensure_stable_id(_ensure_filename(meta_df.copy()))
                    join_col = "filename"
                    if (
                        str(st.session_state.get("_faiss_key_kind", "filename")) == "stable_id"
                        and "stable_id" in meta_df.columns
                        and ("stable_id" in df.columns or ("filename" in df.columns and "stable_id" not in df.columns))
                    ):
                        # Ensure df has a stable_id column when FAISS keys are stable_id.
                        if "stable_id" not in df.columns and "filename" in df.columns:
                            df = df.copy()
                            df["stable_id"] = df["filename"].astype(str)
                        join_col = "stable_id"

                    keep_meta = [join_col] + [c for c in st.session_state["result_meta_cols"] if c in meta_df.columns]
                    # Keep filename for display if metadata has it.
                    if join_col == "stable_id" and "filename" in meta_df.columns and "filename" not in keep_meta:
                        keep_meta.append("filename")

                    df = df.merge(meta_df[keep_meta], on=join_col, how="left")
                    if join_col == "stable_id":
                        # After the merge, keep stable_id for lookups and prefer metadata filename for display.
                        if "filename_y" in df.columns:
                            df["filename"] = df["filename_y"].astype(str)
                        if "filename_x" in df.columns:
                            df = df.drop(columns=["filename_x"])
                        if "filename_y" in df.columns:
                            df = df.drop(columns=["filename_y"])

                    # Attach full paths from Step 4 index so thumbnails always resolve.
                    try:
                        idx_df2 = st.session_state.get("index_df", pd.DataFrame())
                        idx_key_kind = str(st.session_state.get("_index_key_kind", "filename"))
                        df = attach_paths_from_index(df, idx_df2, idx_key_kind)
                    except Exception:
                        pass

                st.session_state["last_results_df"] = df
                s.update(label=f"✅ Done ({len(df):,} results)", state="complete")
                st.session_state["_last_search_ok"] = True
            except Exception as e:
                if isinstance(e, OSError) and getattr(e, "winerror", None) == 1455:
                    st.error(
                        "Windows paging file is too small for this operation (OSError 1455). "
                        "Increase the Windows pagefile, or reduce index size / K / filters."
                    )
                    st.caption(repr(e))
                else:
                    st.error(str(e))
                s.update(label="❌ Search failed", state="error")

    # Always render the last results so diagnostic buttons work on reruns.
    fm = st.session_state.get("filemap", {}) or {}
    df = st.session_state.get("last_results_df", pd.DataFrame())
    if df.empty and st.session_state.get("_last_search_ok", True):
        st.warning("No results (try fewer filters or a higher K).")
    elif not df.empty:
        # Show mode-relevant score columns (but keep full df in session_state).
        last_mode = str(st.session_state.get("_last_search_mode", "") or "")
        df_show = df.copy()
        if last_mode == "Text":
            # Hide image-to-image cosmetics in text mode.
            drop_cols = [
                c
                for c in (
                    "match_quality",
                    "score_0_100",
                    "cosine_similarity",
                    "cosine_distance",
                    "raw_score",
                    "distance",
                )
                if c in df_show.columns
            ]
            if drop_cols:
                df_show = df_show.drop(columns=drop_cols)
        else:
            # Hide text-only score columns in image/crop modes.
            text_cols = [c for c in df_show.columns if str(c).startswith("text_")]
            if text_cols:
                df_show = df_show.drop(columns=text_cols)

        st_dataframe_stretch(df_show)
        st.download_button(
            "⬇️ Download results (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="retriever_results.csv",
            mime="text/csv",
        )

        if (
            str(st.session_state.get("_last_search_mode", "")) == "Text"
            and st.session_state.get("faiss_ready")
            and st.session_state.get("_last_text_qvec")
        ):
            with st.expander("Text diagnostics (post-search)", expanded=False):
                st.write(
                    {
                        "text": str(st.session_state.get("_last_text_query", "")),
                        "prompts_used": st.session_state.get("_last_text_prompts_used", []),
                        "metric_used": str(st.session_state.get("_last_text_metric_used", "")),
                    }
                )

                qv = np.asarray(st.session_state.get("_last_text_qvec"), dtype="float32").reshape(1, -1)
                qv = qv / max(float(np.linalg.norm(qv)), 1e-12)

                st.session_state.setdefault("_text_score_dist_result", None)
                st.session_state.setdefault("_text_crossmodal_rows", None)

                cA, cB = st.columns(2)
                if cA.button("Analyze text score distribution", key="post_text_score_dist_btn"):
                    try:
                        idx = st.session_state.get("_faiss_index")
                        nt = int(getattr(idx, "ntotal", 0) or 0) if idx is not None else 0
                        if nt <= 0 or idx is None or not hasattr(idx, "reconstruct"):
                            raise RuntimeError("FAISS index missing or empty")

                        sample_n = int(min(2000, nt))
                        rng = np.random.default_rng(0)
                        positions = rng.integers(low=0, high=nt, size=sample_n, dtype=np.int64)
                        sims = []
                        for p in positions.tolist():
                            v = np.asarray(idx.reconstruct(int(p)), dtype="float32")
                            sims.append(float(np.dot(qv.ravel(), v.ravel())))
                        sims_arr = np.asarray(sims, dtype="float32")

                        scores = df["raw_score"].astype(float)
                        top = float(scores.max())
                        pct = float((sims_arr <= top).mean() * 100.0)
                        st.session_state["_text_score_dist_result"] = {
                            "faiss_ntotal": int(nt),
                            "sample_n": int(sample_n),
                            "sample_mean": float(sims_arr.mean()),
                            "sample_std": float(sims_arr.std()),
                            "sample_p95": float(np.quantile(sims_arr, 0.95)),
                            "sample_p99": float(np.quantile(sims_arr, 0.99)),
                            "top_score": float(top),
                            "top_score_percentile": float(pct),
                        }
                    except Exception as e:
                        st.session_state["_text_score_dist_result"] = {"error": str(e)}

                if cB.button("Analyze cross-modal sanity (re-embed top images)", key="post_text_crossmodal_btn"):
                    try:
                        if not _ELEMENT_OK or _embed_image is None:
                            raise RuntimeError("Element encoder not available")

                        idx_df = st.session_state.get("index_df", pd.DataFrame())
                        sid_map = build_sid_map(idx_df) if str(st.session_state.get("_index_key_kind", "filename")) == "stable_id" else {}

                        idx = st.session_state.get("_faiss_index")
                        pos_map = st.session_state.get("_faiss_pos_map") or {}
                        fns = st.session_state.get("embed_filenames") or []
                        key_kind = str(st.session_state.get("_faiss_key_kind", "filename"))

                        def _ipos_lookup(lookup_key: str) -> tuple[int | None, str]:
                            """Try multiple key variants to locate the FAISS position."""
                            lk = str(lookup_key)
                            candidates: list[str] = []
                            # direct
                            candidates.append(lk)
                            candidates.append(lk.strip())
                            # filename-normalized
                            candidates.append(_fname_key(lk))
                            try:
                                candidates.append(_fname_key(Path(lk).name))
                            except Exception:
                                pass
                            # de-dupe
                            seen: set[str] = set()
                            candidates2 = []
                            for c in candidates:
                                c2 = str(c)
                                if c2 and c2 not in seen:
                                    candidates2.append(c2)
                                    seen.add(c2)

                            for c in candidates2:
                                if pos_map and c in pos_map:
                                    return int(pos_map[c]), f"pos_map[{c}]"
                            for c in candidates2:
                                try:
                                    return int(fns.index(c)), f"embed_filenames.index({c})"
                                except Exception:
                                    continue
                            return None, "not_found"

                        out_rows = []
                        for rr in df.head(5).to_dict("records"):
                            fn = str(rr.get("filename", ""))
                            sid = str(rr.get("stable_id") or "")
                            lookup = sid or fn

                            p = fm.get(str(lookup)) or fm.get(fn) or fm.get(_fname_key(fn)) or fm.get(Path(fn).name) or fm.get(_fname_key(Path(fn).name)) or ""
                            if not p and sid_map and str(key_kind) != "stable_id":
                                sid2 = sid_map.get(_fname_key(fn), "")
                                if sid2:
                                    p = fm.get(str(sid2), "")

                            path_ok = bool(p and Path(p).exists())

                            # Resolve FAISS position; be robust to key-kind mismatches.
                            ipos, ipos_src = _ipos_lookup(lookup)
                            if ipos is None and fn and fn != lookup:
                                ipos, ipos_src = _ipos_lookup(fn)

                            cos_text_faiss = float("nan")
                            cos_text_img = float("nan")
                            cos_faiss_img = float("nan")
                            if idx is not None and hasattr(idx, "reconstruct") and ipos is not None:
                                v_faiss = np.asarray(idx.reconstruct(int(ipos)), dtype="float32").reshape(1, -1)
                                cos_text_faiss = float(np.dot(qv.ravel(), v_faiss.ravel()))
                                if path_ok:
                                    v_img = np.asarray(_embed_image(str(p), crop_box=None), dtype="float32")
                                    if v_img.ndim == 1:
                                        v_img = v_img.reshape(1, -1)
                                    cos_text_img = float(np.dot(qv.ravel(), v_img.ravel()))
                                    cos_faiss_img = float(np.dot(v_faiss.ravel(), v_img.ravel()))

                            out_rows.append(
                                {
                                    "rank": int(rr.get("rank", len(out_rows) + 1)),
                                    "key": str(lookup),
                                    "ipos": (-1 if ipos is None else int(ipos)),
                                    "ipos_src": str(ipos_src),
                                    "path_ok": bool(path_ok),
                                    "path": ("" if not p else str(p)),
                                    "raw_score": float(rr.get("raw_score", float("nan"))),
                                    "text·faiss_vec": float(cos_text_faiss),
                                    "text·reembed_img": float(cos_text_img),
                                    "faiss_vec·reembed_img": float(cos_faiss_img),
                                }
                            )

                        st.session_state["_text_crossmodal_rows"] = out_rows
                    except Exception as e:
                        st.session_state["_text_crossmodal_rows"] = [{"error": str(e)}]

                if st.session_state.get("_text_score_dist_result"):
                    st.write(st.session_state.get("_text_score_dist_result"))
                if st.session_state.get("_text_crossmodal_rows"):
                    st.dataframe(pd.DataFrame(st.session_state.get("_text_crossmodal_rows")), use_container_width=True)

                st.divider()
                st.markdown("**Text probe (known image)**")
                st.session_state.setdefault("_text_probe_path", "")
                st.session_state.setdefault("_text_probe_topn", 2000)
                probe_path = st.text_input(
                    "Probe image path",
                    key="_text_probe_path",
                    help="Paste the full path to an image you believe should match the current text query.",
                )
                topn = st.number_input(
                    "Probe top-N",
                    min_value=10,
                    max_value=20000,
                    value=int(st.session_state.get("_text_probe_topn", 2000)),
                    step=10,
                )
                st.session_state["_text_probe_topn"] = int(topn)

                if st.button("Run probe", key="run_text_probe"):
                    try:
                        if not _ELEMENT_OK or _embed_image is None:
                            raise RuntimeError("Element encoder not available")
                        if not probe_path or not Path(probe_path).exists():
                            raise RuntimeError("Probe path does not exist")

                        v_img = np.asarray(_embed_image(str(probe_path), crop_box=None), dtype="float32")
                        if v_img.ndim == 1:
                            v_img = v_img.reshape(1, -1)
                        cos_t_img = float(np.dot(qv.ravel(), v_img.ravel()))

                        idx = st.session_state.get("_faiss_index")
                        fns = st.session_state.get("embed_filenames") or []
                        key_kind = str(st.session_state.get("_faiss_key_kind", "filename"))
                        probe_key = str(Path(probe_path).name).strip() if key_kind == "stable_id" else _fname_key(Path(probe_path).name)

                        rank_found = None
                        if idx is not None:
                            Dp, Ip = _faiss_search(idx, qv.astype("float32"), int(topn))
                            I0 = Ip[0].tolist()
                            for rnk, ii in enumerate(I0, start=1):
                                if 0 <= int(ii) < len(fns):
                                    if str(fns[int(ii)]) == str(probe_key):
                                        rank_found = int(rnk)
                                        break

                        st.write(
                            {
                                "probe_path": str(probe_path),
                                "probe_key": str(probe_key),
                                "text·probe_image": float(cos_t_img),
                                "rank_in_topN": (None if rank_found is None else int(rank_found)),
                                "topN": int(topn),
                            }
                        )
                    except Exception as e:
                        st.warning(f"Probe failed: {e}")

        st.markdown("**Thumbnails**")
        cols = st.columns(5)
        index_key_kind = str(st.session_state.get("_index_key_kind", "filename"))
        sid_map = build_sid_map(st.session_state.get("index_df", pd.DataFrame())) if index_key_kind == "stable_id" else {}
        for i, r in enumerate(df.to_dict("records")):
            display_name = r.get("filename", "")
            path = str(r.get("full_path") or "")
            if not (path and Path(path).exists()):
                if index_key_kind == "stable_id":
                    sid = str(r.get("stable_id") or "").strip()
                    if not sid and display_name:
                        sid = sid_map.get(_fname_key(str(display_name)), "")
                    lookup_key = sid or ""
                else:
                    lookup_key = _fname_key(str(display_name)) if display_name else ""
                path = fm.get(str(lookup_key), "") if lookup_key else ""
            cap = f"{i+1}. {display_name}"
            last_mode = str(st.session_state.get("_last_search_mode", "") or "")
            if last_mode == "Text":
                try:
                    stars = str(r.get("text_stars", ""))
                    if stars:
                        cap += f"\n{stars}"

                    metric_tag = str(r.get("text_metric", "") or "")
                    if metric_tag == "cosine" and ("text_score" in df.columns or "raw_score" in df.columns):
                        cos = float(r.get("text_score", r.get("raw_score", float("nan"))))
                        ang = float(r.get("text_angle_deg", float("nan")))
                        gap = r.get("text_gap_to_next", None)
                        cap += f"\ncos={cos:0.3f}"
                        if np.isfinite(ang):
                            cap += f" | ang={ang:0.1f}°"
                        if gap is not None and pd.notna(gap):
                            cap += f" | gap={float(gap):0.3f}"
                    elif metric_tag == "l2" and ("text_score" in df.columns or "distance" in df.columns):
                        d = float(r.get("text_score", r.get("distance", float("nan"))))
                        gap = r.get("text_gap_to_next", None)
                        cap += f"\nl2={d:0.3f}"
                        if gap is not None and pd.notna(gap):
                            cap += f" | gap={float(gap):0.3f}"
                    else:
                        # Fallback: show raw score.
                        rs = float(r.get("raw_score", float("nan")))
                        if np.isfinite(rs):
                            cap += f"\nscore={rs:0.3f}"
                except Exception:
                    pass
            else:
                if "match_quality" in df.columns:
                    cap += f"\n{r.get('match_quality', '')}"
            with cols[i % 5]:
                if path and Path(path).exists():
                    if mode == "Crop (patch index)" and all(k in r for k in ("x0", "y0", "x1", "y1")):
                        try:
                            img = _draw_boxes_on_image(path, [(r.get("x0"), r.get("y0"), r.get("x1"), r.get("y1"))])
                            st.image(img, caption=cap, width=img_w)
                        except Exception:
                            display_image(path, width=img_w, caption=cap)
                    else:
                        display_image(path, width=img_w, caption=cap)
                else:
                    st.write(f"❓ {cap}")

    cL, cR = st.columns(2)
    cL.button("⬅️ Back", on_click=lambda: st.session_state.update(step=5), key="back_step_6")
    cR.button("➡️ Next (Stacked view)", on_click=lambda: st.session_state.update(step=7),
              disabled=st.session_state.get("last_results_df", pd.DataFrame()).empty, key="next_step_6")


# ========== Step 7: Stacked view ==========
elif st.session_state["step"] == 7:
    st.markdown("**Step 7: Stacked view**")
    st.info(
        "A reading-friendly view of the results: images are shown in order with optional metadata next to each one. "
    )
    df = st.session_state.get("last_results_df", pd.DataFrame()).copy()
    fm = st.session_state.get("filemap", {}) or {}
    last_mode = str(st.session_state.get("_last_search_mode", "") or "")
    qpath = str(st.session_state.get("_last_query_image_path") or st.session_state.get("query_image_path", ""))
    if df.empty:
        st.warning("No results in session_state. Go back to Step 6.")
    else:
        c0, c1, c2, c3 = st.columns(4)
        hide_meta = c0.checkbox("Hide metadata", value=False)
        show_query = c1.checkbox("Show query at top", value=True)
        top_n = c2.number_input("How many neighbors", 1, int(max(1, len(df))), min(30, int(len(df))))
        img_w = c3.slider("Image width", 160, 1600, 520, 20)

        meta_cols = [c for c in (st.session_state.get("result_meta_cols") or []) if c in df.columns]

        if "rank" in df.columns:
            df = df.sort_values("rank", ascending=True).head(int(top_n))
        elif "distance" in df.columns:
            df = df.sort_values("distance", ascending=True).head(int(top_n))
        else:
            df = df.head(int(top_n))

        def show_meta_block(row: dict):
            if not meta_cols:
                st.caption("No metadata columns selected.")
                return
            lines = []
            for c in meta_cols:
                v = row.get(c, "—")
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    v = "—"
                v = str(v)[:1000]
                lines.append(f"- **{c}**: {v}")
            st.markdown("\n".join(lines))

        query_meta_row: Optional[dict] = None
        # Only attempt query-image metadata lookup for image/crop modes (Text has no query image).
        if last_mode != "Text" and qpath and Path(qpath).exists() and meta_cols:
            try:
                qmeta = load_metadata(st.session_state["meta_path"]) if st.session_state.get("meta_ok") else None
                if qmeta is not None and not qmeta.empty:
                    qmeta = _ensure_stable_id(_ensure_filename(qmeta))
                    qk = st.session_state.get("query_key") or _fname_key(Path(qpath).name)
                    if "stable_id" in qmeta.columns and str(qk).strip() and len(str(qk).strip()) == 16:
                        hit = qmeta.loc[qmeta["stable_id"].astype(str).str.strip() == str(qk).strip()]
                    elif "filename" in qmeta.columns:
                        hit = qmeta.loc[qmeta["filename"].astype(str) == _fname_key(Path(qpath).name)]
                    else:
                        hit = pd.DataFrame()
                    if not hit.empty:
                        query_meta_row = hit.iloc[0].to_dict()
            except Exception:
                query_meta_row = None

        if show_query:
            if last_mode == "Text":
                st.subheader("Query (Text)")
                txt = str(st.session_state.get("_last_text_query", "") or "").strip()
                prompts = st.session_state.get("_last_text_prompts_used", []) or []
                metric_used = str(st.session_state.get("_last_text_metric_used", "") or "")
                if txt:
                    st.code(txt)
                if metric_used:
                    st.caption(f"metric_used: {metric_used}")
                if prompts:
                    with st.expander("Prompts used", expanded=False):
                        st.write(prompts)
            elif last_mode in ("Crop (light)", "Crop (patch index)"):
                st.subheader(f"Query ({last_mode})")
                # Prefer an explicit saved crop (Crop(light) with cropper bytes)
                crop_path = ""
                crop_box = None
                if last_mode == "Crop (light)":
                    crop_path = str(st.session_state.get("_last_crop_light_saved_crop_path") or "")
                    crop_box = st.session_state.get("_last_crop_light_crop_box")
                else:
                    crop_box = st.session_state.get("_last_crop_patch_crop_box")
                    pid = st.session_state.get("_last_crop_patch_id")
                    if pid is not None:
                        st.caption(f"snapped patch_id: {int(pid)}")

                if crop_path and Path(crop_path).exists():
                    display_image(crop_path, width=img_w, caption=Path(crop_path).name)
                elif qpath and Path(qpath).exists() and crop_box is not None:
                    try:
                        from PIL import Image

                        img = Image.open(qpath).convert("RGB")
                        x0, y0, x1, y1 = [int(v) for v in crop_box]
                        cropped = img.crop((x0, y0, x1, y1))
                        st.image(cropped, caption=f"Crop from: {Path(qpath).name}")
                    except Exception:
                        # Fallback: show the source image if cropping fails.
                        display_image(qpath, width=img_w, caption=Path(qpath).name)

                # Optional: show metadata for the source image (crop itself won't exist in metadata).
                if not hide_meta:
                    if query_meta_row is not None:
                        show_meta_block(query_meta_row)
                    else:
                        st.caption("No metadata found for the source image.")
            else:
                # Whole-image and any other image-based mode
                if qpath and Path(qpath).exists():
                    st.subheader("Query image")
                    if hide_meta:
                        display_image(qpath, width=img_w, caption=Path(qpath).name)
                    else:
                        colL, colR = st.columns([1, 2])
                        with colL:
                            display_image(qpath, width=img_w, caption=Path(qpath).name)
                        with colR:
                            if query_meta_row is not None:
                                show_meta_block(query_meta_row)
                            else:
                                st.caption("No metadata found for query image.")

        st.subheader("Nearest neighbors")
        if hide_meta:
            grid_cols = st.slider("Grid columns", 1, 8, 3)
            recs = df.to_dict("records")
            for i in range(0, len(recs), grid_cols):
                row = recs[i : i + grid_cols]
                cols = st.columns(len(row))
                for j, (c, r) in enumerate(zip(cols, row), start=1):
                    fn = r.get("filename", "")
                    sid = r.get("stable_id")
                    lookup = sid or fn
                    cap = f"#{i+j} — {fn}"
                    if "match_quality" in df.columns and "cosine_similarity" in df.columns:
                        try:
                            cap += f"\n{r.get('match_quality','')} (cos={float(r.get('cosine_similarity')):.3f})"
                        except Exception:
                            cap += f"\n{r.get('match_quality','')}"
                    with c:
                        p = ""
                        if fm:
                            p = fm.get(str(lookup)) or fm.get(fn) or fm.get(_fname_key(fn)) or fm.get(Path(fn).name) or fm.get(_fname_key(Path(fn).name)) or ""
                        if p and Path(p).exists():
                            display_image(p, width=img_w, caption=cap)
                        else:
                            st.write(f"{cap}\n(path not found)")
        else:
            for i, r in enumerate(df.to_dict("records"), start=1):
                fn = r.get("filename", "")
                sid = r.get("stable_id")
                lookup = sid or fn
                cap = f"#{i} — {fn}"
                if "match_quality" in df.columns and "cosine_similarity" in df.columns:
                    try:
                        cap += f"\n{r.get('match_quality','')} (cos={float(r.get('cosine_similarity')):.3f})"
                    except Exception:
                        cap += f"\n{r.get('match_quality','')}"
                colL, colR = st.columns([1, 2])
                with colL:
                    p = ""
                    if fm:
                        p = fm.get(str(lookup)) or fm.get(fn) or fm.get(_fname_key(fn)) or fm.get(Path(fn).name) or fm.get(_fname_key(Path(fn).name)) or ""
                    if p and Path(p).exists():
                        display_image(p, width=img_w, caption=cap)
                    else:
                        st.write(f"{cap}\n(path not found)")
                with colR:
                    show_meta_block(r)

        st.download_button(
            "⬇️ Download stacked view (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="stacked_view.csv",
            mime="text/csv",
        )

    cL, cR = st.columns(2)

    def _back_to_step6_preserve_mode() -> None:
        # Streamlit widgets *should* preserve `query_mode`, but in practice users have reported
        # it snapping back to Whole-image when navigating back from Step 7.
        lm = str(st.session_state.get("_last_search_mode", "") or "")
        if lm in {"Whole-image", "Text", "Crop (light)", "Crop (patch index)"}:
            st.session_state["query_mode"] = lm
        st.session_state["step"] = 6

    cL.button("⬅️ Back", on_click=_back_to_step6_preserve_mode, key="back_step_7")
    cR.button("➡️ Next (Annotate)", on_click=lambda: st.session_state.update(step=8),
              disabled=st.session_state.get("last_results_df", pd.DataFrame()).empty, key="next_step_7")


# ========== Step 8: Annotate ==========
elif st.session_state["step"] == 8:
    st.markdown("**Step 8: Annotate (overlay)**")
    st.info(
        "Add your own categories/labels to images (annotations). "
        "These annotations are saved as an overlay in output_dir/_overlay/<session> and never overwrite your original metadata."
    )

    df_res = st.session_state.get("last_results_df", pd.DataFrame()).copy()
    fm = st.session_state.get("filemap", {}) or {}
    idx_df = st.session_state.get("index_df", pd.DataFrame()).copy()
    idx_df = _std_cols(idx_df) if not idx_df.empty else idx_df

    out_dir = str(st.session_state.get("output_dir") or ".")

    # Session selector
    c0, c1 = st.columns([2, 1])
    session = str(c0.text_input("Session", value=st.session_state.get("overlay_session", "iconography_2025q4")) or "")
    st.session_state["overlay_session"] = session

    # Load schema for this session
    schema = overlay_load_schema(out_dir, session)
    fields = schema.get("fields", {}) if isinstance(schema, dict) else {}

    # Fields management
    with st.expander("Fields (schema.json)"):
        st.write(
            "Define annotation fields (new columns) for this session. "
            "Each field is saved under a stable key (the overlay 'marker') and can be exported back into a merged CSV."
        )
        st.caption(
            "**Field title** is the human-friendly label shown in the UI. "
            "**Field key** is the stable identifier used in overlay storage and in exported column names. "
            "Tip: keep keys short, lowercase, and without spaces (e.g. `humans_count`, `style_period`)."
        )
        with st.form("add_field_form", clear_on_submit=True):
            t_col1, t_col2 = st.columns([3, 1])
            title = t_col1.text_input("Field title (shown in UI)")
            key_suggest = _field_key_from_title(title or "")
            key = t_col2.text_input("Field key (used for export/storage)", value=key_suggest)

            ftype = st.selectbox("Type", ["Category", "Number"], index=0)
            if str(ftype).lower().startswith("number"):
                num_kind = st.selectbox("Number kind", ["int", "float"], index=0)
                allow_blank = st.checkbox("Allow blank/clear", value=True)
                kind = "single"
                vals = ""
                allow_custom = False
            else:
                kind = st.selectbox("Kind", ["single", "multi"], index=0)
                vals = st.text_input("Allowed values (comma-separated)")
                allow_custom = st.checkbox("Allow custom values (user can type)", value=False)
                allow_blank = st.checkbox("Allow blank/clear (only for single)", value=False)
                num_kind = None

            # Export preview for end users
            preview_key = (key or key_suggest or "").strip() or "<key>"
            if str(ftype).lower().startswith("number"):
                st.caption(f"Export preview: creates a numeric column named **{preview_key}**")
            else:
                if str(kind) == "single":
                    st.caption(f"Export preview: creates a single column named **{preview_key}**")
                else:
                    st.caption(f"Export preview: creates a list column named **{preview_key}_list**")
            st.caption(
                "Changing a field key later is like renaming a column: existing saved annotations keep the old key. "
                "If you need a new key, add a new field and (optionally) delete the old one."
            )

            submitted = st.form_submit_button("Add / Update field")
            if submitted:
                key = (key or key_suggest or "").strip()
                if not key:
                    st.error("Field key is required.")
                else:
                    values_list = [v.strip() for v in vals.split(",") if v.strip()] if vals else []
                    fields.setdefault(key, {})
                    fields[key]["title"] = title or key

                    if str(ftype).lower().startswith("number"):
                        fields[key]["type"] = "number"
                        fields[key]["kind"] = "single"
                        fields[key]["num_kind"] = str(num_kind or "int")
                        fields[key]["values"] = []
                        fields[key]["allow_custom"] = False
                        fields[key]["allow_blank"] = bool(allow_blank)
                    else:
                        fields[key]["type"] = "category"
                        fields[key]["kind"] = kind
                        fields[key]["values"] = values_list
                        fields[key]["allow_custom"] = bool(allow_custom)
                        fields[key]["allow_blank"] = bool(allow_blank) if kind == "single" else False

                    schema["fields"] = fields
                    overlay_save_schema(out_dir, session, schema)
                    st.success(f"Saved field: {key}")
                    try:
                        safe_rerun()
                    except Exception:
                        # compatibility: older/newer streamlit may not expose rerun
                        pass

        st.markdown("---")
        if fields:
            st.write("Existing fields:")
            for fk, spec in list(fields.items()):
                col_a, col_b, col_c = st.columns([4, 1, 1])
                _t = str(spec.get("type") or "category")
                _k = str(spec.get("kind") or "")
                if _t.lower() == "number":
                    _k = "number"
                col_a.write(f"**{spec.get('title', fk)}** (`{fk}`) — {_k}")

                # persistent confirmation checkbox (rendered always, stored in session_state)
                confirm_key = f"del_confirm_{fk}"
                # Place the checkbox in the third column so it's visible alongside the Delete button
                confirm = col_c.checkbox(f"Also remove existing annotations for `{fk}`?", key=confirm_key, value=False)

                if col_b.button("Delete", key=f"del_field_{fk}"):
                    # delete field from schema
                    if st.session_state.get(confirm_key):
                        cur = overlay_load_current(out_dir, session)
                        if not cur.empty:
                            cur = _std_cols(cur)
                            mask = cur["marker"].astype(str) == str(fk)
                            rows = cur.loc[mask].to_dict("records")
                            cur = cur.loc[~mask].copy()
                            cur.to_csv(overlay_current_path(out_dir, session), index=False, encoding="utf-8")
                            for r in rows:
                                overlay_append_log(out_dir, session, "remove", r)

                    fields.pop(fk, None)
                    schema["fields"] = fields
                    overlay_save_schema(out_dir, session, schema)
                    st.success(f"Deleted field: {fk}")
                    try:
                        safe_rerun()
                    except Exception:
                        pass
        else:
            st.info("No fields defined yet. Add fields above to start annotating.")

    # If no fields, prompt user to define
    if not fields:
        st.warning("No annotation fields defined for this session. Use 'Fields (schema.json)' to add one.")
        cL, cR = st.columns(2)
        cL.button("⬅️ Back", on_click=lambda: st.session_state.update(step=7), key="back_step_8")
        cR.button("➡️ Next (Projection)", on_click=lambda: st.session_state.update(step=9), key="next_step_8")
        st.stop()

    # Select active field to annotate
    field_keys = list(fields.keys())
    def fmt_fk(k):
        spec = fields.get(k, {})
        return f"{spec.get('title', k)} ({k})"

    # Wire overlay_marker to the Step 8 selected field so it's meaningful:
    # - Default selection uses st.session_state['overlay_marker'] when it exists in the schema
    # - Any user change updates st.session_state['overlay_marker']
    if "s8_selected_field" not in st.session_state:
        preferred = st.session_state.get("overlay_marker")
        if preferred in field_keys:
            st.session_state["s8_selected_field"] = preferred
        else:
            st.session_state["s8_selected_field"] = field_keys[0]

    sel_key = st.selectbox("Field to annotate", options=field_keys, format_func=fmt_fk, key="s8_selected_field")
    st.session_state["overlay_marker"] = sel_key
    field = fields.get(sel_key, {})
    marker = str(sel_key)

    # Controls for annotation UI
    cA, cB, cC = st.columns(3)
    top_n = cA.number_input("Top N", 1, int(max(1, len(df_res))), min(50, int(len(df_res))), key="s8_topn")
    img_w = cB.slider("Image width", 160, 1200, 420, 20, key="s8_imgw")
    cols_n = cC.slider("Columns", 1, 6, 3, key="s8_cols_n")

    custom_unlock = st.checkbox(
        "Unlock custom values (this session)",
        value=bool(st.session_state.get("s8_unlock_custom", False)),
        key="s8_unlock_custom",
        help="When locked, only allowed values can be used for category fields.",
    )

    df_show = df_res.sort_values("rank", ascending=True).head(int(top_n)) if "rank" in df_res.columns else df_res.head(int(top_n))

    # Prefer stable_id as the selection key if we have it (avoids filename collisions).
    ann_key_kind = "stable_id" if ("stable_id" in df_show.columns and df_show["stable_id"].astype(str).str.strip().ne("").any()) else ("stable_id" if ("stable_id" in idx_df.columns and idx_df["stable_id"].astype(str).str.strip().ne("").any()) else "filename")
    sid_to_filename = st.session_state.get("_sid_to_filename", {}) or {}

    if ann_key_kind == "stable_id":
        if "stable_id" in df_show.columns:
            page_keys = [str(x).strip() for x in df_show["stable_id"].astype(str).tolist() if str(x).strip()]
        else:
            sid_map = build_sid_map(idx_df)
            page_keys = []
            for fn in df_show["filename"].astype(str).tolist():
                sid = sid_map.get(_fname_key(fn), "")
                if str(sid).strip():
                    page_keys.append(str(sid).strip())
    else:
        page_keys = [str(x) for x in df_show["filename"].astype(str).tolist()]

    st.markdown("---")
    st.subheader("Batch tools")
    sel_all = st.checkbox("Select all on page", value=False, key="s8_sel_all")
    # synchronize the session_state-backed multiselect when Select all is toggled
    # If the user checks Select all, set the session_state value to the current page keys.
    # If the user unchecks Select all and their current selection exactly matches page_keys, clear it.
    if sel_all:
        if st.session_state.get("s8_selected_list") != page_keys:
            st.session_state["s8_selected_list"] = page_keys
    else:
        cur_sel = st.session_state.get("s8_selected_list", [])
        try:
            if set(cur_sel) == set(page_keys):
                st.session_state["s8_selected_list"] = []
        except Exception:
            # fallback: if comparison fails, don't modify user's selection
            pass
    def _fmt_ann_key(x: str) -> str:
        x = str(x)
        if ann_key_kind == "stable_id":
            nm = sid_to_filename.get(x, "")
            return f"{nm} ({x})" if nm else x
        return x

    try:
        selected = st.multiselect("Selected images", options=page_keys, key="s8_selected_list", format_func=_fmt_ann_key)
    except TypeError:
        selected = st.multiselect("Selected images", options=page_keys, key="s8_selected_list")

    # value input (depends on field)
    field_type = str(field.get("type") or "category").strip().lower()
    allowed_vals = field.get("values", []) or []
    allow_custom = bool(field.get("allow_custom", False))
    kind = field.get("kind", "multi")
    allow_blank = bool(field.get("allow_blank", False))
    num_kind = str(field.get("num_kind") or "int").strip().lower()

    batch_value = None
    if field_type == "number":
        cN1, cN2 = st.columns([2, 1])
        do_clear = cN2.checkbox("Clear value", value=False, key="s8_num_clear")
        if num_kind == "float":
            v = cN1.number_input("Number", value=float(st.session_state.get("s8_num_value", 0.0)), step=0.1, key="s8_num_value")
        else:
            v = cN1.number_input("Number", value=int(st.session_state.get("s8_num_value_int", 0)), step=1, key="s8_num_value_int")
        batch_value = "" if do_clear else str(v)
    else:
        if allowed_vals:
            if kind == "single" and allow_blank:
                opts = ["(clear)"] + allowed_vals
            else:
                opts = allowed_vals
            batch_sel = st.selectbox("Select value", options=opts, index=0 if opts else 0, key="s8_batch_sel")
            batch_value = "" if batch_sel == "(clear)" else batch_sel
            if allow_custom and custom_unlock:
                custom = st.text_input("Custom value (overrides selection)", key="s8_batch_custom")
                if custom and custom.strip():
                    batch_value = custom.strip()
        else:
            # free text
            batch_value = st.text_input("Value", key="s8_batch_value")

    cX, cY = st.columns(2)
    # Build sid_map once to avoid repeated work inside loops (used when selection is filename-based)
    sid_map = build_sid_map(idx_df)
    if cX.button(
        "➕ Add to selected" if kind != "single" else "✅ Set for selected",
        disabled=(
            not selected
            or (batch_value is None)
            or (
                not str(batch_value).strip()
                and not (kind == "single" and allow_blank)
                and not (field_type == "number")
            )
        ),
        key="batch_add_s8",
    ):
        with st.status("Adding…", expanded=True) as s:
            progress_bar = st.progress(0, text="0")
            for i, k in enumerate(selected, 1):
                if ann_key_kind == "stable_id":
                    sid = str(k).strip()
                    fn = sid_to_filename.get(sid, sid)
                else:
                    fn = str(k)
                    sid = sid_map.get(_fname_key(fn), _fname_key(fn))
                if kind == "single" or field_type == "number":
                    # clear existing values for this filename+marker
                    overlay_clear_field(out_dir, session, sid, marker, filename=fn)
                    if str(batch_value).strip() != "":
                        overlay_add(out_dir, session, sid, fn, marker, str(batch_value).strip())
                else:
                    overlay_add(out_dir, session, sid, fn, marker, str(batch_value).strip())
                progress_bar.progress(int(100 * i / max(1, len(selected))), text=f"{i}/{len(selected)}")
            s.update(label="✅ Done", state="complete")

    if cY.button("➖ Remove from selected", disabled=(not selected or (batch_value is None) or not str(batch_value).strip()), key="batch_remove_s8"):
        with st.status("Removing…", expanded=True) as s:
            progress_bar = st.progress(0, text="0")
            for i, k in enumerate(selected, 1):
                if ann_key_kind == "stable_id":
                    sid = str(k).strip()
                    fn = sid_to_filename.get(sid, sid)
                else:
                    fn = str(k)
                    sid = sid_map.get(_fname_key(fn), _fname_key(fn))
                overlay_remove(out_dir, session, sid, marker, str(batch_value).strip())
                progress_bar.progress(int(100 * i / max(1, len(selected))), text=f"{i}/{len(selected)}")
            s.update(label="✅ Done", state="complete")

    st.markdown("---")
    st.subheader("Per-image annotate")
    current_map = overlay_values_map(out_dir, session, marker, key_kind=ann_key_kind)

    recs = df_show.to_dict("records")
    for i in range(0, len(recs), int(cols_n)):
        row = recs[i : i + int(cols_n)]
        cols = st.columns(len(row))
        for c, r in zip(cols, row):
            fn = str(r.get("filename", ""))
            sid = str(r.get("stable_id") or sid_map.get(_fname_key(fn), _fname_key(fn))).strip()
            lookup = sid if ann_key_kind == "stable_id" else fn
            p = ""
            if fm:
                p = fm.get(str(lookup)) or fm.get(fn) or fm.get(_fname_key(fn)) or fm.get(Path(fn).name) or fm.get(_fname_key(Path(fn).name)) or ""
            with c:
                if p and Path(p).exists():
                    cap = fn
                    if "match_quality" in df_show.columns and "cosine_similarity" in df_show.columns:
                        try:
                            cap += f"\n{r.get('match_quality','')} (cos={float(r.get('cosine_similarity')):.3f})"
                        except Exception:
                            cap += f"\n{r.get('match_quality','')}"
                    display_image(p, width=img_w, caption=cap)
                else:
                    st.write(f"{fn}\n(path not found)")

                cur_key = sid if ann_key_kind == "stable_id" else _fname_key(fn)
                cur_vals = sorted(list(current_map.get(cur_key, set())))
                st.caption("Current: " + ("; ".join(cur_vals) if cur_vals else "—"))

                b1, b2 = st.columns(2)
                if field_type == "number":
                    # per-image numeric set/clear
                    try:
                        cur_num_raw = cur_vals[0] if cur_vals else ""
                        cur_num = float(cur_num_raw) if str(cur_num_raw).strip() != "" else 0.0
                    except Exception:
                        cur_num = 0.0
                    if num_kind == "float":
                        v = st.number_input("Value", value=float(cur_num), step=0.1, key=f"s8_num_{marker}_{sid}")
                    else:
                        v = st.number_input("Value", value=int(round(cur_num)), step=1, key=f"s8_num_{marker}_{sid}")

                    if b1.button("Set", key=f"s8_setnum_{marker}_{sid}"):
                        overlay_clear_field(st.session_state.get("output_dir", "."), session, sid, marker, filename=fn)
                        overlay_add(st.session_state.get("output_dir", "."), session, sid, fn, marker, str(v))
                        st.success("Set.")
                    if b2.button("Clear", key=f"s8_clearnum_{marker}_{sid}"):
                        overlay_clear_field(st.session_state.get("output_dir", "."), session, sid, marker, filename=fn)
                        st.info("Cleared.")

                elif kind == "multi":
                    if b1.button("Add", key=f"s8_add_{marker}_{sid}"):
                        if str(batch_value).strip():
                            overlay_add(st.session_state.get("output_dir", "."), session, sid, fn, marker, str(batch_value).strip())
                            st.success("Added.")
                        else:
                            st.warning("Enter a value first.")
                    if b2.button("Remove", key=f"s8_remove_{marker}_{sid}"):
                        if str(batch_value).strip():
                            overlay_remove(st.session_state.get("output_dir", "."), session, sid, marker, str(batch_value).strip())
                            st.info("Removed.")
                        else:
                            st.warning("Enter a value first.")
                else:
                    if b1.button("Set", key=f"s8_set_{marker}_{sid}"):
                        # set single: clear existing then add if not blank
                        overlay_clear_field(st.session_state.get("output_dir", "."), session, sid, marker, filename=fn)
                        if str(batch_value).strip() != "":
                            overlay_add(st.session_state.get("output_dir", "."), session, sid, fn, marker, str(batch_value).strip())
                            st.success("Set.")
                        else:
                            if allow_blank:
                                st.info("Cleared.")
                            else:
                                st.warning("Blank not allowed for this field.")
                    if allow_blank and b2.button("Clear", key=f"s8_clear_{marker}_{sid}"):
                        overlay_clear_field(st.session_state.get("output_dir", "."), session, sid, marker, filename=fn)
                        st.info("Cleared.")

    # Export controls (per-field or all fields)
    with st.expander("Export merged CSV (overlay -> metadata)"):
        export_scope = st.selectbox("Export scope", ["Selected field", "All fields"], index=0, key="s8_export_scope")
        export_style = st.selectbox(
            "Output style",
            ["Smart columns (recommended)", "Wide (list column)", "Binary columns"],
            index=0,
            key="s8_export_style",
        )
        st.caption(
            "**Smart columns (recommended):** uses your field type to pick a sensible column shape — "
            "Number → one numeric column; Category single → one text column; Category multi → one `*_list` column.\n"
            "**Wide (list column):** exports multi-value categories as one semicolon-separated `*_list` column (easy to read, easy to edit).\n"
            "**Binary columns:** exports multi-value categories as one-hot columns like `marker_value` (best for stats/ML, creates many columns)."
        )
        export_include_extras = st.checkbox("Include overlay-only rows (not present in metadata)", value=False, key="s8_export_include_extras")
        if st.button("⬇️ Download merged CSV", key="s8_download_overlay_merged"):
            try:
                if export_scope == "Selected field":
                    merged = export_overlay_merged(
                        st.session_state.get("meta_path", ""),
                        st.session_state.get("index_csv_path", None),
                        st.session_state.get("output_dir", "."),
                        session,
                        marker,
                        export_style,
                        export_include_extras,
                        field_spec=field,
                    )
                    out_name = f"{marker}_merged_{session}.csv"
                else:
                    merged = export_overlay_merged_all_fields(
                        st.session_state.get("meta_path", ""),
                        st.session_state.get("index_csv_path", None),
                        st.session_state.get("output_dir", "."),
                        session,
                        export_style,
                        export_include_extras,
                    )
                    out_name = f"all_fields_merged_{session}.csv"

                csvb = merged.to_csv(index=False).encode("utf-8")

                # Save server-side into repo root / "custom annotations"
                try:
                    root = Path(__file__).resolve().parent
                except Exception:
                    root = Path.cwd()
                out_dir = root / "custom annotations"
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = f"{Path(out_name).stem}_{ts}{Path(out_name).suffix}"
                out_path = out_dir / safe_name
                out_path.write_bytes(csvb)
                st.success(f"Saved merged CSV to: {str(out_path)}")

                # Offer download of the saved file
                st.download_button(
                    "Click to download merged CSV",
                    data=csvb,
                    file_name=out_name,
                    mime="text/csv",
                    key="s8_download_overlay_merged_file",
                )
            except Exception as e:
                st.error(f"Export failed: {e}")

    cL, cR = st.columns(2)
    cL.button("⬅️ Back", on_click=lambda: st.session_state.update(step=7), key="back_step_8")
    cR.button("➡️ Next (Projection)", on_click=lambda: st.session_state.update(step=9), key="next_step_8")


# --------------------------------
# Step 9 — PROJECTION (UMAP/t-SNE) + EXPORT
# --------------------------------
elif st.session_state["step"] == 9:
    st.markdown("**Step 9: Projection (UMAP/t-SNE)**")
    st.info(
        "Create a 2D map of your image collection based on embeddings. Points that appear near each other are more visually similar (approximately). "
        "Use this for exploratory browsing and presentations."
    )

    # ======== Dependency checks (use optional-deps already imported at top) ========
    has_plotly = bool(_PLOTLY_OK)

    umap_available = bool(_UMAP_OK)
    tsne_available = bool(_TSNE_OK)
    if not (umap_available or tsne_available):
        st.error("Neither UMAP nor t-SNE available. Install with: pip install umap-learn scikit-learn")
        cL, cR = st.columns(2)
        cL.button("⬅️ Back", on_click=lambda: st.session_state.update(step=8), key="back_step_9")
        cR.button("↺ Refresh", on_click=safe_rerun, key="refresh_step_9")
        st.stop()

    # ======== Controls ========
    c0, c1, c2, c3 = st.columns([1, 1, 1, 1])
    proj_options = []
    if umap_available:
        proj_options.append("UMAP")
    if tsne_available:
        proj_options.append("t-SNE")

    proj_method = c0.selectbox("Projection method", proj_options, index=0, key="export_proj_method")
    max_points = c1.number_input(
        "Max points (sampled)",
        min_value=200,
        max_value=200000,
        value=10000,
        step=500,
        key="export_max_points",
    )

    metric = "cosine"
    nn = 30
    md = 0.10
    perplexity = 30.0
    n_iter = 1000
    early_exag = 12.0
    learning_rate = "auto"
    tsne_cap = 12000

    if proj_method == "UMAP":
        nn = c2.number_input("UMAP n_neighbors", min_value=5, max_value=200, value=30, step=5, key="export_umap_nn")
        md = c3.slider("UMAP min_dist", min_value=0.0, max_value=1.0, value=0.10, step=0.01, key="export_umap_md")
        metric = st.selectbox("UMAP metric", ["cosine", "euclidean"], index=0, key="export_umap_metric")
    else:
        perplexity = c2.slider("t-SNE perplexity", 5.0, 100.0, 30.0, 1.0, key="export_tsne_perp")
        n_iter = c3.slider("t-SNE iterations", 250, 5000, 1000, 250, key="export_tsne_iters")
        col_tsne = st.columns(3)
        early_exag = col_tsne[0].slider("Early exaggeration", 4.0, 20.0, 12.0, 0.5, key="export_tsne_ee")
        lr_mode = col_tsne[1].selectbox("Learning rate", ["auto", "custom"], index=0, key="export_tsne_lrmode")
        if lr_mode == "custom":
            learning_rate = col_tsne[2].slider("LR value", 10.0, 2000.0, 200.0, 10.0, key="export_tsne_lrval")
        else:
            learning_rate = "auto"
        tsne_cap = st.number_input(
            "t-SNE max points (cap)",
            min_value=1000,
            max_value=50000,
            value=12000,
            step=1000,
            help="Safety cap; t-SNE is O(n²). Query/nearest are always kept.",
            key="export_tsne_cap",
        )

    st.markdown("---")
    color_mode = st.radio(
        "Coloring",
        ["Clear categories (query/nearest/other)", "Gradient by similarity to query (cosine)"],
        index=1,
        key="export_color_mode",
    )

    if color_mode.startswith("Gradient"):
        cA, cB = st.columns(2)
        pct_lo, pct_hi = cA.slider(
            "Gradient focus (percentile window on similarity)",
            0.0,
            100.0,
            (80.0, 100.0),
            0.5,
            key="export_grad_window",
        )
        gamma = cB.slider("Contrast (gamma, >1 = focus high end)", 0.2, 5.0, 2.0, 0.1, key="export_grad_gamma")
    else:
        pct_lo, pct_hi, gamma = 0.0, 100.0, 1.0

    compute = st.button(f"🧮 Compute {proj_method}", key="export_compute_projection")

    # ======== Prepare data -> compute projection (deterministic, only when user clicks) ========
    if compute:
        with st.status("Preparing data for projection…", expanded=False) as s:
            try:
                emb = load_embeddings(st.session_state["embed_path"])
                emb = _ensure_stable_id(_ensure_filename(emb))

                # Determine key kind (stable_id preferred).
                has_emb_sid = "stable_id" in emb.columns and emb["stable_id"].astype(str).str.strip().ne("").any()
                proj_key_kind = "stable_id" if has_emb_sid else "filename"
                key_col = "stable_id" if proj_key_kind == "stable_id" else "filename"

                # Vector columns: prefer the ones used for FAISS build (if available), else detect similarly.
                vec_cols: List[str] = []
                cand = st.session_state.get("_faiss_vec_cols")
                if isinstance(cand, list) and cand and all(str(c) in emb.columns for c in cand):
                    vec_cols = [str(c) for c in cand]
                else:
                    import re

                    orig_cols = [str(c) for c in emb.columns]
                    lower_cols = [c.strip().lower() for c in orig_cols]
                    orig_by_lower = dict(zip(lower_cols, orig_cols))

                    vec_re = re.compile(r"^(?:v|emb_)(\d+)$")
                    vec_regex_cols = [c for c in lower_cols if vec_re.match(c or "")]
                    if vec_regex_cols:
                        vec_cols_lower = sorted(vec_regex_cols, key=lambda c: int(vec_re.match(c).group(1)))  # type: ignore[union-attr]
                    else:
                        exclude = {
                            "stable_id",
                            "filename",
                            "file_name",
                            "original_filename",
                            "image",
                            "img",
                            "name",
                            "full_path",
                            "path",
                            "filepath",
                            "file_path",
                            "width",
                            "height",
                            "w",
                            "h",
                            "size_bytes",
                            "size",
                            "bytes",
                            "mtime",
                            "timestamp",
                            "year",
                            "id",
                            "index",
                            "patch_id",
                        }
                        sample = emb.head(200).copy()
                        sample.columns = [str(c).strip().lower() for c in sample.columns]
                        vec_cols_lower = []
                        for c in lower_cols:
                            if c in exclude:
                                continue
                            if c not in sample.columns:
                                continue
                            ser = pd.to_numeric(sample[c], errors="coerce")
                            non_nan = float(ser.notna().mean()) if len(ser) else 0.0
                            if non_nan >= 0.90:
                                vec_cols_lower.append(c)

                    vec_cols = [orig_by_lower[c] for c in vec_cols_lower if c in orig_by_lower]

                if not vec_cols:
                    raise RuntimeError("No detectable vector columns for projection.")

                meta_df = None
                if st.session_state.get("meta_ok") and Path(st.session_state["meta_path"]).exists():
                    try:
                        meta_df = pd.read_csv(st.session_state["meta_path"], low_memory=False)
                        meta_df = _ensure_stable_id(_ensure_filename(meta_df))
                        _has_filename(meta_df, "Metadata CSV")
                    except Exception:
                        meta_df = None

                allowed = build_allowed_set(meta_df, key_kind=proj_key_kind) if meta_df is not None else None
                if allowed is None:
                    pool = set(emb[key_col].astype(str))
                else:
                    pool = {f for f in emb[key_col].astype(str) if f in allowed}

                qpath = st.session_state.get("query_image_path", "")
                qkey = st.session_state.get("query_key")
                if proj_key_kind != "stable_id":
                    qkey = _fname_key(Path(qpath).name) if qpath else None
                else:
                    if (not qkey) and qpath and Path(qpath).exists():
                        try:
                            pp = Path(qpath)
                            stt = pp.stat()
                            qkey = _stable_id(pp, int(stt.st_size), int(stt.st_mtime))
                        except Exception:
                            qkey = None
                qkey = str(qkey).strip() if qkey else None
                if qkey and qkey in set(emb[key_col].astype(str)):
                    pool.add(qkey)

                df_res = st.session_state.get("last_results_df", pd.DataFrame())
                if (not df_res.empty) and proj_key_kind == "stable_id" and ("stable_id" in df_res.columns):
                    nearest_set = set(df_res["stable_id"].astype(str).str.strip())
                else:
                    nearest_set = set(df_res["filename"].astype(str)) if (not df_res.empty and "filename" in df_res.columns) else set()
                pool |= nearest_set

                if not pool:
                    raise RuntimeError("No filenames available for projection (check filters and embeddings).")

                pool_list = sorted(list(pool))
                rnd = random.Random(42)
                rnd.shuffle(pool_list)

                if len(pool_list) > int(max_points):
                    keep = set(pool_list[: int(max_points)])
                    if qkey:
                        keep.add(qkey)
                    keep |= nearest_set
                else:
                    keep = set(pool_list)

                sub = emb[emb[key_col].astype(str).isin(keep)].copy()
                if sub.empty:
                    raise RuntimeError("No rows matched after sampling; try increasing Max points.")

                def _label_fn(x: str) -> str:
                    x = str(x).strip()
                    if qkey and x == qkey:
                        return "query"
                    if x in nearest_set:
                        return "nearest"
                    return "other"

                sub["label"] = sub[key_col].astype(str).map(_label_fn)
                X_full = np.ascontiguousarray(sub[vec_cols].to_numpy(dtype="float32"), dtype="float32")

                if qkey and (emb[key_col].astype(str) == qkey).any():
                    qrow = emb.loc[emb[key_col].astype(str) == qkey].iloc[[0]]
                    qvec = np.ascontiguousarray(qrow[vec_cols].to_numpy(dtype="float32"), dtype="float32")
                    sim_raw = _cosine_similarity(qvec, X_full)
                    sim01_full = ((sim_raw + 1.0) / 2.0).astype("float32")
                else:
                    sim01_full = np.full(shape=(X_full.shape[0],), fill_value=np.nan, dtype="float32")

                if proj_method == "t-SNE":
                    n = X_full.shape[0]
                    cap = int(tsne_cap)
                    if n > cap:
                        idx = np.arange(n)
                        labels_np = sub["label"].to_numpy()
                        keep_mask = (labels_np == "query") | (labels_np == "nearest")
                        idx_keep = idx[keep_mask]
                        idx_rest = idx[~keep_mask]
                        need = max(0, cap - len(idx_keep))
                        if need < len(idx_rest):
                            sel = np.random.RandomState(42).choice(idx_rest, size=need, replace=False)
                            idx_final = np.concatenate([idx_keep, sel])
                        else:
                            idx_final = idx
                        X = X_full[idx_final]
                        sub = sub.iloc[idx_final].reset_index(drop=True)
                        sim01 = sim01_full[idx_final]
                    else:
                        X = X_full
                        sim01 = sim01_full
                else:
                    X = X_full
                    sim01 = sim01_full

                s.update(label=f"Prepared {len(sub):,} items. Computing {proj_method}…", state="running")

                if proj_method == "UMAP":
                    assert UMAP is not None
                    umap_model = UMAP(
                        n_neighbors=int(nn),
                        min_dist=float(md),
                        metric=metric,
                        random_state=42,
                    )
                    coords = _to_dense_2d(umap_model.fit_transform(X))
                    tsne_kwargs: Optional[dict[str, Any]] = None
                else:
                    from inspect import signature

                    assert TSNE is not None
                    tsne_kwargs = dict(
                        n_components=2,
                        perplexity=float(min(perplexity, max(5.0, (X.shape[0] - 1) / 3.0))),
                        early_exaggeration=float(early_exag),
                        init="pca",
                        metric="euclidean",
                        random_state=42,
                    )
                    params = signature(TSNE.__init__).parameters
                    if "n_iter" in params:
                        tsne_kwargs["n_iter"] = int(n_iter)
                    elif "max_iter" in params:
                        tsne_kwargs["max_iter"] = int(n_iter)
                    if "learning_rate" in params:
                        tsne_kwargs["learning_rate"] = learning_rate if isinstance(learning_rate, str) else float(learning_rate)

                    try:
                        tsne = TSNE(**tsne_kwargs)
                    except TypeError:
                        tsne_kwargs["learning_rate"] = 200.0
                        tsne_kwargs.pop("n_iter", None)
                        tsne_kwargs.pop("max_iter", None)
                        tsne = TSNE(**tsne_kwargs)

                    coords = _to_dense_2d(tsne.fit_transform(X))

                coords = _to_dense_2d(coords)

                plot_df = pd.DataFrame(
                    {
                        "x": coords[:, 0],
                        "y": coords[:, 1],
                        "filename": sub["filename"].astype(str).values,
                        "label": sub["label"].values,
                        "sim_to_query": sim01,
                    }
                )

                st.session_state["umap_df"] = plot_df
                st.session_state["umap_params"] = dict(
                    method=proj_method,
                    max_points=int(max_points),
                    n_neighbors=int(nn) if proj_method == "UMAP" else None,
                    min_dist=float(md) if proj_method == "UMAP" else None,
                    metric=metric if proj_method == "UMAP" else None,
                    perplexity=float(tsne_kwargs.get("perplexity", np.nan)) if (proj_method == "t-SNE" and tsne_kwargs) else None,
                    n_iter=int(tsne_kwargs.get("n_iter", tsne_kwargs.get("max_iter", np.nan))) if (proj_method == "t-SNE" and tsne_kwargs) else None,
                    learning_rate=tsne_kwargs.get("learning_rate", None) if (proj_method == "t-SNE" and tsne_kwargs) else None,
                    early_exaggeration=float(tsne_kwargs.get("early_exaggeration", np.nan)) if (proj_method == "t-SNE" and tsne_kwargs) else None,
                )
                s.update(label=f"Computed {proj_method} for {len(plot_df):,} points.", state="complete")
            except Exception as e:
                s.update(label=f"Error preparing data: {e}", state="error")
                st.stop()

    # ======== Plot (uses cached last computed projection) ========
    plot_df = st.session_state.get("umap_df", pd.DataFrame())
    if plot_df.empty:
        st.info("No projection computed yet. Click 'Compute' above.")
    else:
        st.markdown("**Projection**")
        plotted = False

        if has_plotly:
            try:
                import plotly.graph_objects as go

                fig = go.Figure()
                if color_mode.startswith("Clear"):
                    ddq = plot_df[plot_df["label"] == "query"]
                    if not ddq.empty:
                        fig.add_trace(
                            go.Scattergl(
                                x=ddq["x"],
                                y=ddq["y"],
                                mode="markers",
                                name="query",
                                text=ddq["filename"],
                                hoverinfo="text",
                                marker=dict(color="green", size=12, line=dict(width=1, color="black"), symbol="diamond"),
                            )
                        )
                    ddn = plot_df[plot_df["label"] == "nearest"]
                    if not ddn.empty:
                        fig.add_trace(
                            go.Scattergl(
                                x=ddn["x"],
                                y=ddn["y"],
                                mode="markers",
                                name="nearest",
                                text=ddn["filename"],
                                hoverinfo="text",
                                marker=dict(color="yellow", size=8),
                            )
                        )
                    ddo = plot_df[plot_df["label"] == "other"]
                    if not ddo.empty:
                        fig.add_trace(
                            go.Scattergl(
                                x=ddo["x"],
                                y=ddo["y"],
                                mode="markers",
                                name="other",
                                text=ddo["filename"],
                                hoverinfo="text",
                                marker=dict(color="red", size=6),
                            )
                        )
                    fig.update_layout(height=720, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="Group")
                    st_plotly_chart_stretch(fig)
                    plotted = True
                else:
                    base = plot_df[plot_df["label"] != "query"]["sim_to_query"].dropna().to_numpy(dtype="float32")
                    if base.size >= 2:
                        lo = float(np.percentile(base, pct_lo))
                        hi = float(np.percentile(base, pct_hi))
                        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                            lo, hi = 0.0, 1.0
                    else:
                        lo, hi = 0.0, 1.0
                    sim_raw = plot_df["sim_to_query"].fillna((lo + hi) / 2.0).to_numpy(dtype="float32")
                    sim_norm = np.clip((sim_raw - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
                    sim_focus = 1.0 - np.power(1.0 - sim_norm, float(gamma))
                    plot_df = plot_df.copy()
                    plot_df["sim_color"] = sim_focus

                    ddq = plot_df[plot_df["label"] == "query"]
                    if not ddq.empty:
                        fig.add_trace(
                            go.Scattergl(
                                x=ddq["x"],
                                y=ddq["y"],
                                mode="markers",
                                name="query",
                                text=ddq["filename"],
                                hoverinfo="text",
                                marker=dict(color="green", size=12, line=dict(width=1, color="black"), symbol="diamond"),
                            )
                        )
                    fig.add_trace(
                        go.Scattergl(
                            x=plot_df["x"],
                            y=plot_df["y"],
                            mode="markers",
                            name="similarity",
                            text=plot_df["filename"],
                            hoverinfo="text",
                            marker=dict(
                                size=7,
                                color=plot_df["sim_color"],
                                cmin=0,
                                cmax=1,
                                colorscale=[[0.0, "darkred"], [0.5, "orange"], [1.0, "yellow"]],
                                showscale=True,
                                colorbar=dict(title=f"cosine norm\n{pct_lo:.0f}–{pct_hi:.0f} pctl\nγ={gamma:.1f}"),
                            ),
                        )
                    )
                    fig.update_layout(height=720, margin=dict(l=10, r=10, t=10, b=10))
                    st_plotly_chart_stretch(fig)
                    plotted = True
            except Exception as e:
                st.warning(f"Plotly failed ({e.__class__.__name__}: {e}). Using Vega-Lite fallback…")

        if not plotted:
            if color_mode.startswith("Clear"):
                spec = {
                    "layer": [
                        {
                            "mark": {"type": "point", "filled": True, "tooltip": True},
                            "transform": [{"filter": "datum.label == 'other'"}],
                            "encoding": {
                                "x": {"field": "x", "type": "quantitative"},
                                "y": {"field": "y", "type": "quantitative"},
                                "color": {"value": "red"},
                                "size": {"value": 30},
                                "tooltip": [{"field": "filename", "type": "nominal"}],
                            },
                        },
                        {
                            "mark": {"type": "point", "filled": True, "tooltip": True},
                            "transform": [{"filter": "datum.label == 'nearest'"}],
                            "encoding": {
                                "x": {"field": "x", "type": "quantitative"},
                                "y": {"field": "y", "type": "quantitative"},
                                "color": {"value": "yellow"},
                                "size": {"value": 40},
                                "tooltip": [{"field": "filename", "type": "nominal"}],
                            },
                        },
                        {
                            "mark": {"type": "point", "filled": True, "tooltip": True, "shape": "diamond"},
                            "transform": [{"filter": "datum.label == 'query'"}],
                            "encoding": {
                                "x": {"field": "x", "type": "quantitative"},
                                "y": {"field": "y", "type": "quantitative"},
                                "color": {"value": "green"},
                                "size": {"value": 60},
                                "tooltip": [{"field": "filename", "type": "nominal"}],
                            },
                        },
                    ],
                    "height": 720,
                }
                st_vega_lite_chart_stretch(plot_df, spec)
            else:
                base = plot_df[plot_df["label"] != "query"]["sim_to_query"].dropna().to_numpy(dtype="float32")
                if base.size >= 2:
                    lo = float(np.percentile(base, pct_lo))
                    hi = float(np.percentile(base, pct_hi))
                    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                        lo, hi = 0.0, 1.0
                else:
                    lo, hi = 0.0, 1.0
                sim_raw = plot_df["sim_to_query"].fillna((lo + hi) / 2.0).to_numpy(dtype="float32")
                sim_norm = np.clip((sim_raw - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
                sim_focus = 1.0 - np.power(1.0 - sim_norm, float(gamma))
                plot_df = plot_df.copy()
                plot_df["sim_color"] = sim_focus

                spec = {
                    "layer": [
                        {
                            "mark": {"type": "point", "tooltip": True},
                            "encoding": {
                                "x": {"field": "x", "type": "quantitative"},
                                "y": {"field": "y", "type": "quantitative"},
                                "color": {
                                    "field": "sim_color",
                                    "type": "quantitative",
                                    "scale": {"domain": [0, 1], "range": ["darkred", "orange", "yellow"]},
                                    "legend": {"title": f"cosine norm {pct_lo:.0f}–{pct_hi:.0f} pctl, γ={gamma:.1f}"},
                                },
                                "size": {"value": 36},
                                "tooltip": [{"field": "filename", "type": "nominal"}],
                            },
                        },
                        {
                            "transform": [{"filter": "datum.label == 'query'"}],
                            "mark": {"type": "point", "filled": True, "shape": "diamond"},
                            "encoding": {
                                "x": {"field": "x", "type": "quantitative"},
                                "y": {"field": "y", "type": "quantitative"},
                                "color": {"value": "green"},
                                "size": {"value": 80},
                            },
                        },
                    ],
                    "height": 720,
                }
                st_vega_lite_chart_stretch(plot_df, spec)

        with st.expander("Data (first rows)"):
            st_dataframe_stretch(plot_df.head(100))
        st.download_button(
            "⬇️ Download projection (CSV)",
            data=plot_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{st.session_state.get('umap_params', {}).get('method', 'projection').lower()}_projection.csv",
            mime="text/csv",
        )

    cL, cR = st.columns(2)
    cL.button("⬅️ Back", on_click=lambda: st.session_state.update(step=8), key="back_step_9")
