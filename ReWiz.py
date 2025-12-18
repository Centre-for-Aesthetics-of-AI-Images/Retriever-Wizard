# -*- coding: utf-8 -*-
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
import math
import time
import random
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import streamlit as st


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
        pass

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


def _safe_write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


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
    df = _ensure_filename(df)
    _has_filename(df, "Metadata CSV")
    return df


@st.cache_data(show_spinner=False)
def load_embeddings(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = _ensure_filename(df)
    _has_filename(df, "Embeddings CSV")
    vec_cols = [c for c in df.columns if c != "filename"]
    if not vec_cols:
        raise ValueError("Embeddings CSV has no vector columns (only filename).")
    df[vec_cols] = df[vec_cols].astype("float32")
    return df


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


def build_faiss_index(emb_path: str, metric: str, progress_cb: Optional[Callable[[int, int], None]] = None):
    if not _FAISS_OK:
        raise RuntimeError("FAISS not installed. Run: pip install faiss-cpu")
    assert faiss is not None

    emb = load_embeddings(emb_path)
    vec_cols = [c for c in emb.columns if c != "filename"]
    X = np.ascontiguousarray(emb[vec_cols].to_numpy(dtype="float32"), dtype="float32")
    filenames = emb["filename"].astype(str).tolist()
    dim = int(X.shape[1])

    if "Cosine" in metric:
        faiss.normalize_L2(X)
        index: Any = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    n = X.shape[0]
    B = 20000
    for start in range(0, n, B):
        end = min(start + B, n)
        index.add(np.ascontiguousarray(X[start:end], dtype="float32"))
        if progress_cb is not None:
            progress_cb(end, n)

    return index, filenames, dim


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


def knn_filtered(
    index,
    index_filenames: List[str],
    emb_path: str,
    query_filename: str,
    k: int,
    allowed: Optional[Set[str]],
    metric: str,
) -> pd.DataFrame:
    emb = load_embeddings(emb_path)
    vec_cols = [c for c in emb.columns if c != "filename"]

    qkey = _fname_key(query_filename)
    qrow = emb.loc[emb["filename"] == qkey]
    if qrow.empty:
        raise ValueError(f"Query '{query_filename}' (key='{qkey}') not found in embeddings.")
    qvec = np.ascontiguousarray(qrow[vec_cols].to_numpy(dtype="float32").reshape(1, -1), dtype="float32")

    if "Cosine" in metric:
        nrm = np.linalg.norm(qvec, axis=1, keepdims=True)
        nrm[nrm == 0.0] = 1e-12
        qvec = qvec / nrm

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
            rows.append({"rank": len(rows) + 1, "filename": fn, "distance": distance})
            if len(rows) >= k:
                break

        if len(rows) >= k:
            break
        if k_search >= max_index_size:
            break
        k_search = min(k_search * 2, max_index_size)

    return pd.DataFrame(rows)


def add_cosine_columns(df: pd.DataFrame, emb_path: str, query_filename: str) -> pd.DataFrame:
    if df.empty:
        return df

    emb = load_embeddings(emb_path)
    vec_cols = [c for c in emb.columns if c != "filename"]

    qkey = _fname_key(query_filename)
    qrow = emb.loc[emb["filename"] == qkey]
    if qrow.empty:
        df = df.copy()
        df["cosine_similarity"] = np.nan
        df["cosine_distance"] = np.nan
        df["score_0_100"] = np.nan
        df["match_quality"] = "n/a"
        return df

    qvec = np.ascontiguousarray(qrow[vec_cols].to_numpy(dtype="float32").reshape(1, -1), dtype="float32")
    cand = emb.set_index("filename").reindex(df["filename"].astype(str))
    M = np.ascontiguousarray(cand[vec_cols].to_numpy(dtype="float32"), dtype="float32")

    mask = np.any(np.isnan(M), axis=1)
    if mask.any():
        Mf = M.copy()
        Mf[np.isnan(Mf)] = 0.0
        sims = _cosine_similarity(qvec, Mf)
        sims[mask] = np.nan
    else:
        sims = _cosine_similarity(qvec, M)

    df = df.copy()
    df["cosine_similarity"] = sims
    df["cosine_distance"] = 1.0 - sims
    df["score_0_100"] = np.round(((sims + 1.0) / 2.0) * 100.0, 1)
    df["match_quality"] = [_match_label(s) if pd.notna(s) else "n/a" for s in sims]
    return df


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


def _stable_id_for_filename(fn: str, idx_df: pd.DataFrame) -> str:
    """Return stable_id for a filename using idx_df if present; otherwise fallback to filename key.

    This safely handles idx_df missing a 'stable_id' column.
    """
    if idx_df is None or idx_df.empty:
        return _fname_key(fn)
    if "stable_id" not in idx_df.columns:
        return _fname_key(fn)
    try:
        keys = idx_df["filename"].astype(str).map(_fname_key)
        vals = idx_df["stable_id"].astype(str)
        mapping = dict(zip(keys, vals))
        return mapping.get(_fname_key(fn), _fname_key(fn))
    except Exception:
        return _fname_key(fn)


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
    if "filename" not in df.columns:
        df["filename"] = df["full_path"].astype(str).map(_fname_key)
    else:
        df["filename"] = df["filename"].astype(str).map(_fname_key)

    # Detect basename collisions (duplicate filename keys)
    try:
        if len(df) > 0:
            vc = df["filename"].astype(str).value_counts()
            dups = vc[vc > 1]
            st.session_state["_index_has_duplicates"] = bool(len(dups) > 0)
            st.session_state["_index_duplicate_examples"] = list(dups.head(10).index.astype(str))
        else:
            st.session_state["_index_has_duplicates"] = False
            st.session_state["_index_duplicate_examples"] = []
    except Exception:
        # Don't block index usage due to warnings
        pass

    # Use vectorized zip to build the mapping (faster than iterrows)
    keys = df["filename"].astype(str).tolist()
    vals = df["full_path"].astype(str).tolist()
    return dict(zip(keys, vals))


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
def build_allowed_set(meta_df: pd.DataFrame) -> Optional[Set[str]]:
    include_rules: Dict[str, object] = st.session_state.get("filter_include", {}) or {}
    exclude_rules: Dict[str, object] = st.session_state.get("filter_exclude", {}) or {}
    query_text = (st.session_state.get("filter_query_text", "") or "").strip()

    if not include_rules and not exclude_rules and not query_text:
        return None

    df = meta_df.copy()
    df = _ensure_filename(df)
    _has_filename(df, "Metadata DF")

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
            pass
    return pd.DataFrame(columns=["stable_id", "filename", "marker", "value", "timestamp"])


def overlay_values_map(output_dir: str, session: str, marker: str) -> Dict[str, Set[str]]:
    cur = overlay_load_current(output_dir, session)
    if cur.empty:
        return {}
    need = {"filename", "marker", "value"}
    if not need.issubset(set(cur.columns)):
        return {}
    cur = cur[cur["marker"].astype(str) == str(marker)].copy()
    out: Dict[str, Set[str]] = {}
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
                "timestamp": row.get("timestamp") or datetime.utcnow().isoformat(),
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
        "timestamp": datetime.utcnow().isoformat(),
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

    # group values per filename
    def uniq_sorted(vals):
        return sorted({str(v).strip() for v in vals if str(v).strip()})

    grp = cur.groupby("filename")["value"].agg(uniq_sorted)

    if style.startswith("Wide"):
        col = f"{marker}_list"
        if col in meta.columns:
            col = f"{col}_overlay"
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
            m = export_overlay_merged(meta_path, index_csv_path, output_dir, session, key, style, include_extras)
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

    # Default to project-local example data (anchored to the folder containing this script).
    ss.setdefault("meta_path", str((_HERE / "examples" / "metadata.csv").resolve()))
    ss.setdefault("embed_path", str((_HERE / "examples" / "embeddings.csv").resolve()))
    ss.setdefault("images_root", str((_HERE / "examples" / "images").resolve()))
    ss.setdefault("output_dir", str((_HERE / "examples" / "_index").resolve()))
    ss.setdefault("index_name", "index.csv")
    ss.setdefault("auto_load_index", True)

    ss.setdefault("index_metric", "Cosine (IP + normalization)")
    ss.setdefault("k_neighbors", 10)

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

    ss.setdefault("query_image_path", "")

    ss.setdefault("filter_include", {})
    ss.setdefault("filter_exclude", {})
    ss.setdefault("filter_query_text", "")

    ss.setdefault("result_meta_cols", [])
    ss.setdefault("last_results_df", pd.DataFrame())

    ss.setdefault("overlay_session", "iconography_2025q4")
    ss.setdefault("overlay_marker", "ikonografi")
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
            "- **Index**: A fast lookup table connecting filenames to file paths (and stable IDs).\n"
            "- **Nearest neighbors**: The most visually similar images to your chosen query image.\n"
            "- **Overlay / annotation**: Your added labels saved separately from the original metadata.\n"
            "- **Projection (UMAP/t-SNE)**: A 2D map where nearby points look more similar (approx.)."
        )

    if st.button("🔍 Validate metadata", key="validate_metadata"):
        with st.status("Validating metadata…", expanded=True) as s:
            try:
                df = load_metadata(st.session_state["meta_path"])
                st.session_state["meta_ok"] = True
                st.session_state["meta_head"] = df.head(20)
                s.update(label="✅ Metadata OK", state="complete")
            except Exception as e:
                st.session_state["meta_ok"] = False
                st.session_state["meta_head"] = pd.DataFrame()
                st.error(str(e))
                s.update(label="❌ Metadata NOT OK", state="error")

    st.dataframe(st.session_state["meta_head"], use_container_width=True)
    st.button("➡️ Next", disabled=not st.session_state["meta_ok"], on_click=lambda: st.session_state.update(step=2), key="next_step_1")

    if st.button("⚡ Auto-validate Steps 1–5 and go to Step 6", key="auto_validate_1_5"):
        with st.status("Auto-validating steps 1–5…", expanded=True) as s:
            # Step 1: metadata
            try:
                df = load_metadata(st.session_state["meta_path"])
                st.session_state["meta_ok"] = True
                st.session_state["meta_head"] = df.head(20)
                s.update(label="✅ Metadata OK", state="running")
            except Exception as e:
                st.session_state["meta_ok"] = False
                st.session_state["meta_head"] = pd.DataFrame()
                s.update(label=f"❌ Metadata failed: {e}", state="error")
                st.error(f"Metadata validation failed: {e}")

            # Step 2: embeddings
            try:
                df_emb = load_embeddings(st.session_state["embed_path"])
                st.session_state["embed_ok"] = True
                st.session_state["embed_head"] = df_emb.head(20)
                vec_cols = [c for c in df_emb.columns if c != "filename"]
                s.update(label=f"✅ Embeddings OK (dim={len(vec_cols)})", state="running")
            except Exception as e:
                st.session_state["embed_ok"] = False
                st.session_state["embed_head"] = pd.DataFrame()
                s.update(label=f"❌ Embeddings failed: {e}", state="error")
                st.error(f"Embeddings validation failed: {e}")

            # Step 3: images
            try:
                roots = parse_roots(st.session_state["images_root"])
                if not roots:
                    st.session_state["images_ok"] = False
                    st.session_state["images_count"] = 0
                    s.update(label="❌ Images not found (no valid roots)", state="error")
                    st.error("No valid image roots configured.")
                else:
                    files = scan_images(roots)
                    st.session_state["images_ok"] = len(files) > 0
                    st.session_state["images_count"] = len(files)
                    s.update(label=f"✅ Images OK ({len(files):,} files)", state="running")
            except Exception as e:
                st.session_state["images_ok"] = False
                st.session_state["images_count"] = 0
                s.update(label=f"❌ Images failed: {e}", state="error")
                st.error(f"Image scan failed: {e}")

            # Step 4: consistency check (metadata vs embeddings vs disk)
            try:
                meta = load_metadata(st.session_state["meta_path"]) if st.session_state.get("meta_ok") else pd.DataFrame()
                emb = load_embeddings(st.session_state["embed_path"]) if st.session_state.get("embed_ok") else pd.DataFrame()
                roots = parse_roots(st.session_state["images_root"]) if st.session_state.get("images_ok") else []
                files = scan_images(roots) if roots else []
                disk_set = {Path(p).name.lower() for p in files}

                meta_set = set(meta["filename"].astype(str)) if not meta.empty else set()
                emb_set = set(emb["filename"].astype(str)) if not emb.empty else set()

                miss_meta_on_disk = sorted(list(meta_set - disk_set))[:10]
                miss_emb_on_disk = sorted(list(emb_set - disk_set))[:10]
                miss_meta_in_emb = sorted(list(meta_set - emb_set))[:10]
                miss_emb_in_meta = sorted(list(emb_set - meta_set))[:10]

                st.write("**Auto-check mismatches (top 10)**")
                st.write("In metadata but missing on disk:", miss_meta_on_disk or "—")
                st.write("In embeddings but missing on disk:", miss_emb_on_disk or "—")
                st.write("In metadata but not in embeddings:", miss_meta_in_emb or "—")
                st.write("In embeddings but not in metadata:", miss_emb_in_meta or "—")
                s.update(label="✅ Consistency checks done", state="running")
            except Exception as e:
                s.update(label=f"❌ Consistency checks failed: {e}", state="error")
                st.error(f"Consistency check failed: {e}")

            # Step 5: load or build index
            try:
                ok, idx_path, fm, df_idx = try_load_existing_index(st.session_state["output_dir"], st.session_state["index_name"])
                if ok:
                    st.session_state["index_csv_path"] = idx_path
                    st.session_state["filemap"] = fm
                    st.session_state["index_df"] = df_idx
                    s.update(label=f"✅ Loaded existing index: {idx_path}", state="running")
                else:
                    # build index (may take time)
                    bar = st.empty()

                    def progress_cb(done, total):
                        pct = int(100 * done / max(1, total))
                        bar.progress(pct, text=f"Indexing files: {done}/{total}")

                    with st.status("Building index…", expanded=True) as s2:
                        preview, n, out_path, df_full = build_files_index(
                            st.session_state["images_root"],
                            st.session_state["output_dir"],
                            st.session_state["index_name"],
                            progress_cb=progress_cb,
                        )
                        st.dataframe(preview, use_container_width=True)
                        if out_path:
                            st.session_state["index_csv_path"] = out_path
                            df_idx = load_index_df(out_path)
                            st.session_state["index_df"] = df_idx
                            st.session_state["filemap"] = filemap_from_index(df_idx)
                            s2.update(label=f"✅ Wrote index: {out_path} ({n:,} rows)", state="running")
                        else:
                            if df_full is None:
                                raise RuntimeError("Index build returned no dataframe.")
                            assert df_full is not None
                            st.session_state["index_df"] = df_full
                            st.session_state["filemap"] = filemap_from_index(df_full)
                            s2.update(label=f"✅ Built index in-memory ({n:,} rows)", state="running")

                # final: if everything up to step5 appears ok, navigate to step6
                st.session_state.update(step=6)
                st.success("Auto-validate complete — moving to Step 6")
            except Exception as e:
                s.update(label=f"❌ Index step failed: {e}", state="error")
                st.error(f"Index build/load failed: {e}")


# ========== Step 2: Embeddings ==========
elif st.session_state["step"] == 2:
    st.markdown("**Step 2: Validate embeddings (CSV)**")
    st.info(
        "Load the embeddings table. Embeddings are numbers that describe visual features of each image. "
        "They enable similarity search (finding images that look alike)."
    )

    if st.button("🔍 Validate embeddings", key="validate_embeddings"):
        with st.status("Validating embeddings…", expanded=True) as s:
            try:
                df = load_embeddings(st.session_state["embed_path"])
                st.session_state["embed_ok"] = True
                st.session_state["embed_head"] = df.head(20)
                vec_cols = [c for c in df.columns if c != "filename"]
                st.write(f"Vector dim: **{len(vec_cols)}**")
                s.update(label="✅ Embeddings OK", state="complete")
            except Exception as e:
                st.session_state["embed_ok"] = False
                st.session_state["embed_head"] = pd.DataFrame()
                st.error(str(e))
                s.update(label="❌ Embeddings NOT OK", state="error")

    st.dataframe(st.session_state["embed_head"], use_container_width=True)
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
            roots = parse_roots(st.session_state["images_root"])
            if not roots:
                st.session_state["images_ok"] = False
                st.session_state["images_count"] = 0
                st.error("No valid roots. Check the paths.")
                s.update(label="❌ Images NOT OK", state="error")
            else:
                files = scan_images(roots)
                st.session_state["images_ok"] = len(files) > 0
                st.session_state["images_count"] = len(files)
                st.write(f"Found **{len(files):,}** images across **{len(roots)}** root(s).")
                s.update(label="✅ Images OK" if files else "❌ Images NOT OK", state="complete" if files else "error")

    st.metric("Images found", f"{st.session_state['images_count']:,}")
    cL, cR = st.columns(2)
    cL.button("⬅️ Back", on_click=lambda: st.session_state.update(step=2), key="back_step_3")
    cR.button("➡️ Next", disabled=not st.session_state["images_ok"], on_click=lambda: st.session_state.update(step=4), key="next_step_3")


# ========== Step 4: Consistency check ==========
elif st.session_state["step"] == 4:
    st.markdown("**Step 4: Consistency check (metadata vs embeddings vs disk)**")
    st.info(
        "This step checks whether the three sources agree: your metadata table, your embeddings table, and the actual image files on disk. "
        "It helps catch missing files, spelling differences, or mismatched exports before searching."
    )

    if st.button("🧪 Run check", key="run_check"):
        with st.status("Running checks…", expanded=True) as s:
            try:
                meta = load_metadata(st.session_state["meta_path"])
                emb = load_embeddings(st.session_state["embed_path"])
                roots = parse_roots(st.session_state["images_root"])
                files = scan_images(roots)
                disk_set = {Path(p).name.lower() for p in files}

                meta_set = set(meta["filename"].astype(str))
                emb_set = set(emb["filename"].astype(str))

                st.write(f"Metadata: **{len(meta_set):,}**")
                st.write(f"Embeddings: **{len(emb_set):,}**")
                st.write(f"Disk images: **{len(disk_set):,}**")

                miss_meta_on_disk = sorted(list(meta_set - disk_set))[:10]
                miss_emb_on_disk = sorted(list(emb_set - disk_set))[:10]
                miss_meta_in_emb = sorted(list(meta_set - emb_set))[:10]
                miss_emb_in_meta = sorted(list(emb_set - meta_set))[:10]

                st.markdown("**Top 10 mismatches**")
                st.write("In metadata but missing on disk:", miss_meta_on_disk or "—")
                st.write("In embeddings but missing on disk:", miss_emb_on_disk or "—")
                st.write("In metadata but not in embeddings:", miss_meta_in_emb or "—")
                st.write("In embeddings but not in metadata:", miss_emb_in_meta or "—")

                s.update(label="✅ Checks complete", state="complete")
            except Exception as e:
                st.error(str(e))
                s.update(label="❌ Checks failed", state="error")

    cL, cR = st.columns(2)
    cL.button("⬅️ Back", on_click=lambda: st.session_state.update(step=3), key="back_step_4")
    cR.button("➡️ Next", on_click=lambda: st.session_state.update(step=5), key="next_step_4")


# ========== Step 5: Build/load index ==========
elif st.session_state["step"] == 5:
    st.markdown("**Step 5: Build/load file index (filename -> full_path + stable_id)**")
    st.info(
        "Build (or load) a simple index that connects each image filename to its full file path. "
        "This makes displaying thumbnails fast and keeps your later steps responsive."
    )

    cA, cB = st.columns(2)
    if cA.button("📖 Load existing index", key="load_index"):
        ok, idx_path, fm, df_idx = try_load_existing_index(st.session_state["output_dir"], st.session_state["index_name"])
        if ok:
            st.session_state["index_csv_path"] = idx_path
            st.session_state["filemap"] = fm
            st.session_state["index_df"] = df_idx
            st.success(f"Loaded index: {idx_path} ({len(fm):,})")
            if st.session_state.get("_index_has_duplicates"):
                ex = st.session_state.get("_index_duplicate_examples", []) or []
                hint = f" Examples: {', '.join(ex[:5])}" if ex else ""
                st.warning("⚠️ Duplicate basenames detected in index; results may map to the wrong file." + hint)
            if st.session_state.get("stable_id_strategy_changed"):
                st.warning(st.session_state.get("stable_id_strategy_message", "stable_id strategy changed; rebuild index recommended"))
            st.dataframe(df_idx.head(20), use_container_width=True)
        else:
            st.warning("No existing index found.")

    if cB.button("🏗️ Build index file", key="build_index_file"):
        bar = st.empty()

        def prog(done, total):
            pct = int(100 * done / max(1, total))
            bar.progress(pct, text=f"Indexing files: {done}/{total}")

        with st.status("Building index…", expanded=True) as s:
            preview, n, out_path, df_full = build_files_index(
                st.session_state["images_root"],
                st.session_state["output_dir"],
                st.session_state["index_name"],
                progress_cb=prog,
            )
            st.dataframe(preview, use_container_width=True)

            if out_path:
                st.session_state["index_csv_path"] = out_path
                df_idx = load_index_df(out_path)
                st.session_state["index_df"] = df_idx
                st.session_state["filemap"] = filemap_from_index(df_idx)
                st.success(f"✅ Wrote index: {out_path} ({n:,} rows)")
            else:
                st.success(f"✅ Built index in-memory ({n:,} rows)")
                if df_full is None:
                    raise RuntimeError("Index build returned no dataframe.")
                assert df_full is not None
                st.download_button(
                    "⬇️ Download index.csv",
                    data=df_full.to_csv(index=False).encode("utf-8"),
                    file_name=st.session_state["index_name"],
                    mime="text/csv",
                )
                st.session_state["index_df"] = df_full
                st.session_state["filemap"] = filemap_from_index(df_full)
            s.update(label="✅ Index ready", state="complete")

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
    else:
        if st.button("🏗️ Build FAISS index", key="build_faiss_index"):
            bar = st.empty()

            def prog(done, total):
                pct = int(100 * done / max(1, total))
                bar.progress(pct, text=f"Adding vectors: {done}/{total}")

            with st.status("Building FAISS…", expanded=True) as s:
                try:
                    index, filenames, dim = build_faiss_index(
                        st.session_state["embed_path"],
                        st.session_state["index_metric"],
                        progress_cb=prog,
                    )
                    st.session_state["_faiss_index"] = index
                    st.session_state["embed_filenames"] = filenames
                    st.session_state["faiss_dim"] = dim
                    st.session_state["faiss_ready"] = True
                    s.update(label=f"✅ FAISS ready ({len(filenames):,} vectors, dim={dim})", state="complete")
                except Exception as e:
                    st.session_state["faiss_ready"] = False
                    st.error(str(e))
                    s.update(label="❌ FAISS build failed", state="error")

    st.markdown("**Choose query image**")
    fm = st.session_state.get("filemap", {}) or {}

    cU, cP = st.columns(2)
    uploaded = cU.file_uploader("Upload image", type=[e.strip(".") for e in sorted(SUPPORTED_IMG)])
    if uploaded is not None:
        name = Path(uploaded.name).name
        key = _fname_key(name)
        if key in fm and Path(fm[key]).exists():
            st.session_state["query_image_path"] = fm[key]
            st.success(f"Matched index file: {name}")
        else:
            tmp_dir = Path(st.session_state["output_dir"] or ".")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / name
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.session_state["query_image_path"] = str(tmp_path)
            st.info(f"Saved temp: {tmp_path}")

    if fm:
        filt = cP.text_input("Filter filenames (substring)", "")
        keys = sorted(list(fm.keys()))
        show = [k for k in keys if filt.lower() in k.lower()] if filt.strip() else keys[:1000]
        sel = cP.selectbox("Pick from index", options=["(none)"] + show)
        if sel != "(none)" and cP.button("Use selected filename", key="use_selected_filename"):
            p = fm.get(sel, "")
            if p and Path(p).exists():
                st.session_state["query_image_path"] = p
                st.success(f"Selected: {Path(p).name}")
            else:
                st.error("Path not found for selected key.")

    qpath = st.session_state.get("query_image_path", "")
    st.write("Selected path:")
    st.code(qpath or "(none)")
    if qpath and Path(qpath).exists():
        display_image(qpath, width=360)

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
    can_search = bool(st.session_state.get("faiss_ready")) and bool(qpath)
    with st.form("knn_form"):
        k = st.number_input("K neighbors", min_value=1, max_value=200, value=int(st.session_state["k_neighbors"]))
        img_w = st.slider("Image width (px)", 160, 1200, 320, step=20)
        do = st.form_submit_button("🔎 Find K nearest", disabled=not can_search)

    if do:
        st.session_state["k_neighbors"] = int(k)

        allowed = build_allowed_set(meta_df) if meta_df is not None else None
        qname = Path(qpath).name

        with st.status("Searching…", expanded=True) as s:
            try:
                df = knn_filtered(
                    st.session_state["_faiss_index"],
                    st.session_state["embed_filenames"],
                    st.session_state["embed_path"],
                    qname,
                    int(k),
                    allowed=allowed,
                    metric=st.session_state["index_metric"],
                )
                df = add_cosine_columns(df, st.session_state["embed_path"], qname)

                if meta_df is not None and st.session_state.get("result_meta_cols"):
                    keep = ["filename"] + [c for c in st.session_state["result_meta_cols"] if c in meta_df.columns]
                    df = df.merge(meta_df[keep], on="filename", how="left")

                st.session_state["last_results_df"] = df
                s.update(label=f"✅ Done ({len(df):,} results)", state="complete")
            except Exception as e:
                st.error(str(e))
                s.update(label="❌ Search failed", state="error")

        df = st.session_state.get("last_results_df", pd.DataFrame())
        if df.empty:
            st.warning("No results (try fewer filters or a higher K).")
        else:
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "⬇️ Download results (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="retriever_results.csv",
                mime="text/csv",
            )

            # thumbnails
            st.markdown("**Thumbnails**")
            cols = st.columns(5)
            for i, r in enumerate(df.to_dict("records")):
                fn = r.get("filename", "")
                path = fm.get(fn, "")
                cap = f"{i+1}. {fn}"
                if "match_quality" in df.columns:
                    cap += f"\n{r.get('match_quality', '')}"
                with cols[i % 5]:
                    if path and Path(path).exists():
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
    qpath = st.session_state.get("query_image_path", "")
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

        if show_query and qpath and Path(qpath).exists():
            st.subheader("Query image")
            display_image(qpath, width=img_w, caption=Path(qpath).name)

        st.subheader("Nearest neighbors")
        if hide_meta:
            grid_cols = st.slider("Grid columns", 1, 8, 3)
            recs = df.to_dict("records")
            for i in range(0, len(recs), grid_cols):
                row = recs[i : i + grid_cols]
                cols = st.columns(len(row))
                for j, (c, r) in enumerate(zip(cols, row), start=1):
                    fn = r.get("filename", "")
                    cap = f"#{i+j} — {fn}"
                    if "match_quality" in df.columns and "cosine_similarity" in df.columns:
                        try:
                            cap += f"\n{r.get('match_quality','')} (cos={float(r.get('cosine_similarity')):.3f})"
                        except Exception:
                            cap += f"\n{r.get('match_quality','')}"
                    with c:
                        p = ""
                        if fm:
                            p = fm.get(fn) or fm.get(_fname_key(fn)) or fm.get(Path(fn).name) or fm.get(_fname_key(Path(fn).name)) or ""
                        if p and Path(p).exists():
                            display_image(p, width=img_w, caption=cap)
                        else:
                            st.write(f"{cap}\n(path not found)")
        else:
            for i, r in enumerate(df.to_dict("records"), start=1):
                fn = r.get("filename", "")
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
                        p = fm.get(fn) or fm.get(_fname_key(fn)) or fm.get(Path(fn).name) or fm.get(_fname_key(Path(fn).name)) or ""
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
    cL.button("⬅️ Back", on_click=lambda: st.session_state.update(step=6), key="back_step_7")
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
        st.write("Define annotation fields for this session. Each field becomes a 'marker' in overlay storage.")
        with st.form("add_field_form", clear_on_submit=True):
            t_col1, t_col2 = st.columns([3, 1])
            title = t_col1.text_input("Field title (human)")
            key_suggest = _field_key_from_title(title or "")
            key = t_col2.text_input("Field key (marker)", value=key_suggest)
            kind = st.selectbox("Kind", ["single", "multi"], index=0)
            vals = st.text_input("Allowed values (comma-separated)")
            allow_custom = st.checkbox("Allow custom values (user can type)", value=True)
            allow_blank = st.checkbox("Allow blank/clear (only for single)", value=False)
            submitted = st.form_submit_button("Add / Update field")
            if submitted:
                key = (key or key_suggest or "").strip()
                if not key:
                    st.error("Field key is required.")
                else:
                    values_list = [v.strip() for v in vals.split(",") if v.strip()] if vals else []
                    fields.setdefault(key, {})
                    fields[key]["title"] = title or key
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
                col_a.write(f"**{spec.get('title', fk)}** (`{fk}`) — {spec.get('kind', '')}")

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

    df_show = df_res.sort_values("rank", ascending=True).head(int(top_n)) if "rank" in df_res.columns else df_res.head(int(top_n))
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
    selected = st.multiselect("Selected filenames", options=page_keys, key="s8_selected_list")

    # value input (depends on field)
    allowed_vals = field.get("values", []) or []
    allow_custom = bool(field.get("allow_custom", False))
    kind = field.get("kind", "multi")
    allow_blank = bool(field.get("allow_blank", False))

    batch_value = None
    if allowed_vals:
        if kind == "single" and allow_blank:
            opts = ["(clear)"] + allowed_vals
        else:
            opts = allowed_vals
        batch_sel = st.selectbox("Select value", options=opts, index=0 if opts else 0, key="s8_batch_sel")
        batch_value = "" if batch_sel == "(clear)" else batch_sel
        if allow_custom:
            custom = st.text_input("Custom value (overrides selection)", key="s8_batch_custom")
            if custom and custom.strip():
                batch_value = custom.strip()
    else:
        # free text
        batch_value = st.text_input("Value", key="s8_batch_value")

    cX, cY = st.columns(2)
    # Build sid_map once to avoid repeated work inside loops
    sid_map = build_sid_map(idx_df)
    if cX.button("➕ Add to selected", disabled=(not selected or (batch_value is None) or (not str(batch_value).strip() and not (kind == "single" and allow_blank))), key="batch_add_s8"):
        with st.status("Adding…", expanded=True) as s:
            progress_bar = st.progress(0, text="0")
            for i, fn in enumerate(selected, 1):
                sid = sid_map.get(_fname_key(fn), _fname_key(fn))
                if kind == "single":
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
            for i, fn in enumerate(selected, 1):
                sid = sid_map.get(_fname_key(fn), _fname_key(fn))
                overlay_remove(out_dir, session, sid, marker, str(batch_value).strip())
                progress_bar.progress(int(100 * i / max(1, len(selected))), text=f"{i}/{len(selected)}")
            s.update(label="✅ Done", state="complete")

    st.markdown("---")
    st.subheader("Per-image annotate")
    current_map = overlay_values_map(out_dir, session, marker)

    recs = df_show.to_dict("records")
    for i in range(0, len(recs), int(cols_n)):
        row = recs[i : i + int(cols_n)]
        cols = st.columns(len(row))
        for c, r in zip(cols, row):
            fn = str(r.get("filename", ""))
            sid = sid_map.get(_fname_key(fn), _fname_key(fn))
            p = ""
            if fm:
                p = fm.get(fn) or fm.get(_fname_key(fn)) or fm.get(Path(fn).name) or fm.get(_fname_key(Path(fn).name)) or ""
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

                cur_vals = sorted(list(current_map.get(fn, set())))
                st.caption("Current: " + ("; ".join(cur_vals) if cur_vals else "—"))

                b1, b2 = st.columns(2)
                if kind == "multi":
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
        export_style = st.selectbox("Output style", ["Wide (list column)", "Binary columns"], index=0, key="s8_export_style")
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
                vec_cols = [c for c in emb.columns if c != "filename"]

                meta_df = None
                if st.session_state.get("meta_ok") and Path(st.session_state["meta_path"]).exists():
                    try:
                        meta_df = pd.read_csv(st.session_state["meta_path"], low_memory=False)
                        meta_df = _ensure_filename(meta_df)
                        _has_filename(meta_df, "Metadata CSV")
                    except Exception:
                        meta_df = None

                allowed = build_allowed_set(meta_df) if meta_df is not None else None
                if allowed is None:
                    pool = set(emb["filename"].astype(str))
                else:
                    pool = {f for f in emb["filename"].astype(str) if f in allowed}

                qpath = st.session_state.get("query_image_path", "")
                qkey = _fname_key(Path(qpath).name) if qpath else None
                if qkey and qkey in set(emb["filename"].astype(str)):
                    pool.add(qkey)

                df_res = st.session_state.get("last_results_df", pd.DataFrame())
                nearest_set = set(df_res["filename"].astype(str)) if not df_res.empty else set()
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

                sub = emb[emb["filename"].astype(str).isin(keep)].copy()
                if sub.empty:
                    raise RuntimeError("No rows matched after sampling; try increasing Max points.")

                def _label_fn(x: str) -> str:
                    x = str(x)
                    if qkey and x == qkey:
                        return "query"
                    if x in nearest_set:
                        return "nearest"
                    return "other"

                sub["label"] = sub["filename"].astype(str).map(_label_fn)
                X_full = np.ascontiguousarray(sub[vec_cols].to_numpy(dtype="float32"), dtype="float32")

                if qkey and (emb["filename"] == qkey).any():
                    qrow = emb.loc[emb["filename"] == qkey].iloc[[0]]
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
                    st.plotly_chart(fig, use_container_width=True)
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
                    st.plotly_chart(fig, use_container_width=True)
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
                st.vega_lite_chart(plot_df, spec, use_container_width=True)
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
                st.vega_lite_chart(plot_df, spec, use_container_width=True)

        with st.expander("Data (first rows)"):
            st.dataframe(plot_df.head(100), use_container_width=True)
        st.download_button(
            "⬇️ Download projection (CSV)",
            data=plot_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{st.session_state.get('umap_params', {}).get('method', 'projection').lower()}_projection.csv",
            mime="text/csv",
        )

    cL, cR = st.columns(2)
    cL.button("⬅️ Back", on_click=lambda: st.session_state.update(step=8), key="back_step_9")
