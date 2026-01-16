"""Element encoder utilities for Retriever Wizard.

Provides SigLIP2 text and image embeddings in a shared joint space.

Public API:
- embed_text(text) -> np.ndarray shape (1, dim)
- embed_image(path_or_pil, crop_box=None) -> np.ndarray shape (1, dim)

Outputs are L2-normalized (appropriate for cosine similarity / inner-product FAISS).

Model selection:
- Set env var REWIZ_SIGLIP2_MODEL to override default model id.
"""

from __future__ import annotations

import os
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


def _pick_device() -> str:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@lru_cache(maxsize=1)
def _resolve_model_id(model_id: str) -> str:
    raw = str(model_id or "").strip()
    if not raw:
        return "google/siglip2-giant-opt-patch16-384"

    key = raw.lower().replace("-", "_")
    aliases = {
        # User-friendly alias; maps to the canonical Transformers checkpoint.
        "siglip2_giant384": "google/siglip2-giant-opt-patch16-384",
        "siglip2_giant_opt_patch16_384": "google/siglip2-giant-opt-patch16-384",
    }
    return aliases.get(key, raw)


@lru_cache(maxsize=4)
def _load_siglip2(model_id: str):
    model_id = _resolve_model_id(model_id)

    # Hugging Face Hub defaults to a fairly small read timeout (often 10s),
    # which can cause noisy retries on slower networks.
    # Allow users to override via env, but set safer defaults here.
    os.environ.setdefault("HF_HUB_READ_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_CONNECT_TIMEOUT", "30")
    # huggingface_hub uses a separate timeout for HEAD/etag metadata calls (commonly 10s default).
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

    # Keep HF/Transformers logging from spamming Streamlit terminals unless the
    # user explicitly opts in to debug-level noise.
    try:
        rewiz_debug = str(os.environ.get("REWIZ_DEBUG", "")).strip().lower() in {"1", "true", "yes"}
        if rewiz_debug:
            logging.getLogger("huggingface_hub").setLevel(logging.INFO)
            logging.getLogger("huggingface_hub.utils._http").setLevel(logging.INFO)
            logging.getLogger("transformers").setLevel(logging.INFO)
        else:
            # Default: keep Streamlit terminal clean.
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.WARNING)
    except Exception:
        pass

    # Lazy imports so ReWiz can run without HF deps unless element search is used.
    import torch  # type: ignore
    from transformers import AutoModel, AutoProcessor  # type: ignore

    device = _pick_device()
    def _looks_like_timeout(err: Exception) -> bool:
        msg = repr(err)
        return (
            "ReadTimeout" in msg
            or "ReadTimeoutError" in msg
            or "connect timeout" in msg.lower()
            or "read timeout" in msg.lower()
        )

    # Prefer fast processors to avoid deprecation warnings.
    # However, some fast image processors require torchvision; if it's missing,
    # fall back to the slow processor so the app still works.
    def _load_processor(local_only: bool = False):
        try:
            return AutoProcessor.from_pretrained(model_id, use_fast=True, local_files_only=local_only)
        except TypeError:
            # Older Transformers versions may not support `use_fast`.
            return AutoProcessor.from_pretrained(model_id, local_files_only=local_only)
        except Exception as e:
            msg = str(e)
            if "torchvision" in msg.lower():
                # Fast processor requested but torchvision missing; retry with slow processor.
                try:
                    return AutoProcessor.from_pretrained(model_id, use_fast=False, local_files_only=local_only)
                except TypeError:
                    # If use_fast isn't accepted, just retry without it.
                    return AutoProcessor.from_pretrained(model_id, local_files_only=local_only)
            raise

    def _load_model(local_only: bool = False):
        return AutoModel.from_pretrained(model_id, local_files_only=local_only)

    # If the user sets REWIZ_HF_LOCAL_ONLY=1, never touch the network.
    local_only_env = str(os.environ.get("REWIZ_HF_LOCAL_ONLY", "")).strip().lower() in {"1", "true", "yes"}

    try:
        processor = _load_processor(local_only=local_only_env)
        model = _load_model(local_only=local_only_env)
    except Exception as e:
        # If we hit hub timeouts (or user asked for local-only), retry using local cache only.
        if local_only_env or _looks_like_timeout(e):
            processor = _load_processor(local_only=True)
            model = _load_model(local_only=True)
        else:
            raise

    model.eval()
    model.to(device)

    return model, processor, device


def current_model_id() -> str:
    """Return the currently configured model id (after alias resolution)."""
    return _resolve_model_id(os.environ.get("REWIZ_SIGLIP2_MODEL", "google/siglip2-giant-opt-patch16-384"))


def clear_model_cache() -> None:
    """Clear the in-process model cache.

    Useful when switching REWIZ_SIGLIP2_MODEL inside a long-running Streamlit process.
    """
    try:
        _load_siglip2.cache_clear()
    except Exception:
        pass


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    n = np.linalg.norm(vec, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return vec / n


def _to_pil(path_or_pil):
    from PIL import Image

    if hasattr(path_or_pil, "read"):
        # file-like
        img = Image.open(path_or_pil)
        return img.convert("RGB")

    if isinstance(path_or_pil, (str, os.PathLike, Path)):
        img = Image.open(str(path_or_pil))
        return img.convert("RGB")

    # Assume PIL.Image.Image
    if isinstance(path_or_pil, Image.Image):
        return path_or_pil.convert("RGB")

    raise TypeError("path_or_pil must be a file path or PIL.Image.Image")


def embed_text(text: str) -> np.ndarray:
    """Embed text into the SigLIP2 joint space.

    Returns a float32 numpy array of shape (1, dim), L2-normalized.
    """

    if text is None or not str(text).strip():
        raise ValueError("Text must be non-empty")

    model_id = os.environ.get("REWIZ_SIGLIP2_MODEL", "google/siglip2-giant-opt-patch16-384")
    model, processor, device = _load_siglip2(model_id)

    import torch  # type: ignore

    # NOTE: Hugging Face's SigLIP docs recommend `padding="max_length"` because
    # that's how the model was trained. Using dynamic padding can measurably
    # change retrieval behavior.
    inputs = processor(text=[str(text)], padding="max_length", truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        if hasattr(model, "get_text_features"):
            feats = model.get_text_features(**inputs)
        else:
            out = model(**inputs)
            feats = getattr(out, "text_embeds", None)
            if feats is None:
                raise RuntimeError("Model output has no text_embeds; unsupported model class")

    vec = feats.detach().float().cpu().numpy()
    return _l2_normalize(vec)


def embed_image(
    path_or_pil: Union[str, os.PathLike, "object"],
    crop_box: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """Embed an image (optionally cropped) into the SigLIP2 joint space.

    crop_box is (x0, y0, x1, y1) in pixel coordinates.

    Returns a float32 numpy array of shape (1, dim), L2-normalized.
    """

    img = _to_pil(path_or_pil)
    if crop_box is not None:
        x0, y0, x1, y1 = crop_box
        img = img.crop((int(x0), int(y0), int(x1), int(y1)))

    model_id = os.environ.get("REWIZ_SIGLIP2_MODEL", "google/siglip2-giant-opt-patch16-384")
    model, processor, device = _load_siglip2(model_id)

    import torch  # type: ignore

    inputs = processor(images=[img], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        if hasattr(model, "get_image_features"):
            feats = model.get_image_features(**inputs)
        else:
            out = model(**inputs)
            feats = getattr(out, "image_embeds", None)
            if feats is None:
                raise RuntimeError("Model output has no image_embeds; unsupported model class")

    vec = feats.detach().float().cpu().numpy()
    return _l2_normalize(vec)


# Internal helper for patch-index batching (not part of the required public API).
def _embed_pil_batch(pil_images) -> np.ndarray:
    model_id = os.environ.get("REWIZ_SIGLIP2_MODEL", "google/siglip2-giant-opt-patch16-384")
    model, processor, device = _load_siglip2(model_id)

    import torch  # type: ignore

    inputs = processor(images=list(pil_images), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        if hasattr(model, "get_image_features"):
            feats = model.get_image_features(**inputs)
        else:
            out = model(**inputs)
            feats = getattr(out, "image_embeds", None)
            if feats is None:
                raise RuntimeError("Model output has no image_embeds; unsupported model class")

    vec = feats.detach().float().cpu().numpy().astype(np.float32)
    return _l2_normalize(vec)
