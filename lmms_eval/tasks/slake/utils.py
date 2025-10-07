from __future__ import annotations
import os, re, zipfile
from functools import lru_cache
from typing import Dict, Any, List, Optional

from PIL import Image
from loguru import logger as eval_logger

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None  # we'll error cleanly if missing

def _dir_nonempty(p: str) -> bool:
    return os.path.isdir(p) and bool(os.listdir(p))

def _extract_zip_once(zp: str, out_dir: str) -> Optional[str]:
    try:
        if _dir_nonempty(out_dir):
            return out_dir
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(zp) as zf:
            eval_logger.info(f"Extracting {os.path.basename(zp)} -> {out_dir} (one-time)...")
            zf.extractall(out_dir)
        return out_dir if _dir_nonempty(out_dir) else None
    except Exception as e:
        eval_logger.warning(f"Failed to extract {zp}: {e}")
        return None

@lru_cache(maxsize=8)
def _slake_image_root_for_repo(repo_id: str) -> str:
    """
    Resolve (and if needed, download) the SLAKE images directory.
    BoKelvin/SLAKE has an 'imgs.zip' that expands to an 'imgs/' folder containing
    relative paths that match 'img_name' (e.g., 'xmlab100/source.jpg').  :contentReference[oaicite:1]{index=1}
    """
    if not repo_id:
        eval_logger.error("SLAKE resolver: empty repo_id")
        return ""
    if snapshot_download is None:
        eval_logger.error("huggingface_hub not installed. pip install -U huggingface_hub")
        return ""

    snap = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["imgs/*", "imgs.zip"],
        resume_download=True,
        local_files_only=False,
    )

    imgs_dir = os.path.join(snap, "imgs")
    if _dir_nonempty(imgs_dir):
        return imgs_dir

    zp = os.path.join(snap, "imgs.zip")
    if os.path.isfile(zp):
        out = _extract_zip_once(zp, imgs_dir)
        if out and _dir_nonempty(out):
            return out

    eval_logger.error(f"SLAKE resolver: could not find 'imgs/' in snapshot {snap}")
    return ""

# ---------------- Text / Target ----------------

def _slake_norm(s: str) -> str:
    """Lowercase, strip, collapse spaces, remove trivial punctuation for fair exact match."""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.strip().lower()
    # yes/no aliases
    y = {"yes", "y", "yeah", "yep", "true", "1"}
    n = {"no", "n", "nope", "false", "0"}
    if s in y: return "yes"
    if s in n: return "no"
    # remove punctuation except alphanum and spaces
    s = re.sub(r"[^a-z0-9\s.%/-]", "", s)   # keep %, /, - for medical terms
    s = re.sub(r"\s+", " ", s).strip()
    return s

def slake_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """
    SLAKE sample fields (per HF viewer): img_name, question, answer, q_lang, answer_type (OPEN/CLOSED), ...  :contentReference[oaicite:2]{index=2}
    """
    pre = ""
    post = ""
    lang_filter = None
    if lmms_eval_specific_kwargs:
        pre = lmms_eval_specific_kwargs.get("pre_prompt", "") or ""
        post = lmms_eval_specific_kwargs.get("post_prompt", "") or ""
        lang_filter = lmms_eval_specific_kwargs.get("q_lang")

    if lang_filter and str(doc.get("q_lang", "")).lower() != str(lang_filter).lower():
        # Return a harmless prompt; process_results will skip scoring for non-matching lang.
        return ""

    q = (doc.get("question") or "").strip()
    return f"{pre}{q}\n{post}".strip()

def slake_doc_to_target(doc: Dict[str, Any]) -> str:
    return (doc.get("answer") or "").strip()

# ---------------- Images ----------------

def slake_doc_to_visual(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None):
    """Return [PIL.Image] for the path referenced by 'img_name' inside the 'imgs/' tree."""
    repo_id = ""
    if lmms_eval_specific_kwargs:
        repo_id = lmms_eval_specific_kwargs.get("dataset_repo") or lmms_eval_specific_kwargs.get("dataset_path") or ""
    if not repo_id:
        repo_id = "BoKelvin/SLAKE"

    root = _slake_image_root_for_repo(repo_id)
    if not root:
        raise RuntimeError("SLAKE images not available; ensure network/HF token and disk space are OK.")

    img_rel = str(doc.get("img_name") or "").strip().lstrip("/\\")
    if not img_rel:
        return []

    path = os.path.join(root, img_rel)
    if not os.path.isfile(path):
        # fallback: search by basename if path mapping fails
        basename = os.path.basename(img_rel)
        for r, _d, files in os.walk(root):
            if basename in files:
                path = os.path.join(r, basename)
                break

    if not os.path.isfile(path):
        eval_logger.warning(f"SLAKE: image not found for {img_rel}")
        return []

    try:
        with Image.open(path) as im:
            return [im.convert("RGB")]
    except Exception as e:
        eval_logger.warning(f"SLAKE: failed to open {path}: {e}")
        return []

# ---------------- Scoring ----------------

def slake_process_results(doc: Dict[str, Any], results: List[str], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None):
    """
    Produce per-item metric dict. If q_lang filter is set and doesn't match, skip this item.
    """
    lang_filter = None
    if lmms_eval_specific_kwargs:
        lang_filter = lmms_eval_specific_kwargs.get("q_lang")
    if lang_filter and str(doc.get("q_lang", "")).lower() != str(lang_filter).lower():
        # returning empty dict means it won't contribute to metrics
        return {}

    pred_raw = (results[0] if results else "") or ""
    # Heuristic: if model outputs "Answer: ..." take the last such tag
    m = list(re.finditer(r"answer\s*[:\-]\s*(.+)", pred_raw, flags=re.IGNORECASE))
    if m:
        pred = m[-1].group(1).strip()
    else:
        pred = pred_raw.strip()
    target = slake_doc_to_target(doc)

    # normalize
    p = _slake_norm(pred)
    t = _slake_norm(target)
    score = 1.0 if (p == t and p != "") else 0.0

    # build a stable id
    qid = str(doc.get("qid", "")) or f"{doc.get('img_name','')}::{doc.get('question','')}"[:256]
    return {"accuracy": {"question_id": qid, "score": score}}

def slake_aggregate_results(results: List[Dict[str, Any]]) -> float:
    """
    Aggregate accuracy over items that actually produced a metric dict.
    """
    if not results:
        return 0.0
    total = 0.0
    count = 0
    for r in results:
        if not isinstance(r, dict): 
            continue
        acc = r.get("score")
        # When called by lmms-eval's aggregator, we may receive dicts or already-extracted scores
        if acc is None:
            # try nested {"accuracy":{"question_id":..., "score":...}}
            acc = r.get("accuracy", {}).get("score", None)
        if acc is None:
            continue
        total += float(acc)
        count += 1
    acc_pct = (total / count) * 100.0 if count else 0.0
    eval_logger.info(f"SLAKE Accuracy: {acc_pct:.2f}")
    return acc_pct
