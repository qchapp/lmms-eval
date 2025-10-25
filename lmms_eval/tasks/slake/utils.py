# utils_slake.py
from __future__ import annotations

import json
import os
import re
import zipfile
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image
from loguru import logger as eval_logger

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

# ---------------------------
# Language filter (hardcoded to English)
# ---------------------------
LANG_KEEP = "en"

def _is_en(doc: Dict[str, Any]) -> bool:
    return str(doc.get("q_lang", "")).lower() == LANG_KEEP

# ---------------------------
# Debug logging (opt-in)
# ---------------------------
DEBUG_PATH = os.environ.get("LMMS_DEBUG_SAMPLES")
DEBUG_MAX = int(os.environ.get("LMMS_DEBUG_MAX", "50"))

def _debug_log(payload: Dict[str, Any]) -> None:
    if not DEBUG_PATH:
        return
    try:
        _debug_log._n += 1  # type: ignore[attr-defined]
    except AttributeError:
        _debug_log._n = 1  # type: ignore[attr-defined]
    if _debug_log._n > DEBUG_MAX:  # type: ignore[attr-defined]
        return
    try:
        with open(DEBUG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ---------------------------
# Legacy VQA-style normalization (kept for completeness)
# ---------------------------
_ARTICLES = {"a", "an", "the"}
_NUM_MAP = {
    "zero":"0","one":"1","two":"2","three":"3","four":"4",
    "five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10",
}
_YES = {"yes","y","yeah","yep","true","1"}
_NO  = {"no","n","nope","false","0"}

def _normalize_vqa(s: str) -> str:
    """Legacy normalizer (not used in the new comparison, but kept to avoid breaking imports)."""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.strip().lower()
    if s in _YES: return "yes"
    if s in _NO:  return "no"
    s = re.sub(r"[^\w\s%/.-]", " ", s)  # keep %, /, -, .
    toks = [t for t in re.split(r"\s+", s) if t]
    out: List[str] = []
    for t in toks:
        if t in _ARTICLES:
            continue
        out.append(_NUM_MAP.get(t, t))
    s2 = " ".join(out)
    s2 = re.sub(r"\s+", " ", s2).strip().strip(".")
    return s2

# ---------------------------
# Enhanced extraction + normalization
# ---------------------------

# Lightweight synonym/alias map (extend as needed)
_SYNONYMS: Dict[str, str] = {
    # modalities
    "computed tomography": "ct",
    "ct scan": "ct",
    "x ray": "x-ray",
    "cxr": "x-ray",

    # anatomy / regions
    "thorax": "chest",

    # diseases
    "ca": "cancer",
    "carcinoma": "cancer",
}

# tokens that we consider "sides"/laterality
_SIDES = {"left", "right", "bilateral"}

# basic set of anatomy tokens where naïve singularization is safe
_SAFE_SINGULAR = {
    "lung", "lobe", "kidney", "rib", "vertebra", "spine",
    "spinal", "cord", "liver", "heart", "vessel", "artery", "vein",
    "bronchus", "bronchi", "pleura", "pleurae", "lymph", "node", "nodes",
    "chest"
}

_SPLIT_PAT = re.compile(r"(?:\band\b)|[,/;]|、|，|；", flags=re.IGNORECASE)

def _apply_synonyms(text: str) -> str:
    s = text
    # phrase-level first
    for k, v in _SYNONYMS.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s)
    return s

def _simple_singular(token: str) -> str:
    """Very light lemmatization to handle common plurals safely."""
    t = token
    if t.endswith("ies") and len(t) > 3:
        return t[:-3] + "y"  # anomalies -> anomaly
    if t.endswith("ves") and len(t) > 3:
        return t[:-3] + "f"  # not common here, but harmless
    if t.endswith("s") and len(t) > 3 and ((t[:-1] in _SAFE_SINGULAR) or (t.rstrip("s") in _SAFE_SINGULAR)):
        return t.rstrip("s")
    return t

def _normalize_atom(token: str) -> str:
    """Normalize a single span, preserving %, /, -, . and applying synonyms + singularization."""
    t = token.strip().lower()
    if not t:
        return ""
    if t in _YES: return "yes"
    if t in _NO:  return "no"
    # keep %, /, -, . ; drop other punctuation
    t = re.sub(r"[^\w\s%/.-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _apply_synonyms(t)
    # singularize where safe
    words = [_simple_singular(w) for w in t.split()]
    t = " ".join(words)
    # drop articles
    toks = [w for w in t.split() if w not in _ARTICLES]
    return " ".join(toks).strip().strip(".")

def _expand_sided_organs(items: List[str]) -> List[str]:
    """
    If we have side-only tokens plus organs elsewhere (e.g., "left, right" + "lung"),
    expand to "left lung", "right lung" and remove the standalone sides.
    Handles 'bilateral' -> left+right.
    """
    s = set(items)
    sides = {x for x in s if x in _SIDES}
    if "bilateral" in sides:
        sides.discard("bilateral")
        sides.update({"left", "right"})
    # organs we care to pair with sides
    organs = {x for x in s if x not in _SIDES and x not in {"left lung", "right lung"}}
    out = [x for x in items if x not in _SIDES]  # drop standalone sides for now
    if sides and ("lung" in organs or any(x.endswith(" lung") for x in organs)):
        # if plain 'lung' present, generate sided versions
        if "lung" in organs:
            out = [x for x in out if x != "lung"]
            for side in sorted(sides):
                out.append(f"{side} lung")
        # phrases like 'upper lobe of right lung' are left untouched
    return out

def _split_multi(s: str) -> List[str]:
    """Split multi-label strings on commas/;/slashes/‘and’ while being conservative."""
    parts = [p.strip() for p in _SPLIT_PAT.split(s) if p and p.strip()]
    return parts if parts else [s.strip()]

def _normalize_to_either_span_or_set(text: str) -> Tuple[str, Union[str, Tuple[str, ...]]]:
    """
    Returns:
      - ("set", tuple(sorted_elems)) for multi-label answers
      - ("span", normalized_string) for single answers
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = text.strip()
    if not t:
        return ("span", "")

    # First pass: atom normalize whole string
    t = _normalize_atom(t)

    # Early return for simple yes/no
    if t in {"yes", "no"}:
        return ("span", t)

    # Split into candidates
    parts = _split_multi(t)
    parts = [_normalize_atom(p) for p in parts if p]

    # Sided organ expansion (only if looks like multi)
    if len(parts) > 1:
        parts = _expand_sided_organs(parts)

    # dedupe empties, sort for order-invariance
    parts = [p for p in parts if p]
    uniq = sorted(set(parts))

    if len(uniq) <= 1:
        return ("span", uniq[0] if uniq else t)

    return ("set", tuple(uniq))

def _extract_final_answer(text: str) -> str:
    """
    More robust final answer extraction:
    - prefer 'final answer|answer|prediction|=>|Ans' patterns (last occurrence)
    - otherwise, if a trailing yes/no exists, take it
    - otherwise, take the last short span (<=5 tokens) from the last non-empty line
    """
    if not text:
        return ""
    # common markers
    pat = re.compile(
        r"(?:final\s*answer|answer|prediction|predicted\s*answer|=>|ans(?:wer)?)[\s:：\-]*([^\n]+)",
        flags=re.IGNORECASE
    )
    m_all = list(pat.finditer(text))
    if m_all:
        cand = m_all[-1].group(1).strip()
        return cand

    # try to fish out a yes/no near the end
    tail = text.strip().lower().splitlines()[-1] if text.strip() else ""
    yn = re.search(r"\b(yes|no|true|false)\b", tail)
    if yn:
        return yn.group(1)

    # fallback: last short span (<= 5 tokens) from the last non-empty line
    for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
        toks = line.split()
        if 1 <= len(toks) <= 5:
            return line
    return text.strip()

# ---------------------------
# Image resolver (imgs.zip → imgs/)
# ---------------------------
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
    if not repo_id:
        eval_logger.error("SLAKE resolver: empty repo_id")
        return ""
    if snapshot_download is None:
        eval_logger.error("Please install huggingface_hub: pip install -U huggingface_hub")
        return ""
    snap = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["imgs/*", "imgs.zip"],
        resume_download=True,
    )
    imgs_dir = os.path.join(snap, "imgs")
    if _dir_nonempty(imgs_dir):
        return imgs_dir
    zp = os.path.join(snap, "imgs.zip")
    if os.path.isfile(zp):
        out = _extract_zip_once(zp, imgs_dir)
        if out and _dir_nonempty(out):
            return out
    eval_logger.error(f"SLAKE resolver: imgs/ not found in snapshot {snap}")
    return ""

# ---------------------------
# Task adapters
# ---------------------------
def slake_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """Return question text only for English items; otherwise return empty string (skipped)."""
    if not _is_en(doc):
        return ""
    pre = post = ""
    if lmms_eval_specific_kwargs:
        pre = lmms_eval_specific_kwargs.get("pre_prompt", "") or ""
        post = lmms_eval_specific_kwargs.get("post_prompt", "") or ""
    q = (doc.get("question") or "").strip()
    text = f"{pre}{q}\n{post}".strip()

    # Optional: log the prompt we’re about to send (handy when DEBUG_PATH is set)
    _debug_log({
        "task": "slake",
        "stage": "doc_to_text",
        "qid": str(doc.get("qid", "")),
        "q_lang": doc.get("q_lang"),
        "question": q,
        "prompt_text": text,
    })

    return text

def slake_doc_to_target(doc: Dict[str, Any]) -> str:
    return (doc.get("answer") or "").strip()

def slake_doc_to_visual(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None):
    """Resolve 'img_name' under imgs/; non-English docs will be ignored upstream."""
    repo_id = ""
    if lmms_eval_specific_kwargs:
        repo_id = lmms_eval_specific_kwargs.get("dataset_repo") or lmms_eval_specific_kwargs.get("dataset_path") or ""
    if not repo_id:
        repo_id = "BoKelvin/SLAKE"
    root = _slake_image_root_for_repo(repo_id)
    if not root:
        raise RuntimeError("SLAKE images unavailable; ensure network/HF token and disk space.")
    rel = str(doc.get("img_name") or "").strip().lstrip("/\\")
    if not rel:
        return []
    path = os.path.join(root, rel)
    if not os.path.isfile(path):
        base = os.path.basename(rel)
        for r, _d, files in os.walk(root):
            if base in files:
                path = os.path.join(r, base)
                break
    if not os.path.isfile(path):
        eval_logger.warning(f"SLAKE: image not found for {rel}")
        return []
    try:
        with Image.open(path) as im:
            return [im.convert("RGB")]
    except Exception as e:
        eval_logger.warning(f"SLAKE: failed to open {path}: {e}")
        return []

def slake_process_results(
    doc: Dict[str, Any],
    results: List[str],
    lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None
):
    """Emit accuracy only for English items; non-English return {} (ignored by aggregator)."""
    if not _is_en(doc):
        return {}

    # Full model output (raw) and extracted segment (before parsing/normalization)
    pred_raw = (results[0] if results else "") or ""
    pred_extracted = _extract_final_answer(pred_raw)

    # Normalize to span-or-set
    pred_kind, pred_norm_obj = _normalize_to_either_span_or_set(pred_extracted)
    tgt_kind,  tgt_norm_obj  = _normalize_to_either_span_or_set(slake_doc_to_target(doc))

    # equality: set == set by set equality; span == span by string equality;
    # span vs set(1) allowed when the single element matches
    def _eq(a_kind: str, a_obj: Union[str, Tuple[str, ...]],
            b_kind: str, b_obj: Union[str, Tuple[str, ...]]) -> bool:
        if a_kind == "set" and b_kind == "set":
            return set(a_obj) == set(b_obj)
        if a_kind == "span" and b_kind == "span":
            return a_obj == b_obj and a_obj != ""
        if a_kind == "span" and b_kind == "set" and len(b_obj) == 1:
            return a_obj == next(iter(b_obj))
        if a_kind == "set" and b_kind == "span" and len(a_obj) == 1:
            return next(iter(a_obj)) == b_obj
        return False

    score = 1.0 if _eq(pred_kind, pred_norm_obj, tgt_kind, tgt_norm_obj) else 0.0

    # stringify normalized forms for logging
    def _to_str(kind: str, obj: Union[str, Tuple[str, ...]]) -> str:
        if kind == "set":
            return " | ".join(obj)  # stable, sorted order
        return obj

    pred_norm_str = _to_str(pred_kind, pred_norm_obj)
    tgt_norm_str  = _to_str(tgt_kind,  tgt_norm_obj)

    qid = str(doc.get("qid", "")) or f"{doc.get('img_name','')}::{doc.get('question','')}"[:256]

    # Rich debug log (initial logger requested fields included)
    _debug_log({
        "task": "slake",
        "stage": "process_results",
        "qid": qid,
        "q_lang": doc.get("q_lang"),
        "question": doc.get("question",""),
        "target_raw": doc.get("answer",""),
        "target_norm": tgt_norm_str,
        "pred_raw": pred_raw,              # full model output
        "pred_extracted": pred_extracted,  # answer before parsing to set/span
        "pred_norm": pred_norm_str,        # final normalized form
        "score": score,
    })

    return {"accuracy": {"question_id": qid, "score": score}}

def slake_aggregate_results(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    total = 0.0
    count = 0
    for r in results:
        acc = r.get("accuracy", {}).get("score", None)
        if acc is None:
            acc = r.get("score", None)
        if acc is None:
            continue
        total += float(acc)
        count += 1
    pct = (total / count) * 100.0 if count else 0.0
    eval_logger.info(f"SLAKE Accuracy: {pct:.2f}")
    return pct