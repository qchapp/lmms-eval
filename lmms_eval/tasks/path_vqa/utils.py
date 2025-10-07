from __future__ import annotations
import re
from typing import Dict, Any, List, Optional
from PIL import Image
from loguru import logger as eval_logger

def _pathvqa_norm(s: str) -> str:
    """Lower/strip + tiny yes/no aliasing + light punctuation cleanup."""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.strip().lower()
    y = {"yes", "y", "yeah", "yep", "true", "1"}
    n = {"no", "n", "nope", "false", "0"}
    if s in y: return "yes"
    if s in n: return "no"
    # keep digits/letters/space and a few medical-friendly chars
    s = re.sub(r"[^a-z0-9\s.%/-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def pathvqa_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """PathVQA has fields: image (PIL), question (str), answer (str)."""
    pre = post = ""
    if lmms_eval_specific_kwargs:
        pre = lmms_eval_specific_kwargs.get("pre_prompt", "") or ""
        post = lmms_eval_specific_kwargs.get("post_prompt", "") or ""
    q = (doc.get("question") or "").strip()
    return f"{pre}{q}\n{post}".strip()

def pathvqa_doc_to_target(doc: Dict[str, Any]) -> str:
    return (doc.get("answer") or "").strip()

def pathvqa_doc_to_visual(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> List[Image.Image]:
    """Images are embedded directly in the dataset."""
    im = doc.get("image")
    if im is None:
        eval_logger.warning("PathVQA: sample has no image")
        return []
    try:
        return [im.convert("RGB")]
    except Exception as e:
        eval_logger.warning(f"PathVQA: failed to convert image: {e}")
        return []

def pathvqa_process_results(doc: Dict[str, Any], results: List[str], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None):
    """Exact-match accuracy with light normalization. Accept 'Answer: ...' or raw text."""
    pred_raw = (results[0] if results else "") or ""
    # Prefer the last explicit "Answer: ..." tag if present
    m = list(re.finditer(r"answer\s*[:\-]\s*(.+)", pred_raw, flags=re.IGNORECASE))
    pred = m[-1].group(1).strip() if m else pred_raw.strip()

    p = _pathvqa_norm(pred)
    t = _pathvqa_norm(pathvqa_doc_to_target(doc))
    score = 1.0 if (p == t and p != "") else 0.0

    qid = f"{doc.get('question','')[:200]}::{doc.get('answer','')[:40]}"
    return {"accuracy": {"question_id": qid, "score": score}}

def pathvqa_aggregate_results(results: List[Dict[str, Any]]) -> float:
    """Aggregate accuracy (%) over emitted per-item dicts."""
    if not results:
        return 0.0
    total = 0.0
    count = 0
    for r in results:
        if not isinstance(r, dict):
            continue
        acc = r.get("accuracy", {}).get("score", None)
        if acc is None:
            # some runners pass already-flattened dicts
            acc = r.get("score", None)
        if acc is None:
            continue
        total += float(acc)
        count += 1
    pct = (total / count) * 100.0 if count else 0.0
    eval_logger.info(f"PathVQA Accuracy: {pct:.2f}")
    return pct
