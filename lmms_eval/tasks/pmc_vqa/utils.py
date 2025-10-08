from __future__ import annotations

import os
import re
import zipfile
from functools import lru_cache
from typing import List, Dict, Any, Optional

from PIL import Image
from loguru import logger as eval_logger
try:
    from huggingface_hub import snapshot_download
except:
    snapshot_download = None

CHOICE_LETTERS = ["A", "B", "C", "D"]

# ------------- PMC-VQA helpers -------------

def _dir_nonempty(path: str) -> bool:
    return os.path.isdir(path) and bool(os.listdir(path))


def _normalize_leaf_folder(folder: str) -> str:
    """
    Some zips contain an outer folder named 'images' and an inner 'images' again.
    If we extracted to .../images but the real files are in .../images/images/*,
    dive one level to the leaf that actually has files.
    Preference order: images2/ then images/ (if both exist).
    """
    if not os.path.isdir(folder):
        return folder

    # Prefer explicit leafs if present inside this folder
    for d in ("images2", "images"):
        p = os.path.join(folder, d)
        if _dir_nonempty(p):
            return p

    # If the folder itself is already non-empty, keep it
    if _dir_nonempty(folder):
        return folder

    # Try .../images inside it as a last resort (even if empty check above failed)
    inner = os.path.join(folder, "images")
    if _dir_nonempty(inner):
        return inner

    return folder


def _extract_zip(zp: str, out_dir: str) -> Optional[str]:
    """
    Extract once; if already extracted and non-empty, reuse.
    Normalize to the leaf folder that actually holds the files.
    """
    try:
        if _dir_nonempty(out_dir):
            return _normalize_leaf_folder(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(zp) as zf:
            eval_logger.info(f"Extracting {os.path.basename(zp)} -> {out_dir} (one-time)...")
            zf.extractall(out_dir)
        return _normalize_leaf_folder(out_dir)
    except Exception as e:
        eval_logger.warning(f"Failed to extract {zp}: {e}")
        return None


@lru_cache(maxsize=16)
def _resolve_image_root_for_repo(repo_id: str) -> str:
    """
    Find (or fetch) the images folder for the dataset repo.
    - Downloads to the standard HF cache (honors HF_HOME / HF_HUB_CACHE).
    - Prefers 'images2/' over 'images/'.
    - Auto-extracts 'images2.zip'/'images.zip' if needed.
    Returns an absolute path to the directory that contains the image files,
    or "" if it cannot be resolved.
    """
    if not repo_id:
        eval_logger.error("No repo_id provided to image resolver.")
        return ""

    if snapshot_download is None:
        eval_logger.error("huggingface_hub not installed. Run: pip install -U huggingface_hub")
        return ""

    # Download (or reuse cached) snapshot.
    snap_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["images2/*", "images/*", "images2.zip", "images.zip"],
        resume_download=True,
    )

    # Prefer already-extracted folders
    for d in ("images2", "images"):
        p = os.path.join(snap_dir, d)
        if _dir_nonempty(p):
            return _normalize_leaf_folder(p)

    # Otherwise extract zips once
    for zname, outdir in (("images2.zip", "images2"), ("images.zip", "images")):
        zp = os.path.join(snap_dir, zname)
        if os.path.isfile(zp):
            out = _extract_zip(zp, os.path.join(snap_dir, outdir))
            if out and _dir_nonempty(out):
                return _normalize_leaf_folder(out)

    eval_logger.error(f"Could not locate images under snapshot for {repo_id}.")
    return ""


def _find_file(root: str, filename: str) -> Optional[str]:
    """
    Robust filename location:
      1) Try common direct paths.
      2) If not found, walk under root and return the first matching basename.
    """
    candidates = [
        os.path.join(root, filename),                    # .../images/<file>
        os.path.join(root, "images", filename),          # .../images/images/<file>
        os.path.join(root, "images2", filename),         # .../images/images2/<file> (rare)
        os.path.join(root, "train", filename),
        os.path.join(root, "test", filename),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    # Fallback: search (first match wins)
    for r, _dirs, files in os.walk(root):
        if filename in files:
            return os.path.join(r, filename)
    return None

# ------------- PMC-VQA adapters -------------

def _collect_choices(doc: Dict[str, Any]) -> List[str]:
    # Column names may contain spaces exactly as in CSV: 'Choice A', 'Choice B', etc.
    choices: List[str] = []
    for letter in CHOICE_LETTERS:
        for key in (f"Choice {letter}", f"Choice_{letter}", f"choice_{letter}"):
            if key in doc and doc[key] is not None:
                raw = doc[key]
                if isinstance(raw, str):
                    # Some entries look like 'A:Something'
                    parts = raw.split(":", 1)
                    if len(parts) == 2 and len(parts[0]) <= 3:  # naive 'A:'/'B:' prefix
                        raw = parts[1].strip()
                choices.append(str(raw).strip())
                break
    return choices


def pmc_vqa_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {"pre_prompt": "", "post_prompt": ""}

    pre = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc.get("Question", doc.get("question", "")).strip()
    choices = _collect_choices(doc)
    formatted = "\n".join(f"{letter}. {text}" for letter, text in zip(CHOICE_LETTERS, choices))
    return f"{pre}{question}\n{formatted}\n{post}"


def pmc_vqa_doc_to_target(doc: Dict[str, Any]) -> str:
    label = doc.get("Answer_label") or doc.get("Answer_Label") or doc.get("answer_label")
    if isinstance(label, str):
        label = label.strip()
        if label in CHOICE_LETTERS:
            return label

    # Fallback: match Answer text against the choices
    answer_text = (doc.get("Answer") or doc.get("answer") or "").strip().lower()
    choices = _collect_choices(doc)
    for idx, choice in enumerate(choices):
        if choice.strip().lower() == answer_text:
            return CHOICE_LETTERS[idx]

    # As a last resort, return the raw answer text (may surface mismatches)
    return (doc.get("Answer") or doc.get("answer") or "").strip()


def _extract_choice_letter(text: str) -> str:
    """
    Robustly extract the chosen letter from model output.
    Prefer the LAST 'Answer: X' (A-D); otherwise try a trailing bare letter.
    """
    if not text:
        return ""
    # 1) Prefer explicit pattern "Answer: X"
    matches = list(re.finditer(r"answer\s*[:\-]\s*([ABCD])", text, flags=re.IGNORECASE))
    if matches:
        return matches[-1].group(1).upper()
    # 2) Look for a single letter at the very end (e.g., "... ) C")
    m2 = re.search(r"\b([ABCD])\b[^\w]*$", text.strip(), flags=re.IGNORECASE)
    if m2:
        return m2.group(1).upper()
    return ""


def pmc_vqa_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Return per-item structured accuracy dict:
      {"accuracy": {"question_id": <id>, "score": 0.0|1.0}}
    """
    pred_raw = results[0] if results else ""
    pred = (pred_raw or "").strip()
    target = pmc_vqa_doc_to_target(doc).strip().upper()

    normalized = _extract_choice_letter(pred)
    score = 1.0 if normalized and (normalized == target) else 0.0

    qid = f"{doc.get('Figure_path', '')}::{doc.get('Question', doc.get('question','')).strip()}"[:256]
    return {"accuracy": {"question_id": qid, "score": score}}


def pmc_vqa_aggregate_results(results: List[Dict[str, Any]]) -> float:
    """
    Aggregate a list of {"question_id":..., "score": 0/1} into overall accuracy (%).
    """
    if not results:
        return 0.0
    total = 0.0
    count = 0
    for r in results:
        if isinstance(r, dict):
            total += float(r.get("score", 0.0))
            count += 1
    acc = (total / count) * 100.0 if count else 0.0
    eval_logger.info(f"PMC-VQA Accuracy: {acc:.2f}")
    return acc


def pmc_vqa_doc_to_visual(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> List[Image.Image]:
    """
    Load the associated image as a PIL.Image.
    Strategy:
      1) Download/resolve images for the repo indicated by YAML (lmms_eval_specific_kwargs).
      2) Locate the file by the basename of Figure_path.
      3) Return [Image] or raise a clear error if not found (to avoid cryptic model errors).
    """
    # 1) Determine repo id from YAML-provided kwargs
    repo_id = ""
    if lmms_eval_specific_kwargs:
        repo_id = (lmms_eval_specific_kwargs.get("dataset_repo")
                   or lmms_eval_specific_kwargs.get("dataset_path") or "")
    if not repo_id:
        repo_id = "RadGenome/PMC-VQA"  # final fallback

    root = _resolve_image_root_for_repo(repo_id)
    if not root:
        raise RuntimeError(
            f"Images folder could not be resolved/downloaded for '{repo_id}'. "
            "Check network and consider running 'hf auth login'."
        )

    # 2) Map figure path to a local file by basename
    fig_key = next((k for k in ("Figure_path", "figure_path", "Figure_Path", "image_path") if k in doc), None)
    if not fig_key:
        raise RuntimeError("No figure path key found in document.")
    filename = os.path.basename(str(doc[fig_key]))

    path = _find_file(root, filename)
    if not path:
        # Provide an actionable error instead of letting the processor crash later.
        raise RuntimeError(
            f"Image file '{filename}' not found under '{root}'. "
            "Verify that images.zip/images2.zip were downloaded correctly."
        )

    try:
        with Image.open(path) as im:
            return [im.convert("RGB")]
    except Exception as e:
        eval_logger.warning(f"Failed to open image {path}: {e}")
        return []
