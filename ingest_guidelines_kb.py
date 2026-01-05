"""
ingest_guidelines_kb.py

Ingest Tier-1 clinical guidelines into a unified FAISS index (kb_guidelines_medcpt).

Publication-grade upgrades:
1) Filter out references/bibliography-like sections (by section title).
2) Filter out reference-list-like text blocks (by conservative heuristics).
3) Strengthen metadata fields for downstream routing, citations, and guardrails.
4) IMPORTANT: guidelines use smaller chunk size (250–400 tokens) for higher retrieval precision.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
import torch

from storage_utils import create_faiss_store
from chunking_utils import chunk_document
from pdf_utils import extract_text_by_page
from config import (
    FAISS_INDICES_DIR,
    GUIDELINE_CHUNK_SIZE,
    GUIDELINE_CHUNK_OVERLAP,
)

# Prefer KB_RAW_DIR / KB_PROCESSED_DIR if present
try:
    from config import KB_RAW_DIR, KB_PROCESSED_DIR
except Exception:
    from config import BOX_FOLDER
    KB_RAW_DIR = str(Path(BOX_FOLDER) / "KB_raw")
    KB_PROCESSED_DIR = str(Path(BOX_FOLDER) / "KB_processed")

from embedding_utils import MedCPTEmbedder

import config

GUIDELINE_CHUNK_SIZE = getattr(config, "GUIDELINE_CHUNK_SIZE", 240)
GUIDELINE_CHUNK_OVERLAP = getattr(config, "GUIDELINE_CHUNK_OVERLAP", 50)
FAISS_INDICES_DIR = config.FAISS_INDICES_DIR
PROJECT_ROOT = config.PROJECT_ROOT

# -----------------------------
# Paths / Store name
# -----------------------------
KB_RAW = Path(KB_RAW_DIR)
KB_PROCESSED = Path(KB_PROCESSED_DIR)
FAISS_DIR = Path(FAISS_INDICES_DIR)

STORE_NAME = "kb_guidelines_medcpt"

GUIDELINES_DIR = KB_RAW / "guidelines"
OUT_DIR = KB_PROCESSED / "guidelines_text"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CHUNKS_JSONL = OUT_DIR / "guidelines_chunks.jsonl"
CDC_JSONL_PATH = GUIDELINES_DIR / "cdc_tb_pages.jsonl"


# -----------------------------
# Documents to ingest (PDFs)
# -----------------------------
PDF_GUIDELINES = [
    {
        "organization": "WHO",
        "disease": "tuberculosis",
        "filename": "WHO_TB_Consolidated_Guidelines.pdf",
        "audience": "clinician",
        "evidence_level": "guideline",
        "year": "2020",
    },
    {
        "organization": "ATS_IDSA",
        "disease": "pneumonia",
        "filename": "ATS_IDSA_CAP_Guideline_2019.pdf",
        "audience": "clinician",
        "evidence_level": "guideline",
        "year": "2019",
    },
]


# -----------------------------
# Utilities
# -----------------------------
def batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    bs = max(1, int(batch_size))
    for i in range(0, len(items), bs):
        yield items[i : i + bs]


def append_chunk_record(jsonl_path: Path, text: str, meta: Dict) -> None:
    rec = {"text": text, "metadata": meta}
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_checkpoint(jsonl_path: Path) -> Tuple[List[str], List[Dict]]:
    texts: List[str] = []
    metas: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            t = (rec.get("text") or "").strip()
            m = rec.get("metadata") or {}
            if not t:
                continue
            texts.append(t)
            metas.append(m)
    return texts, metas


# -----------------------------
# Heading heuristics (PDF -> sections)
# -----------------------------
def _looks_like_heading(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False

    if re.match(r"^\s*(\d+(\.\d+){0,3}|[IVX]+)\.?\s+[A-Za-z].{0,80}$", s):
        return True

    if s.isupper() and 4 <= len(s) <= 80:
        return True

    if re.match(r"^[A-Z][A-Za-z0-9 ,:\-\/\(\)]{0,80}$", s) and len(s) <= 80:
        if re.search(r"\b(19|20)\d{2}\b", s) and ("et al" in s.lower()):
            return False
        return True

    common_starts = (
        "Introduction", "Background", "Methods", "Diagnosis", "Treatment",
        "Management", "Recommendations", "Testing", "Therapy", "Follow-up",
        "Empiric", "Severity", "Risk", "Special", "Adverse", "Prevention"
    )
    if any(s.startswith(x) for x in common_starts) and len(s) <= 80:
        return True

    return False


# -----------------------------
# Reference / bibliography filtering
# -----------------------------
_REF_TITLE_PATTERNS = (
    "references", "reference", "bibliography", "literature cited", "works cited",
    "citations", "acknowledgements", "acknowledgments", "appendix", "supplement",
    "glossary", "abbreviations", "index"
)

def _is_reference_section_title(title: str) -> bool:
    t = (title or "").strip().lower()
    if not t:
        return False
    for p in _REF_TITLE_PATTERNS:
        if t == p or t.startswith(p + " ") or t.startswith(p + ":"):
            return True
    if re.search(r"(?i)^\s*\d{1,2}\s+references\b", t):
        return True
    return False

def _looks_like_reference_block(text: str) -> bool:
    s = (text or "")
    if len(s) < 400:
        return False
    low = s.lower()
    year_hits = len(re.findall(r"\b(19|20)\d{2}\b", s))
    etal_hits = low.count("et al")
    doi_hits = low.count("doi")
    url_hits = low.count("http://") + low.count("https://")

    if (year_hits >= 8 and etal_hits >= 2) or (etal_hits >= 5) or (doi_hits >= 2) or (url_hits >= 3):
        return True

    numbered_lines = sum(
        1 for ln in s.splitlines()
        if re.match(r"^\s*(\[\d+\]\s+|\d+\.\s+|\d+\)\s+)", ln.strip())
    )
    if numbered_lines >= 8 and year_hits >= 6:
        return True

    return False


def split_pdf_into_sections(pages: List[Tuple[str, int]]) -> List[Tuple[str, str, List[int]]]:
    sections: List[Tuple[str, str, List[int]]] = []

    current_title = "INTRODUCTION"
    current_lines: List[str] = []
    current_pages: List[int] = []

    for page_text, page_num in pages:
        for line in page_text.splitlines():
            if _looks_like_heading(line):
                if current_lines:
                    sec_text = "\n".join(current_lines).strip()
                    if sec_text:
                        sections.append((current_title, sec_text, sorted(set(current_pages))))
                current_title = line.strip()
                current_lines = []
                current_pages = []
            else:
                if line.strip():
                    current_lines.append(line)
                    current_pages.append(page_num)

    if current_lines:
        sec_text = "\n".join(current_lines).strip()
        if sec_text:
            sections.append((current_title, sec_text, sorted(set(current_pages))))

    cleaned: List[Tuple[str, str, List[int]]] = []
    for title, sec_text, pnums in sections:
        if _is_reference_section_title(title):
            continue
        if _looks_like_reference_block(sec_text):
            continue
        if len(sec_text) >= 250:
            cleaned.append((title, sec_text, pnums))

    return cleaned


# -----------------------------
# CDC JSONL reader (streaming)
# -----------------------------
def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# -----------------------------
# Main ingestion
# -----------------------------
def main():
    GUIDELINES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_CHUNKS_JSONL.exists():
        OUT_CHUNKS_JSONL.unlink()

    all_count = 0

    # Part 1: PDFs
    for doc in PDF_GUIDELINES:
        pdf_path = GUIDELINES_DIR / doc["filename"]
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing guideline PDF: {pdf_path}")

        print(f"\n[INFO] Reading PDF: {pdf_path.name}")

        pages = extract_text_by_page(str(pdf_path))
        print(f"  - Pages extracted: {len(pages)}")

        sections = split_pdf_into_sections(pages)
        print(f"  - Sections detected: {len(sections)}")

        for sec_title, sec_text, sec_pages in sections:
            if _is_reference_section_title(sec_title) or _looks_like_reference_block(sec_text):
                continue

            doc_name = f"{doc['organization']}::{doc['disease']}::{pdf_path.stem}::{sec_title}"

            # ✅ Use smaller guideline chunks (precision-first)
            chunks = chunk_document(
                text=sec_text,
                document_name=doc_name,
                page_number=None,
                chunk_size=int(GUIDELINE_CHUNK_SIZE),
                overlap=int(GUIDELINE_CHUNK_OVERLAP),
            )

            for ch in chunks:
                chunk_text = (ch.get("text") or "").strip()
                meta = ch.get("metadata", {}) or {}
                if not chunk_text:
                    continue

                meta.update({
                    "source_type": "clinical_guideline",
                    "doc_type": "clinical_guideline",
                    "source": "Guideline",
                    "file_name": pdf_path.name,
                    "section_title": sec_title,
                    "section_group": "guideline",
                    "is_reference": False,

                    "organization": doc["organization"],
                    "disease": doc["disease"],
                    "document": pdf_path.stem,
                    "section": sec_title,
                    "page_numbers": sec_pages,
                    "audience": doc.get("audience", "clinician"),
                    "evidence_level": doc.get("evidence_level", "guideline"),
                    "year": doc.get("year", ""),
                })

                append_chunk_record(OUT_CHUNKS_JSONL, chunk_text, meta)
                all_count += 1

    # Part 2: CDC JSONL
    if not CDC_JSONL_PATH.exists():
        print(f"\n[WARN] CDC JSONL not found at: {CDC_JSONL_PATH}")
    else:
        print(f"\n[INFO] Reading CDC JSONL: {CDC_JSONL_PATH.name}")

        loaded_pages = 0
        for rec in iter_jsonl(CDC_JSONL_PATH):
            loaded_pages += 1

            text = (rec.get("text") or "").strip()
            meta = rec.get("metadata") or {}
            if not text:
                continue
            if _looks_like_reference_block(text):
                continue

            topic = meta.get("topic", "cdc_tb")
            title = meta.get("title", topic)
            doc_name = f"CDC::tuberculosis::CDC_TB_WebGuidance::{topic}"

            chunks = chunk_document(
                text=text,
                document_name=doc_name,
                page_number=None,
                chunk_size=int(GUIDELINE_CHUNK_SIZE),
                overlap=int(GUIDELINE_CHUNK_OVERLAP),
            )

            for ch in chunks:
                chunk_text = (ch.get("text") or "").strip()
                ch_meta = ch.get("metadata", {}) or {}
                if not chunk_text:
                    continue

                ch_meta.update({
                    "source_type": "clinical_guideline",
                    "doc_type": "clinical_guideline",
                    "source": "Guideline",
                    "file_name": "cdc_tb_pages.jsonl",
                    "section_title": title,
                    "section_group": "guideline",
                    "is_reference": False,

                    "organization": "CDC",
                    "disease": "tuberculosis",
                    "document": "CDC_TB_WebGuidance",
                    "section": title,
                    "topic": topic,
                    "url": meta.get("url", ""),
                    "audience": meta.get("audience", "clinician"),
                    "evidence_level": meta.get("evidence_level", "guideline_web"),
                    "year": meta.get("year", ""),
                })

                append_chunk_record(OUT_CHUNKS_JSONL, chunk_text, ch_meta)
                all_count += 1

        print(f"  - CDC pages loaded: {loaded_pages}")

    print(f"\n[CHECKPOINT] Wrote guideline chunks to: {OUT_CHUNKS_JSONL}")
    print(f"[INFO] Total guideline chunks in checkpoint: {all_count:,}")

    if all_count == 0:
        print("[WARN] No guideline chunks produced.")
        return

    # Embed -> FAISS
    texts, metas = load_checkpoint(OUT_CHUNKS_JSONL)
    print(f"[INFO] Total guideline chunks to embed: {len(texts):,}")

    embedder = MedCPTEmbedder(device="cpu")
    dim = embedder.dim
    print(f"[OK] MedCPT embedding dimension: {dim}")

    store = create_faiss_store(
        store_name=STORE_NAME,
        dimension=dim,
        base_dir=str(FAISS_DIR),
    )

    torch.set_grad_enabled(False)

    EMB_BATCH_SIZE = 4
    DOC_MAX_LEN = 256

    print("[INFO] Embedding in batches (memory-safe)...")
    all_embs: List[np.ndarray] = []
    for i, batch_texts in enumerate(batched(texts, EMB_BATCH_SIZE), start=1):
        batch_embs = embedder.embed_texts(
            batch_texts,
            batch_size=EMB_BATCH_SIZE,
            max_length=DOC_MAX_LEN,
        ).astype(np.float32)
        all_embs.append(batch_embs)

        if i % 50 == 0:
            done = min(i * EMB_BATCH_SIZE, len(texts))
            print(f"  - embedded {done}/{len(texts)}")

    embeddings = np.vstack(all_embs).astype(np.float32)

    print("[INFO] Adding vectors to FAISS...")
    store.add_vectors(embeddings, metas)
    store.save()

    print("\n[DONE] Saved kb_guidelines_medcpt index + metadata")
    print(store.get_stats())
    print(f"[DONE] Checkpoint JSONL at: {OUT_CHUNKS_JSONL}")


if __name__ == "__main__":
    main()
