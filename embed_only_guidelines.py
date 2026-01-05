"""
embed_only_guidelines.py

Rebuild the FAISS index for guideline chunks from an existing JSONL checkpoint.
This script DOES NOT re-parse PDFs.

Input checkpoint (produced by ingest_guidelines_kb.py):
  datasets/KB_processed/guidelines_text/guidelines_chunks.jsonl

Output:
  faiss_indices/kb_guidelines_medcpt.index
  faiss_indices/kb_guidelines_medcpt_metadata.json

Key fixes:
- Flatten nested metadata: JSONL records are typically {"text": ..., "metadata": {...}}.
  The RAG pipeline expects fields like organization/document/section at the top level.
- Keep chunk text inside metadata ("text") so the RAG pipeline can build EVIDENCE context.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np

from config import PROJECT_ROOT, FAISS_INDICES_DIR
from embedding_utils import MedCPTDualEmbedder
from storage_utils import create_faiss_store

STORE_NAME = "kb_guidelines_medcpt"
#PROJECT_ROOT = Path(BOX_FOLDER)

CHUNKS_JSONL = (
    PROJECT_ROOT
    / "datasets"
    / "KB_processed"
    / "guidelines_text"
    / "guidelines_chunks.jsonl"
)

BATCH_SIZE = 64


def load_jsonl(path: Path) -> Iterator[Dict]:
    """Stream JSONL records."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # Skip malformed lines rather than crashing the rebuild.
                continue


def flatten_meta(rec: Dict, text: str, ingested_at: str) -> Dict:
    """Flatten nested metadata into a single dict."""
    base_meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}

    meta: Dict = dict(base_meta)  # flatten
    meta["text"] = text

    # Preserve a few helpful fields if they exist.
    for k in ["doc_id", "source_id", "url", "title", "topic", "page_numbers"]:
        if k in rec and rec.get(k) is not None and k not in meta:
            meta[k] = rec[k]

    meta.setdefault("organization", meta.get("organization", "Guideline"))
    meta.setdefault("document", meta.get("document", meta.get("title", meta.get("url", "UnknownDoc"))))
    meta.setdefault("section", meta.get("section", meta.get("topic", "UnknownSection")))
    meta.setdefault("ingested_at", ingested_at)
    return meta


def main():
    if not CHUNKS_JSONL.exists():
        raise FileNotFoundError(f"Missing checkpoint JSONL: {CHUNKS_JSONL}")

    ingested_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    embedder = MedCPTDualEmbedder()
    store = create_faiss_store(STORE_NAME, dimension=embedder.dim, base_dir=str(FAISS_INDICES_DIR))

    buffer_texts: List[str] = []
    buffer_metas: List[Dict] = []
    count = 0

    def flush():
        nonlocal buffer_texts, buffer_metas
        if not buffer_texts:
            return
        vecs = embedder.embed_texts(buffer_texts)
        vecs = np.asarray(vecs, dtype=np.float32)
        store.add_vectors(vecs, buffer_metas)
        buffer_texts = []
        buffer_metas = []

    for rec in load_jsonl(CHUNKS_JSONL):
        text = (rec.get("text") or "").strip()
        if len(text) < 80:
            continue

        meta = flatten_meta(rec, text=text, ingested_at=ingested_at)

        buffer_texts.append(text)
        buffer_metas.append(meta)

        if len(buffer_texts) >= BATCH_SIZE:
            flush()
            count += BATCH_SIZE
            if count % 1000 == 0:
                print(f"[INFO] Embedded+indexed: {count}")

    flush()

    store.save()
    stats = store.get_stats()
    print("[DONE] Saved kb_guidelines_medcpt index + metadata")
    print(stats)


if __name__ == "__main__":
    main()
