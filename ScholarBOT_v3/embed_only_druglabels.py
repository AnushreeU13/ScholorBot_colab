"""
embed_only_druglabels.py

Rebuild FAISS index for DailyMed SPL drug labels from an existing JSONL checkpoint.
This script DOES NOT rescan or rechunk XML files.

Input checkpoint:
  datasets/KB_processed/druglabels_text/druglabels_chunks.jsonl

Output:
  faiss_indices/kb_druglabels_medcpt.index
  faiss_indices/kb_druglabels_medcpt_metadata.json

Key fix:
- Keep chunk text inside metadata ("text") so the RAG pipeline can build evidence context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List
from datetime import datetime

import numpy as np
import faiss

from config import PROJECT_ROOT, FAISS_INDICES_DIR
from embedding_utils import MedCPTEmbedder
from storage_utils import create_faiss_store

STORE_NAME = "kb_druglabels_medcpt"
PROJECT_ROOT = Path(PROJECT_ROOT)

CHUNKS_JSONL = (
    PROJECT_ROOT
    / "datasets"
    / "KB_processed"
    / "druglabels_text"
    / "druglabels_chunks.jsonl"
)

def load_jsonl(path: Path) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def main():
    if not CHUNKS_JSONL.exists():
        raise FileNotFoundError(f"Missing checkpoint JSONL: {CHUNKS_JSONL}")

    # Use Article Encoder for document chunks
    embedder = MedCPTEmbedder()
    dim = embedder.dim
    print(f"[OK] MedCPT embedding dimension: {dim}")

    store = create_faiss_store(STORE_NAME, dimension=dim, base_dir=FAISS_INDICES_DIR)

    # Reset store to avoid mixing old vectors/metadata
    store.index = faiss.IndexFlatIP(dim)
    store.metadata = []

    # We normalize vectors for cosine similarity via inner product
    # (storage_utils.add_vectors should already normalize; keep consistent)
    BATCH_SIZE = 16

    buffer_texts: List[str] = []
    buffer_meta: List[Dict] = []

    ingested_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def flush():
        nonlocal buffer_texts, buffer_meta
        if not buffer_texts:
            return

        embs = embedder.embed_texts(buffer_texts, batch_size=BATCH_SIZE).astype(np.float32)
        store.add_vectors(embs, buffer_meta)

        buffer_texts = []
        buffer_meta = []

    count = 0
    for rec in load_jsonl(CHUNKS_JSONL):
        text = (rec.get("text") or "").strip()
        if len(text) < 80:
            continue

        # IMPORTANT:
        # Keep chunk text in metadata so RAG can build evidence context later.
        meta = dict(rec)
        meta["text"] = text  # ensure key exists and is non-empty

        # Optional provenance fields for reproducibility
        meta.setdefault("source_id", meta.get("virtual_xml_path", ""))
        meta.setdefault("retrieved_at", ingested_at)
        meta.setdefault("embed_model", "MedCPT-Article-Encoder")

        buffer_texts.append(text)
        buffer_meta.append(meta)

        if len(buffer_texts) >= BATCH_SIZE:
            flush()
            count += BATCH_SIZE
            if count % 1000 == 0:
                print(f"[INFO] Embedded+indexed: {count}")

    flush()

    store.save()
    stats = store.get_stats()
    print("[DONE] Saved kb_druglabels_medcpt index + metadata")
    print(stats)

if __name__ == "__main__":
    main()
