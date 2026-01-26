"""
user_ingest_local.py

On-demand ingestion of a user PDF into FAISS (KB_USER_FACT).
Local-first and portable.

Usage:
  python user_ingest_local.py --pdf path/to/file.pdf --doc_name mydoc

Outputs:
  - checkpoint jsonl under datasets/KB_processed/user_fact/
  - updates FAISS index under faiss_indices/
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from config_local_v2 import (
    FAISS_INDICES_DIR, KB_USER_FACT,
    CHUNK_SIZE, OVERLAP,
    KB_PROCESSED_DIR,
)
from embedding_utils import MedCPTDualEmbedder
from storage_utils_local import create_faiss_store
from pdf_utils import extract_text_by_page
from chunking_utils_v2 import chunk_document


def ingest_user_pdf(
    pdf_path: str | Path,
    doc_name: str,
    store_name: str = KB_USER_FACT,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
    embed_batch_size: int = 16,
) -> Dict:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    embedder = MedCPTDualEmbedder()
    dim = embedder.dim
    store = create_faiss_store(store_name, dim, base_dir=FAISS_INDICES_DIR)

    # Read pages
    pages = extract_text_by_page(str(pdf_path))
    chunk_texts: List[str] = []
    chunk_metas: List[Dict] = []

    for page_num, page_text in pages:
        chunks = chunk_document(
            text=page_text,
            document_name=doc_name,
            page_number=page_num,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for ch in chunks:
            t = (ch.get("text") or "").strip()
            if len(t) < 120:
                continue

            meta = (ch.get("metadata") or {})
            meta.update({
                "source_type": "user_pdf",
                "organization": "user",
                "document_name": doc_name,
                "page_number": page_num,
                # CRITICAL: keep the chunk text in metadata for downstream evidence assembly
                "text": t,
                "ingested_at": datetime.utcnow().isoformat(),
                "embed_model": embedder.name,
            })

            chunk_texts.append(t)
            chunk_metas.append(meta)

    # Save checkpoint
    out_dir = (KB_PROCESSED_DIR / "user_fact").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{doc_name}_chunks.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for t, m in zip(chunk_texts, chunk_metas):
            f.write(json.dumps({"text": t, "metadata": m}, ensure_ascii=False) + "\n")

    # Embed + add
    for i in range(0, len(chunk_texts), embed_batch_size):
        batch_texts = chunk_texts[i:i + embed_batch_size]
        batch_metas = chunk_metas[i:i + embed_batch_size]
        embs = embedder.embed_texts(batch_texts, batch_size=len(batch_texts)).astype(np.float32)
        store.add_vectors(embs, batch_metas)

    store.save()
    stats = store.get_stats()
    stats["checkpoint_jsonl"] = str(out_jsonl)
    stats["added_chunks"] = len(chunk_texts)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--doc_name", required=True)
    args = ap.parse_args()

    stats = ingest_user_pdf(args.pdf, args.doc_name)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
