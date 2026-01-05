"""
patch_metadata_add_text.py

Fast, non-reembedding fix.

Problem:
- FAISSStore stores ONLY metadata JSON aligned with vectors.
- Your retrieval expects chunk text inside metadata["text"] (or ["chunk_text"]).
- If metadata was created without text, you can get similarities but 0 usable chunks.

This script patches FAISS metadata JSON by reading the original checkpoint JSONL
that contains {"text": ..., "metadata": {...}} in the same order as vectors
were added to FAISS.

It does NOT touch the .index file.
It overwrites metadata_json (creates a .bak first).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_jsonl", required=True, help="JSONL with {'text':..., 'metadata':...}")
    ap.add_argument("--metadata_json", required=True, help="FAISS metadata json to patch")
    ap.add_argument("--text_key", default="text", help="Key name to store text into metadata (default: text)")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint_jsonl)
    meta_path = Path(args.metadata_json)

    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint_jsonl not found: {ckpt}")
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata_json not found: {meta_path}")

    metadata_list = json.loads(meta_path.read_text(encoding="utf-8"))

    # Backup
    bak = meta_path.with_suffix(meta_path.suffix + ".bak")
    if not bak.exists():
        bak.write_text(json.dumps(metadata_list, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Backup created: {bak}")

    texts = []
    with ckpt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            texts.append((rec.get("text") or "").strip())

    if len(texts) != len(metadata_list):
        raise ValueError(
            f"Length mismatch: checkpoint texts={len(texts)} vs metadata entries={len(metadata_list)}.\n"
            "This must match the exact order/size used when vectors were added."
        )

    patched = 0
    for i in range(len(metadata_list)):
        if args.text_key not in metadata_list[i] or not str(metadata_list[i].get(args.text_key) or "").strip():
            metadata_list[i][args.text_key] = texts[i]
            patched += 1

    meta_path.write_text(json.dumps(metadata_list, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] Patched metadata: {meta_path}")
    print(f"[STATS] patched_entries={patched} total_entries={len(metadata_list)}")


if __name__ == "__main__":
    main()
