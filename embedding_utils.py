"""
embedding_utils.py

MedCPT dual-encoder embedder:
- Query embedding uses: ncbi/MedCPT-Query-Encoder
- Document embedding uses: ncbi/MedCPT-Article-Encoder
- Pooling: CLS token (last_hidden_state[:, 0, :])

This module is designed to be API-stable across scripts:
- embed_texts(...) accepts max_length for compatibility.
"""

from __future__ import annotations

from typing import Optional, List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


def _get_device(device: Optional[str] = None) -> str:
    """Return an available torch device string."""
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _cls_pool(last_hidden_state: torch.Tensor) -> torch.Tensor:
    """CLS pooling: take the first token representation."""
    return last_hidden_state[:, 0, :]


class MedCPTDualEmbedder:
    """
    Dual-encoder embedder for retrieval.
    - embed_query() uses MedCPT Query Encoder
    - embed_texts() uses MedCPT Article Encoder (for KB chunks)
    """

    def __init__(
        self,
        query_model_name: str = "ncbi/MedCPT-Query-Encoder",
        doc_model_name: str = "ncbi/MedCPT-Article-Encoder",
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        self.device = _get_device(device)
        self.max_length = int(max_length)

        print(f"[embedder] Loading MedCPT query encoder: {query_model_name} on {self.device}")
        self.q_tokenizer = AutoTokenizer.from_pretrained(query_model_name)
        self.q_model = AutoModel.from_pretrained(query_model_name).to(self.device)
        self.q_model.eval()

        print(f"[embedder] Loading MedCPT doc encoder: {doc_model_name} on {self.device}")
        self.d_tokenizer = AutoTokenizer.from_pretrained(doc_model_name)
        self.d_model = AutoModel.from_pretrained(doc_model_name).to(self.device)
        self.d_model.eval()

        self.dim = int(self.q_model.config.hidden_size)

    # ----------------------------
    # Query embedding
    # ----------------------------
    def embed_query(self, query: str, max_length: Optional[int] = None) -> np.ndarray:
        """Embed a single query with the Query Encoder."""
        if not isinstance(query, str) or not query.strip():
            return np.zeros((self.dim,), dtype=np.float32)

        ml = int(max_length) if max_length is not None else self.max_length

        inputs = self.q_tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=ml,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.q_model(**inputs)
            vec = _cls_pool(outputs.last_hidden_state)

        return vec.squeeze(0).detach().cpu().numpy().astype(np.float32)

    # ----------------------------
    # Document embedding (KB chunks)
    # ----------------------------
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 512,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of documents using the MedCPT document encoder.

        Args:
            texts: list of strings
            batch_size: embedding batch size
            max_length: tokenizer truncation length (explicit per-call control)
            show_progress: whether to print progress (lightweight)
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        bs = max(1, int(batch_size))
        ml = max(8, int(max_length))  # safety

        all_vecs = []
        n = len(texts)

        for i in range(0, n, bs):
            if show_progress and (i == 0 or i % (bs * 20) == 0):
                print(f"[embedder] Embedding batch {i//bs + 1}/{(n + bs - 1)//bs}")

            batch = texts[i : i + bs]
            inputs = self.d_tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=ml,  # ✅ use per-call max_length
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.d_model(**inputs)
                vecs = _cls_pool(outputs.last_hidden_state)

            all_vecs.append(vecs.detach().cpu().numpy().astype(np.float32))

        return np.vstack(all_vecs)


# Backward-compatible alias (older scripts import MedCPTEmbedder)
MedCPTEmbedder = MedCPTDualEmbedder
