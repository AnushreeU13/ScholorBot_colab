import numpy as np

from embedding_utils import MedCPTDualEmbedder
from storage_utils_local import create_faiss_store

try:
    from config_local_v2 import FAISS_INDICES_DIR, KB_GUIDELINES, KB_DRUGLABELS
except Exception:
    FAISS_INDICES_DIR = "faiss_indices"
    KB_GUIDELINES = "kb_guidelines_medcpt"
    KB_DRUGLABELS = "kb_druglabels_medcpt"

embedder = MedCPTDualEmbedder()
dim = embedder.dim

kb_guidelines = create_faiss_store(KB_GUIDELINES, dim, base_dir=FAISS_INDICES_DIR)
kb_drugs = create_faiss_store(KB_DRUGLABELS, dim, base_dir=FAISS_INDICES_DIR)

def test(kb, query):
    q = embedder.embed_query(query).astype(np.float32)
    sims, idxs, metas = kb.search(q, k=5)
    top = float(sims[0][0]) if sims is not None and len(sims) else None
    print("\nQ:", query)
    print("Top sim:", top)
    for i, m in enumerate(metas[:3]):
        sec = m.get("section") or m.get("section_title") or m.get("section_group") or ""
        doc = m.get("document_name") or m.get("document") or m.get("title") or ""
        print(f"  {i+1}. {m.get('source_type')} | {doc} | {sec}")

test(kb_drugs, "What are common adverse reactions of isoniazid?")
test(kb_guidelines, "How is community acquired pneumonia diagnosed?")
