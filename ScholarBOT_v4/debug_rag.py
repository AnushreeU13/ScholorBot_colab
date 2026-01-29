import os
import sys
from pathlib import Path

# Mock streamlit environment variables if needed
# os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["SCHOLARBOT_ZERO_HALLUCINATION"] = "0" 

# Add current dir to path
sys.path.append(os.getcwd())

from aligned_backend import AlignedScholarBotEngine

print("Initializing Engine...")
engine = AlignedScholarBotEngine(verbose=True)

query = "What are the side effects of Rifampin?"
# Inspect internal retrieval manually
from rag_pipeline_aligned import RAGPipeline
print("\n--- Manual RAG Trace (Top 20) ---")
q_vec = engine.embedder.embed_query(query)
# Guidelines store
store = engine.kb_guidelines
results = store.similarity_search_with_score_by_vector(q_vec, k=20)

for i, (doc, score) in enumerate(results):
    sim = 1.0 - (score / 2.0) # Assume L2
    if sim > 1.0: sim = score # fallback IP
    
    snippet = doc.page_content[:100].replace("\n", " ")
    meta = doc.metadata
    sec = meta.get("section_title", "N/A")
    sec_group = meta.get("section_group", "N/A")
    
    print(f"[{i+1}] Sim:{sim:.4f} | Sec:{sec} | Group:{sec_group} | Text: {ascii(snippet)}...")

print("\n--- Generating Response ---")
response, conf, meta = engine.generate_response(query)


print("\n=== RESULT ===")
print(f"Confidence: {conf}")
print(f"Status: {meta.get('status')}")
print(f"Response: {response[:200]}...")
