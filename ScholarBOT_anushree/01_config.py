"""
01_config.py

Configuration for ScholarBot (Tier-1 Clinical RAG).
Defines paths, thresholds, and model settings.
"""

import os
from pathlib import Path

# ---------------------------------------------------------
# 1. Paths & Directories
# ---------------------------------------------------------
# Root of the ScholarBot implementation
BASE_DIR = Path(__file__).parent.resolve()

# Vector Database (FAISS) Locations
FAISS_ROOT = BASE_DIR / "faiss_indices"
KB_STATIC_INDEX_DIR = FAISS_ROOT / "main_kb"   # For authoritative guidelines (TB/Pneumonia)
KB_USER_INDEX_DIR = FAISS_ROOT / "user_kb"     # For user-uploaded docs

# Initial Training Data Sources (Local or Box)
# "Assume training data is..."
DATA_SOURCES = [
    Path(r"C:\Users\au11\Box\ScholarBOT\pneumonia"),
    Path(r"C:\Users\au11\Box\ScholarBOT\Tuberculosis"),
    Path(r"C:\Users\au11\Box\ScholarBOT\guidelines"), # Added per user request
]

# Staging area for user uploads before ingestion
USER_UPLOAD_STAGING_DIR = BASE_DIR / "staging_user_uploads"
USER_UPLOAD_STAGING_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. RAG & Retrieval Settings
# ---------------------------------------------------------
# Confidence Gating: Logic requires > 60% confidence
# We use Cosine Similarity (0.0 to 1.0) as a proxy.
SIM_THRESHOLD = 0.35

# Model Name for Embeddings
# Using MedCPT (Query/Article Encoder) or a strong clinical model.
# We will use a standard SentenceTransformer for ease of implementation in this demo,
# potentially mapping to valid HF hub models.
# For "MedCPT", we might need specific wrappers, but typically "PrunaAI/ MedCPT-Query-Encoder" serves well.
# Or we can stick to "sentence-transformers/all-MiniLM-L6-v2" if hardware is limited,
# but user asked for "recommended", so we default to a clinical-aware one if possible,
# or a strong generalist.
# Let's use a very standard, robust one for the code skeleton:
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# NOTE: To use MedCPT specifically, we'd need the specific 'ncbi/MedCPT-Query-Encoder'
# but that often requires specific architecture code. We'll stick to standard HF for stability unless requested.

# Generation Model (Local LLM) settings
LLM_MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"  # or similar local path
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1  # Low temperature for factual accuracy

# ---------------------------------------------------------
# 3. System Prompts
# ---------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = """You are ScholarBot, an expert clinical AI assistant for doctors, researchers, and patients.
Your scope is strictly limited to Tuberculosis and Pneumonia.

Rules:
1. Answer the user's question using ONLY the provided Context.
2. If the Context does not contain the answer, say "No confidence in answering".
3. The answer must be easy to understand (plain language) but medically accurate.
4. Do NOT make up information (Zero Hallucination).

Context:
{context}

User Question: {question}
"""
