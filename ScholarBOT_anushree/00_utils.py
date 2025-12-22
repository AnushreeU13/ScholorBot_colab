"""
00_utils.py

Shared utilities for parsing, chunking, and embedding.
Helper functions for both static and user ingestion.
"""

import os
import re
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

# Libraries (assuming langchain, pypdf, sentence-transformers, faiss-cpu)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Config
import importlib.util

# Dynamic import to avoid relative import issues if run as script
# (Though in same dir, "import 01_config" is syntactically invalid due to starting number)
# So we usually rename files or use importlib.
# For simplicity, we assume the user will rename them or we use importlib.
# We will assume this is run as a module or we use a helper.
# Let's try standard import if possible, but python modules can't start with numbers easily.
# FIX: Refactor plan? No, user explicitly asked for 01_, 02_.
# We will use importlib to import the config.

def load_config():
    spec = importlib.util.spec_from_file_location("config", Path(__file__).parent / "01_config.py")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

CFG = load_config()

# ---------------------------------------------------------
# 1. Embedding Model
# ---------------------------------------------------------
def get_embedding_model():
    """
    Returns the HuggingFace embedding model.
    """
    print(f"[INFO] Loading embedding model: {CFG.EMBEDDING_MODEL_NAME}")
    return HuggingFaceEmbeddings(
        model_name=CFG.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

# ---------------------------------------------------------
# 2. Text Splitting (Chunking)
# ---------------------------------------------------------
def get_text_splitter():
    """
    Returns the configured text splitter.
    Recommended: RecursiveCharacterTextSplitter for scientific text.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

# ---------------------------------------------------------
# 3. PDF Extraction
# ---------------------------------------------------------
def load_and_chunk_pdf(pdf_path: Path) -> List[Any]:
    """
    Loads a PDF, extracts metadata, and chunks it.
    Returns a list of LangChain Document objects.
    """
    if not pdf_path.exists():
        print(f"[WARN] File not found: {pdf_path}")
        return []

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    
    # Metadata Enrichment
    # We want to ensure we have "author" (organization), "year", "title" for APA citations.
    # We heuristically extract this from filename or file properties if not present.
    
    file_name = pdf_path.stem  # e.g., "WHO_TB_Consolidated_Guidelines"
    creation_time = datetime.fromtimestamp(pdf_path.stat().st_mtime).year
    
    # Simple heuristic to extract Year from filename if present (e.g. "...2019")
    year_match = re.search(r"(19|20)\d{2}", file_name)
    year = year_match.group(0) if year_match else str(creation_time)
    
    # Heuristic for Author/Org
    if "who" in file_name.lower():
        author = "World Health Organization"
    elif "cdc" in file_name.lower():
        author = "Centers for Disease Control and Prevention"
    elif "ats" in file_name.lower() or "idsa" in file_name.lower():
        author = "American Thoracic Society / IDSA"
    else:
        author = "Unknown Author"

    splitter = get_text_splitter()
    chunks = splitter.split_documents(pages)

    # Update metadata for all chunks
    for chunk in chunks:
        chunk.metadata["source"] = str(pdf_path.name)
        chunk.metadata["title"] = file_name.replace("_", " ").replace("-", " ").title()
        chunk.metadata["author"] = author
        chunk.metadata["year"] = year
        chunk.metadata["file_path"] = str(pdf_path)

    return chunks

# ---------------------------------------------------------
# 4. Vector Store Manager
# ---------------------------------------------------------
def save_vector_store(params_db, folder_path: Path):
    params_db.save_local(str(folder_path))
    print(f"[INFO] Index saved to {folder_path}")

def load_vector_store(folder_path: Path, embeddings):
    if not folder_path.exists():
        return None
    return FAISS.load_local(
        str(folder_path), 
        embeddings, 
        allow_dangerous_deserialization=True # Local env, safe
    )
