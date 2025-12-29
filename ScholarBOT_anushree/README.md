# ScholarBot (Clinical RAG Assistant)

ScholarBot is a specialized RAG (Retrieval-Augmented Generation) application designed for clinical queries regarding **Tuberculosis (TB)** and **Pneumonia**.

It features a **Tiered Retrieval System** that prioritizes standard WHO/ATS guidelines but can seamlessly switch to "Context-Aware" mode to answer questions from user-uploaded documents.

## 🚀 Key Features

*   **Tiered Retrieval**: 
    *   **Tier 1 (Main KB)**: Searches authoritative guidelines (WHO, ATS).
    *   **Tier 2 (User KB)**: Searches user uploads if Guidelines are insufficient.
    *   **Context Mode**: If a user uploads a file, it locks search to *that specific file* for Q&A.
*   **Conversational Memory**: Rewrites vague follow-up questions (e.g., "Tell me more") into full search queries based on chat history.
*   **Patient Illustrator Agent**: Generates medical-grade, text-free diagrams using a cloud AI illustrator (Pollinations.ai) to explain concepts visually.
*   **Zero-Latency Citations**: Extracts metadata (Title, Author, References) during ingestion to provide instant, formatted citations.
*   **Clean UI**: Streamlit interface with clear source attribution and no clutter.

## 🛠️ Installation

1.  **Prerequisites**:
    *   Python 3.10+
    *   [Ollama](https://ollama.com/) running locally (`ollama serve`).
    *   Model: `llama3` (Run `ollama pull llama3`).

2.  **Install Dependencies**:
    ```bash
    pip install streamlit langchain langchain-community sentence-transformers faiss-cpu pypdf requests pillow
    ```

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `00_utils.py` | Core utilities: PDF loading, chunking, and FAISS vector store management. **(New: Smart Metadata Extraction)** |
| `01_config.py` | Configuration: Paths, model names (`llama3`), thresholds (`0.35`). |
| `02_ingest_static.py` | Script to ingest "Gold Standard" guidelines (Main KB). |
| `03_ingest_user.py` | Script to ingest user-uploaded files (User KB). |
| `04_engine.py` | **The Brain**: RAG logic, Tiered Search, Contextual Query Rewriting. |
| `05_citations.py` | Citation refinement logic (Reads pre-calc metadata). |
| `06_app.py` | **The Frontend**: Streamlit Web UI. Handles session state and chat. |
| `07_illustrator.py` | **The Artist**: Generates visual prompts and fetches images. |

## 🏃 Usage

1.  **Start the App**:
    ```bash
    streamlit run 06_app.py
    ```
    
2.  **Modes**:
    *   **General Mode**: Ask "What is the treatment for TB?". It searches WHO Guidelines.
    *   **Focused Mode**: Upload a PDF in the sidebar. The bot will now answer strictly from your uploaded file. (Refresh page to reset).

## 🧠 Technical Highlights

*   **Single-Pass Ingestion**: We extract Titles and Bibliographies *once* when you upload, saving them in the Vector DB. This avoids re-reading the PDF during chat, responding in heavy files instantly.
*   **Confidence Gating**: The bot will say "No confidence" if the similarity score is below `0.35`, reducing hallucinations.
*   **Query Rewriting**: Uses a lightweight LLM call to contextualize user questions before searching FAISS.
