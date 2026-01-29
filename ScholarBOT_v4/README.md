# ScholarBOT: Aligned Clinical RAG

ScholarBOT is a high-precision **Clinical Retrieval-Augmented Generation (RAG)** system specialized in **Tuberculosis (TB)**, **Community-Acquired Pneumonia (CAP)**, and **Drug Labels**.

Unlike traditional RAG systems, ScholarBOT adopts a **Safety-First architecture**. It utilizes **multi-layer Safety Gates** and **deterministic alignment mechanisms** to ensure that every generated medical claim can be accurately traced back to its original evidence snippet, minimizing the risk of medical hallucination in high-stakes clinical settings.

---

## Key Features

### Evidence Traceability
- Unique **paper-style trace** mechanism that maps every generated claim to its supporting snippet and citation.

### Dual-Encoder Retrieval
- Powered by **NCBI MedCPT** models (Query Encoder + Article Encoder), optimized for biomedical literature retrieval.

### Five-Layer Safety Gates

1. **Router Gate**
   - Strict intent classification with a *fail-closed* mechanism for out-of-domain queries.

2. **Drug Anchor Filter**
   - Prevents retrieval drift to incorrect medications  
     *(e.g., avoids Rifampin evidence when querying Isoniazid)*.

3. **Guideline Diagnosis Gate**
   - For diagnosis-related queries, restricts retrieval to diagnostic sections  
     *(e.g., testing, imaging)* to avoid treatment contamination.

4. **Entailment Gate**
   - Verifies that generated bullet points are **lexically entailed** by retrieved evidence, blocking high-risk hallucinations.

5. **Patient Safety Gate**
   - Ensures patient-facing summaries do not introduce medical entities absent from the clinician summary.

### Hybrid LLM Engine
- Supports OpenAI models (e.g., **GPT-4o**) with automatic fallback to local models  
  *(e.g., Qwen, LLaMA)*.

### Multi-Perspective Summaries
- Generates both:
  - **Clinician Summaries** (professional, guideline-aligned)
  - **Patient Summaries** (accessible, safety-constrained)

---

## Prerequisites

- Python **3.9+**
- PyTorch *(CUDA recommended for embedding generation)*
- Streamlit
- FAISS *(CPU or GPU)*
- *(Optional)* OpenAI API Key

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```
### 2. Data Ingestion & Indexing

ScholarBOT relies on pre-built FAISS indices. You must run the ingestion scripts before launching the application.

**Clinical Guidelines (CDC / WHO)**

```bash
python download_cdc_tb_guidance.py
python ingest_guidelines_kb.py
```

### Drug Labels (DailyMed SPL)

> **Note**  
> This step requires raw DailyMed SPL ZIP files to be placed at:  
> `datasets/KB_raw/druglabels/spl_full_release_zips/`

```bash
python build_spl_setid_whitelist.py
python ingest_dailymed_spl_filtered.py
```

### 3. Run the Application

```bash
streamlit run app.py
```

### 4. Configuration (Optional)

Set the OpenAI API key to enable GPT-4o enhancement mode:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

If not set, the system defaults to local model configurations defined in `config.py`.

## Project Structure

```plaintext
ScholarBOT/
├── app.py                      # Streamlit frontend entry point
├── aligned_backend.py          # Backend logic (pipeline + UI formatting)
├── rag_pipeline_aligned.py     # Core RAG engine with safety gates
├── router.py                   # Intent recognition and query routing
├── config.py                   # Global configuration
├── requirements.txt            # Python dependencies
│
├── datasets/                   # Data storage
│   ├── KB_raw/                 # Raw PDFs, HTMLs, ZIP files
│   └── KB_processed/           # Processed JSONL and metadata
│
├── faiss_indices/              # Vector indices and metadata
│
├── utils/
│   ├── embedding_utils.py      # MedCPT dual-encoder wrapper
│   ├── storage_utils.py        # FAISS I/O utilities
│   ├── chunking_utils.py       # Semantic chunking logic
│   ├── pdf_utils.py            # PDF parsing and reference stripping
│   └── illustrator.py          # (Experimental) medical illustration module
│
└── ingestion/                  # ETL pipelines
    ├── ingest_guidelines_kb.py
    ├── ingest_dailymed_spl_filtered.py
    ├── build_spl_setid_whitelist.py
    └── download_cdc_tb_guidance.py

```

## Configuration Options

Key parameters can be adjusted in `config.py`:

### `ZERO_HALLUCINATION_MODE`

- `True`  
  Extractive-only mode (maximum safety)

- `False`  
  Enables constrained generative summarization via the Entailment Gate

### `KB_SIM_THRESHOLD`

- Vector similarity thresholds defined per knowledge base

---

## Data Sources

### Drug Labels
- National Library of Medicine (NLM) – **DailyMed SPL**

### Clinical Guidelines
- CDC Tuberculosis Guidelines  
- WHO Consolidated Tuberculosis Guidelines  
- ATS / IDSA Community-Acquired Pneumonia Guidelines

---

## License

This project is intended for **academic research and demonstration purposes only**.  
Always consult original, authoritative clinical sources before making any medical decisions.


