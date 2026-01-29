# Methodology

## 1. Data Corpus and Preprocessing

The system's knowledge base (KB) is constructed from two primary authoritative sources to ensure clinical accuracy and minimize noise. The first component, the **Clinical Guidelines Corpus**, comprises high-impact clinical practice guidelines focused on Tuberculosis (TB) and Community-Acquired Pneumonia (CAP) from major organizations such as the World Health Organization (WHO), American Thoracic Society (ATS), IDSA, and CDC. These documents undergo structure-aware parsing to retain section hierarchy, followed by segmentation into 240-token chunks with a 50-token overlap to optimize dense retrieval performance for granular clinical queries. The second component, the **Pharmacological Corpus**, integrates a filtered subset of FDA Structured Product Labels (SPL) from DailyMed, specifically targeting anti-infective and antitubercular agents. This XML data is parsed to extract key structural fields such as "Boxed Warnings" and "Dosage," with chunking performed at the section level to prevent context bleeding between different drug entries.

---

## 2. System Architecture: Tiered Clinical RAG

We propose a **Structurally Constrained Retrieval-Augmented Generation (RAG)** architecture designed to enforce a "Fail-Closed" safety behavior. The pipeline operates through three distinct stages: task-aware routing, structurally constrained retrieval, and gated generation.

### 2.1. Task-Aware Routing
Incoming queries are first processed by a semantic router that classifies them into specific intents based on domain triggers and task keywords. For example, queries strictly regarding protocols (e.g., *"How to diagnose latent TB?"*) are classified as **Guideline Intent** and routed exclusively to the Guidelines KB. Conversely, queries requesting medication specifics (e.g., *"Bedaquiline dosage"*) are classified as **Drug Intent** and directed to the Drug Labels KB. Broad in-domain queries that lack specific task keywords (e.g., *"How does TB spread?"*) are assigned a **Mixed Intent**, triggering a search across both knowledge bases to maximize recall. Finally, queries unrelated to the core domain (e.g., *"Management of Type 2 Diabetes"*) are immediately rejected as **Out-of-Domain** to prevent hallucinatory retrieval.

### 2.2. Structurally Constrained Retrieval
Unlike standard dense retrieval, our system imposes structural constraints to reduce the inclusion of irrelevant context. We implement three specific constraint mechanisms:

1.  **Section Boosting**: We address the diverse structure of medical documents by dynamically modifying retrieval scores based on query keywords. If a query contains tokens mapping to a specific section group (e.g., "side effects" maps to "Adverse Reactions"), chunks originating from that section metadata receive a similarity boost of $\alpha=0.12$. This ensures that even if a "Description" section has high semantic overlap, the system prioritizes the clinical safety section.

2.  **Drug Anchor Filter**: To prevent dangerous cross-drug hallucinations (e.g., retrieving "Rifampin" dosage for a "Bedaquiline" query), we enforce a hard lexical constraint. A post-retrieval filter scans all candidate chunks; any chunk that does not explicitly contain the target drug name (e.g., "Bedaquiline") in either its `title` or `text` fields is summarily rejected, regardless of its vector similarity score.

3.  **Diagnosis Evidence Gate**: Diagnostic queries (e.g., "How to diagnose latent TB?") often retrieve treatment protocols due to term co-occurrence. We implemented a dedicated gate that activates only for diagnosis intents. This gate inspects the chunk metadata and local text window (first 900 characters) for diagnostic keywords (e.g., "culture", "radiograph", "test"). Any chunk failing this check is filtered out, ensuring the final answer is derived strictly from diagnostic evidence.

The collective intent of these structural constraints is to mitigate the inherent risks of "semantic drift" in dense retrieval systems, where high cosine similarity can sometimes obscure critical medical distinctions (e.g., conflating distinct clinical contexts). By enforcing these hard logic gates, we prioritize precision and safety over raw recall, ensuring that the system fails closed rather than hallucinating plausible but incorrect associations.

### 2.3. Fail-Closed Generation with Safety Gates
The generation phase employs a 7B-parameter instruction-tuned model wrapped in a verification loop to ensure answer fidelity. A **Lexical Entailment Gate** verifies every generated sentence against the retrieved evidence using a strict two-step process to prune unsupported statements:

1.  **High-Risk Term Block**: The system first checks for a blacklist of high-risk medical terms (e.g., "HIV", "Carbapenem"). If a generated bullet contains such a term but the retrieved evidence does not, the bullet is immediately rejected to prevent dangerous "phantom" diagnoses.
2.  **Token Overlap Ratio**: For remaining bullets, the system calculates a token overlap ratio defined as $(Keywords_{evidence} \cap Keywords_{bullet}) / Keywords_{bullet}$. If the overlap falls below a threshold of $\tau=0.25$, the statement is discarded as an unsupported hallucination.

Additionally, a **Medical Entity Safety Guard** compares a secondary "Patient Summary" against the primary "Clinician Summary"; if the patient version introduces new medical entities not present in the clinical source, the response is flagged as a hallucination. Finally, an **Abstention Protocol** ensures that if retrieval confidence falls below KB-specific thresholds (with a default fallback of $\tau=0.65$) or if verification fails, the system returns a standardized "ABSTAIN" response rather than attempting a low-confidence guess.

---

## 3. Implementation Details

The implementation follows a rigorous end-to-end pipeline beginning with **Data Processing**, where raw PDFs and XMLs are parsed and semantically chunked (keeping section headers intact) before being encoded by **MedCPT**, a domain-specific model trained on PubMed/clinical notes. These vectors are indexed into three dedicated FAISS Knowledge Bases: `kb_guidelines_medcpt` (for protocols), `kb_druglabels_medcpt` (for medications), and a dynamic `user_fact_kb_medcpt` to support real-time user document uploads.

At inference time, the **Task-Aware Router** directs incoming queries to the appropriate KB. The retrieval process then rigorously applies the safety architecture described previously: leveraging **Section Boosting** to prioritize relevant headers, the **Drug Anchor Filter** to lock onto specific medications, and the **Diagnosis Evidence Gate** to isolate diagnostic context. Finally, valid chunks are passed to the generation module, where they undergo the **Lexical Entailment Gate** verification to produce the final answer.

To enhance interpretability and trust, we implemented a **Claim-Snippet Alignment** algorithm. This post-processing step creates a transparent evidence trail by mathematically linking each generated sentence to its source. The algorithm works by:
1.  **Tokenization**: Decomposing both the generated claim and the retrieved evidence chunks into unique key tokens (excluding English stopwords).
2.  **Jaccard Verification**: Calculating the Jaccard Similarity Coefficient $J(A,B) = |A \cap B| / |A \cup B|$ between the claim tokens ($A$) and the tokens of every sentence in the valid evidence set ($B$).
3.  **Traceability**: Assigning the highest-scoring source sentence as the "supporting snippet" for that claim, effectively allowing users to audit the model's reasoning sentence-by-sentence.
This alignment layer serves as an interpretability aid rather than a strict correctness verifier, ensuring that clinicians can rapidly verify the provenance of AI-generated advice.
