"""
04_rag_engine.py

Core Logic for ScholarBot:
1. Tiered Retrieval (User KB -> Main KB)
2. Confidence Gating (Threshold > 0.6)
3. APA Citation Formatting
"""

import importlib.util
from pathlib import Path
from typing import List, Dict, Optional
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# ... (omitted)

# Load Utils & Config
utils_spec = importlib.util.spec_from_file_location("utils", Path(__file__).parent / "00_utils.py")
utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(utils)

CFG = utils.CFG

class ScholarBotEngine:
    def __init__(self):
        print("[INFO] Initializing ScholarBot Engine...")
        self.embeddings = utils.get_embedding_model()
        
        # Load Vector Stores (Fail gracefully if not present)
        self.main_kb = utils.load_vector_store(CFG.KB_STATIC_INDEX_DIR, self.embeddings)
        self.user_kb = utils.load_vector_store(CFG.KB_USER_INDEX_DIR, self.embeddings)
        
        if self.main_kb:
            print("[OK] Main KB loaded.")
        else:
            print("[WARN] Main KB not found. Run 02_ingest_static.py first.")
            
        if self.user_kb:
            print("[OK] User KB loaded.")
        
    def _format_apa_citation(self, meta: Dict) -> str:
        author = meta.get("author", "Unknown Author")
        year = meta.get("year", "n.d.")
        title = meta.get("title", "Untitled Document")
        return f"{author}. ({year}). *{title}*."

    def _is_author_line(self, line: str) -> bool:
        # 1. Strong signals
        l = line.lower()
        if "@" in l: return True
        if "correspondence" in l or "received" in l: return True
        if "financial support" in l or "conflict of interest" in l: return True
        if "avenida" in l or "cep" in l or "brasil" in l: return True
        if "tel.:" in l or "fax:" in l: return True
        
        # 2. Regex for Name, Affiliation patterns (e.g. "Smith1, Jones2")
        import re
        if re.search(r"\w+,\d+", l): return True 
        if re.search(r"\w+\d+,\d+", l): return True
        
        return False
        
    def _is_reference_chunk(self, content: str) -> bool:
        """
        Detects if a chunk is primarily a bibliography.
        """
        l = content.lower()
        # Header check
        if l.strip().startswith("references") or l.strip().startswith("bibliography"):
            return True
            
        # Density check: Count dates (19xx, 20xx) and distinct citation markers (Vol, pp, doi)
        import re
        date_count = len(re.findall(r"\b(19|20)\d{2}\b", l))
        # If we have > 3 dates in a chunk, it's likely a ref list (text rarely has that many dates close by)
        if date_count > 3:
            return True
            
        # Citation style patterns: "Journal. 2016;" or "doi.org"
        if l.count("doi.org") > 1 or l.count("http") > 2:
            return True
            
        return False

    def search(self, query: str) -> Dict:
        """
        Performs the tiered search with Deep Retrieval (k=7) for ~1500 word potential.
        """
        # Common logic for both tiers to ensure consistent formatting
        def process_results(docs, source_name):
            # Filter matches above threshold
            valid_docs = [d for d in docs if d[1] >= CFG.SIM_THRESHOLD]
            
            if not valid_docs:
                return None

            # 1. content aggregation
            cleaned_paragraphs = []
            seen_text = set() # For de-duplication at paragraph/sentence level
            
            for d in valid_docs:
                raw_text = d[0].page_content
                
                # Global Filter: Skip entire chunk if it's a reference list
                if self._is_reference_chunk(raw_text):
                    continue

                # Pre-processing: Force newlines to separate merged Author/Content blobs
                # This helps if PDF parsed "Address. INTRODUCTION" as one line.
                raw_text = raw_text.replace("Financial support:", "\nFinancial support:")
                raw_text = raw_text.replace("Correspondence to:", "\nCorrespondence to:")
                raw_text = raw_text.replace(". INTRODUCTION", ".\nINTRODUCTION")
                raw_text = raw_text.replace(". Introduction", ".\nIntroduction")
                raw_text = raw_text.replace(". ABSTRACT", ".\nABSTRACT")
                raw_text = raw_text.replace(". Abstract", ".\nAbstract")

                # 2. Line-by-line cleaning
                clean_lines = []
                for line in raw_text.split('\n'):
                    if not self._is_author_line(line):
                        clean_lines.append(line.strip())
                
                cleaned_text = " ".join(clean_lines)
                
                # 3. De-duplication (Simple overlap check)
                # If 80% of this text is already in our answer, skip it
                is_duplicate = False
                for seen in seen_text:
                    # Simple intersection ratio or containment
                    if len(cleaned_text) < 50: continue # Skip tiny fragments
                    if cleaned_text in seen or seen in cleaned_text:  
                        is_duplicate = True
                        break
                
                if not is_duplicate and len(cleaned_text) > 20:
                    cleaned_paragraphs.append(cleaned_text)
                    seen_text.add(cleaned_text)
            
            if not cleaned_paragraphs:
                return None

            # Join into one massive block (Excerpt 1)
            giant_content = "\n\n".join(cleaned_paragraphs)
            
            # Metadata from top doc (Excerpt 2)
            top_meta = valid_docs[0][0].metadata
            
            return {
                "status": "success",
                "answer_source": source_name,
                "excerpt_1": giant_content,
                "excerpt_2": top_meta,
                "confidence": valid_docs[0][1]
            }

        # Tier 1: User KB
        if self.user_kb:
            user_docs = self.user_kb.similarity_search_with_relevance_scores(query, k=7)
            result = process_results(user_docs, "user_doc")
            if result: return result

        # Tier 2: Main KB
        if self.main_kb:
            main_docs = self.main_kb.similarity_search_with_relevance_scores(query, k=7)
            result = process_results(main_docs, "main_kb")
            if result: return result
        
        return {
            "status": "abstain",
            "message": "No confidence in answering (Score < Threshold)."
        }




    def generate_response(self, query: str, model_name: str = "llama3"):
        """
        Generative RAG (Ollama):
        1. Retrieve massive context (k=10).
        2. Use Local Ollama Model to synthesize.
        """
        # Always use high K for generative
        retrieval_k = 10
        
        retrieval = self.search(query)
        
        if retrieval["status"] == "abstain":
             return "Final Answer: No confidence in answering.", 0.0

        ctx = retrieval["excerpt_1"]
        meta = retrieval["excerpt_2"]
        
        # -------------------------------------------
        # PATH A: GENERATIVE (Local LLM)
        # -------------------------------------------
        try:
            # Local Ollama
            llm = ChatOllama(model=model_name, temperature=0)
            
            # System Prompt for "Cohesion"
            sys_prompt = """You are an expert clinical research assistant. 
            Task: Answer the user's question using ONLY the provided context. 
            Guidelines:
            1. Write a cohesive, well-structured essay/summary. Do not just list snippets.
            2. Use professional medical tone.
            3. If the context mentions specific studies or drugs (e.g. Bedaquiline), name them.
            4. At the end, cite the provided source metadata.
            
            Context:
            {context}
            """
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", sys_prompt),
                ("human", "{question}")
            ])
            
            chain = prompt | llm
            
            response_msg = chain.invoke({
                "context": ctx,
                "question": query
            })
            
            return response_msg.content, retrieval['confidence']
            
        except Exception as e:
            fallback_msg = f"**Ollama Error**: {str(e)}\n\n*Ensure Ollama is running (`ollama serve`) and `{model_name}` is pulled.*\n\n---\n\n**Raw Context**:\n{ctx}"
            return fallback_msg, 0.0

        # -------------------------------------------
        # PATH B: EXTRACTIVE (Legacy)
        # -------------------------------------------
        # Excerpt 1: Detailed Answer (Cleaned)
        excerpt_1 = f"**Excerpt 1 (Raw Findings)**:\n\n{ctx}"
        
        # Excerpt 2: Metadata
        author = meta.get('author', 'Unknown Author')
        title = meta.get('title', 'Untitled Document')
        year = meta.get('year', 'n.d.')
        
        excerpt_2 = f"**Excerpt 2 (Source Metadata)**:\n**Title**: {title}\n**Author**: {author}\n**Year**: {year}"
        
        ans = f"{excerpt_1}\n\n---\n\n{excerpt_2}"
        return ans, retrieval['confidence']

if __name__ == "__main__":
    engine = ScholarBotEngine()
    print("Engine Ready.")
