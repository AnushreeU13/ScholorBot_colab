"""
ragas_v5.py

Implements RAGAS evaluation metrics for ScholarBOT v5 (Llama 3 + v4 retrieval).
Requires OPENAI_API_KEY to be set in the environment for the *Evaluator* (RAGAS Judge).

Integration:
- Uses v5 AlignedScholarBotEngine for generation.
- Uses RAGAS for evaluation (Faithfulness, Answer Relevancy, Context Precision/Recall).
"""

import os
import sys
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
import logging

# Add current dir to path to find v5 modules
sys.path.append(os.getcwd())

# Import v5 Backend
try:
    from aligned_backend import AlignedScholarBotEngine
except ImportError:
    print("Error: Could not import AlignedScholarBotEngine. Run this script from ScholarBOT_v5 directory.")
    sys.exit(1)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_evaluation_suite(
    test_questions: list[str],
    ground_truths: list[list[str]] = None,
    api_key: str = None
) -> pd.DataFrame:
    """
    Runs end-to-end evaluation:
    1. Generates answers using ScholarBOT v5 (Llama 3).
    2. Evaluates them using RAGAS (GPT-4o/mini Judge).
    """

    # 1. Setup Env
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif not os.getenv("OPENAI_API_KEY"):
         # Fallback to key if set elsewhere or rely on env
         os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not found. RAGAS evaluation (The Judge) will likely fail.")

    # 2. Initialize v5 Engine
    logger.info("Initializing ScholarBOT v5 Engine...")
    engine = AlignedScholarBotEngine(verbose=False) # Suppress debug logs for clean eval output

    questions = []
    answers = []
    contexts = []

    # 3. Generate Answers
    logger.info(f"Generating answers for {len(test_questions)} questions...")
    for q in test_questions:
        try:
            # Generate using v5 pipeline
            response, conf, meta = engine.generate_response(q)
            
            # Extract Evidence Chunks for RAGAS
            # debug_info usually contains the chunks in 'evidence_chunks' if configured
            # But generate_response returns (text, float, dict)
            # We need the raw chunks. The `meta` dict currently returns 'status' etc.
            # We might need to access the pipeline details more directly or trust the returned meta if it has chunk info.
            
            # For v5 aligned_backend.py:
            # result = self.pipeline.retrieve_and_answer(query)
            # return result.answer, result.confidence, result.debug_info
            
            evidence_text_list = []
            if meta and "evidence_chunks" in meta:
                 evidence_text_list = [c.get("text", "") for c in meta["evidence_chunks"]]
            else:
                # Fallback if chunks aren't passed cleanly
                evidence_text_list = ["No context captured."]

            questions.append(q)
            answers.append(response)
            contexts.append(evidence_text_list)

        except Exception as e:
            logger.error(f"Generation failed for q='{q}': {e}")
            questions.append(q)
            answers.append("GENERATION_FAILED")
            contexts.append([""])

    # 4. Prepare RAGAS Dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truths:
        data["ground_truth"] = ground_truths
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    else:
        # Without ground truth, we can only measure faithfulness & relevancy (and partial precision)
        metrics = [faithfulness, answer_relevancy, context_precision]

    dataset = Dataset.from_dict(data)

    # 5. Evaluate
    logger.info("Running RAGAS evaluation...")
    
    # Initialize RAGAS-compatible LLM and Embeddings
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        ragas_llm = ChatOpenAI(model="gpt-4o")
        ragas_embeddings = OpenAIEmbeddings()
    except ImportError:
        logger.warning("Could not import langchain_openai. Using default RAGAS configuration.")
        ragas_llm = None
        ragas_embeddings = None

    try:
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        return results.to_pandas()
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    print("=== ScholarBOT v5 RAGAS Evaluation ===")
    
    # 1. Define Test Set (Expanded List)
    test_questions = [
        "How is tuberculosis diagnosed according to WHO guidelines?",
        "What tests are recommended for diagnosing active tuberculosis?",
        "What is sputum culture conversion and why is it important in TB management?",
        "What does bacteriological confirmation mean in tuberculosis?",
        "What follow-up indicators are used to monitor TB treatment response?",
        "What is the role of sputum smear microscopy in TB diagnosis?",
        "How is bacteriological failure defined in tuberculosis follow-up?",
        "What outcomes are used to assess TB treatment success?",
        "What are the first-line drugs used to treat tuberculosis?",
        "What is isoniazid used for in tuberculosis treatment?",
        "What is rifampin and what role does it play in TB therapy?",
        "What are common adverse reactions of isoniazid?",
        "What adverse effects are associated with rifampin?",
        "What precautions are listed for ethambutol use?",
        "Why is combination therapy used in tuberculosis treatment?",
        "How is community-acquired pneumonia diagnosed?",
        "What clinical criteria are used to assess pneumonia severity?",
        "What are common symptoms of community-acquired pneumonia?",
        "What diagnostic tests are recommended for suspected pneumonia?",
        "What factors influence initial antibiotic selection for CAP?",
        "What are common adverse reactions of azithromycin?",
        "What warnings are listed for levofloxacin?",
        "What is doxycycline commonly prescribed for?",
        "What drug interactions are associated with rifampin?",
        "What precautions should be considered when prescribing fluoroquinolones?",
        "Why is adherence important in tuberculosis treatment?",
        "What risks are associated with incomplete TB treatment?",
        "What is the purpose of standardized TB treatment regimens?"
    ]
    
    # Ground Truths (Provided by User)
    ground_truths = [
        "Bacteriological confirmation using sputum samples; Microbiological tests (smear microscopy, culture, molecular tests); Clinical and radiological assessment as supportive evidence", # Q1
        "Sputum smear microscopy; Mycobacterial culture; WHO-endorsed rapid molecular tests (e.g. Xpert)", # Q2
        "Change from culture-positive to culture-negative; Indicator of treatment response; Used for monitoring treatment effectiveness", # Q3
        "Laboratory evidence of Mycobacterium tuberculosis; Positive smear, culture, or molecular test", # Q4
        "Sputum smear or culture results; Bacteriological failure or relapse; Treatment outcomes at follow-up time points", # Q5
        "Initial diagnostic test; Detects acid-fast bacilli; Used in many resource-limited settings", # Q6
        "Persistent or recurrent positive bacteriology; Occurs during or after treatment; Indicates ineffective therapy", # Q7
        "Cure; Treatment completion; Absence of bacteriological failure or relapse", # Q8
        "Isoniazid; Rifampin; Ethambutol; Pyrazinamide", # Q9
        "Core first-line anti-TB drug; Used in combination regimens; Active against Mycobacterium tuberculosis", # Q10
        "First-line anti-TB antibiotic; Bactericidal activity; Shortens treatment duration", # Q11
        "First-line anti-TB antibiotic; Bactericidal activity; Shortens treatment duration", # Q12 (Note: Matches user input, possible duplicate of Q11)
        "Hepatotoxicity; Gastrointestinal upset; Drug interactions via enzyme induction", # Q13
        "Hepatotoxicity; Gastrointestinal upset; Drug interactions via enzyme induction", # Q14 (Note: Matches user input, possible duplicate of Q13)
        "Prevent drug resistance; Improve treatment efficacy; Target different bacterial populations", # Q15
        "Prevent drug resistance; Improve treatment efficacy; Target different bacterial populations", # Q16 (Note: Matches user input, possible duplicate of Q15)
        "Vital signs; Laboratory findings; Severity scoring systems (general mention)", # Q17
        "Cough; Fever; Shortness of breath; Chest pain", # Q18
        "Chest radiography; Blood tests; Microbiological testing when appropriate", # Q19
        "Disease severity; Patient comorbidities; Likely pathogens", # Q20
        "Gastrointestinal upset; QT prolongation; Liver enzyme abnormalities", # Q21
        "Tendon rupture; QT prolongation; CNS effects", # Q22
        "Bacterial infections; Respiratory tract infections; Broad-spectrum antibiotic use", # Q23
        "Enzyme induction; Reduced effectiveness of co-administered drugs; Multiple clinically significant interactions", # Q24
        "Risk of tendon injury; QT prolongation; CNS and neurological effects", # Q25
        "Prevent resistance; Ensure treatment success; Reduce relapse risk", # Q26
        "Drug resistance; Treatment failure; Disease relapse", # Q27
        "Consistent care; Improved outcomes; Simplified programmatic management" # Q28
    ]

    # 2. Run
    df = run_evaluation_suite(test_questions, ground_truths=ground_truths)
    
    if not df.empty:
        print("\n=== Evals Results ===")
        print(f"Columns: {df.columns.tolist()}")
        
        # Save raw first
        df.to_csv("ragas_v5_results.csv", index=False)
        print("\nSaved detailed results to ragas_v5_results.csv")
        
        # Display subset if columns exist
        candidates = ["user_input", "question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        display_cols = [c for c in candidates if c in df.columns]
        if display_cols:
            print(df[display_cols])
    else:
        print("\nEvaluation produced no results.")
