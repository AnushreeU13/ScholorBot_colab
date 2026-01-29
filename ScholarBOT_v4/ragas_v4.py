"""
ragas_v4.py

Implements RAGAS evaluation metrics for ScholarBOT v4.
Requires OPENAI_API_KEY to be set in the environment.

Dependencies:
    pip install ragas datasets pandas
"""

import os
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_rag_pipeline(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[list[str]] = None,
    api_key: str = None
) -> pd.DataFrame:
    """
    Evaluates RAG pipeline performance using RAGAS metrics.
    
    Args:
        questions: List of questions asked.
        answers: List of generated answers.
        contexts: List of lists of retrieved context chunks.
        ground_truths: (Optional) List of lists of ground truth answers. 
                       Required for context_recall.
        api_key: OpenAI API Key. If None, checks os.environ["OPENAI_API_KEY"].

    Returns:
        pd.DataFrame: Evaluation results with metrics.
    """
    
    # 1. Setup API Key
    # Use provided argument, or fallback to environment, or fallback to hardcoded key
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif not os.getenv("OPENAI_API_KEY"):
         # Fallback to the key provided by user
         os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not found. RAGAS evaluation will likely fail.")

    # 2. Prepare Dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    
    # Use a smaller metric set for verification if needed, but standard is fine
    metrics = [faithfulness, answer_relevancy, context_precision]

    if ground_truths:
        data["ground_truth"] = ground_truths
        metrics.append(context_recall)

    dataset = Dataset.from_dict(data)

    # 3. Run Evaluation
    logger.info(f"Starting RAGAS evaluation on {len(questions)} samples...")
    try:
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
        )
        
        # 4. Convert to Pandas
        df = results.to_pandas()
        logger.info("Evaluation complete.")
        return df

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        raise e

if __name__ == "__main__":
    # Test Block
    print("=== RAGAS v4 Module Test ===")
    
    # Mock Data
    test_questions = ["What is the dosage of Isoniazid?"]
    test_answers = ["The dosage is 5 mg/kg daily."]
    test_contexts = [["Isoniazid is administered at 5 mg/kg once daily for adults."]]
    test_ground_truths = ["5 mg/kg daily"]

    print("Running evaluation with mock data...")
    try:
        df_result = evaluate_rag_pipeline(
            test_questions, 
            test_answers, 
            test_contexts, 
            test_ground_truths
        )
        print("\nEvaluation Successful!")
        print(df_result[["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]])
        print("\nModule imports and API connection successful.")
    except Exception as e:
        print(f"\nEvaluation Failed: {e}")
