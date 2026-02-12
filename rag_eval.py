# rag_eval.py

from pathlib import Path
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevanceMetric, ContextRelevanceMetric
from deepeval.evaluator import evaluate
import json

# === Config ===
RAG_LOG_FILE = "rag_outputs.json" 

# Load logs: expects a list of dicts with 'input', 'actual_output', and 'context'
def load_rag_logs(log_path):
    if not Path(log_path).exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    with open(log_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Define metrics
metrics = [
    FaithfulnessMetric(threshold=0.9),
    AnswerRelevanceMetric(threshold=0.85),
    ContextRelevanceMetric(threshold=0.8),
]

# Evaluate each RAG test case
def evaluate_rag_responses(logs):
    print(f"\n Evaluating {len(logs)} RAG responses...\n")
    for i, entry in enumerate(logs, 1):
        try:
            test_case = LLMTestCase(
                input=entry["input"],
                actual_output=entry["actual_output"],
                context=entry["context"],
                expected_output=entry.get("expected_output", "")
            )
            print(f"\nTest Case {i}: {entry['input'][:80]}...")
            evaluate(test_case, metrics)
        except Exception as e:
            print(f" Error in test case {i}: {e}")

# === Run if script is executed directly ===
if __name__ == "__main__":
    rag_logs = load_rag_logs(RAG_LOG_FILE)
    evaluate_rag_responses(rag_logs)
