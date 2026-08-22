import sys
import types
from pathlib import Path

# 1. Resolve project root
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 2. Compatibility shim for legacy VertexAI import in Ragas 0.4.3
vertex_stub = types.ModuleType("langchain_community.chat_models.vertexai")
vertex_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = vertex_stub

import json
import warnings
import pandas as pd

# Suppress non-breaking warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ragas import evaluate, SingleTurnSample, EvaluationDataset
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.retrieval.vector_store import VectorStore
from src.pipeline import rag_pipeline
from src.generation.llm_service import get_judge_llm
from src.retrieval.embedding_manager import get_embeddings_model
from src.config import settings


def load_test_cases(file_path: str | Path | None = None):
    test_dataset_path = Path(file_path) if file_path else ROOT_DIR / "evaluation" / "test_dataset.json"
    with test_dataset_path.open("r", encoding="utf-8") as f:
        return json.load(f)





def main():
    print("--- Initializing Baseline Ragas Evaluation ---")

    # Load components
    vector_store = VectorStore()
    retriever = vector_store.as_retriever(search_kwargs={"k": settings.TOP_K})
    chain = rag_pipeline(vector_store=vector_store)

    test_cases = load_test_cases()
    samples = []

    print(f"Generating pipeline predictions for {len(test_cases)} test cases...")

    for item in test_cases:
        question = item["question"]
        ground_truth = item["ground_truth"]

        docs = retriever.invoke(question)
        context = [d.page_content for d in docs]
        generated_answer = chain.invoke(question)

        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=context,
                response=generated_answer,
                reference=ground_truth,
            )
        )
    # Initialize Evaluators
    gemini_client = get_judge_llm(temperature=0.0)
    evaluate_llm = LangchainLLMWrapper(gemini_client)
    evaluate_embeddings = LangchainEmbeddingsWrapper(get_embeddings_model())

    metrics = [
        Faithfulness(llm=evaluate_llm),
        ResponseRelevancy(llm=evaluate_llm, embeddings=evaluate_embeddings,strictness=1),
        LLMContextPrecisionWithReference(llm=evaluate_llm),
        LLMContextRecall(llm=evaluate_llm),
    ]

    # Configure controlled worker concurrency
    run_config = RunConfig(
        timeout=180,
        max_workers=1,
        max_retries=2,
        max_wait=30,
    )

    print("\nRunning Ragas evaluation with controlled concurrency...")
    eval_dataset = EvaluationDataset(samples=samples)
    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        run_config=run_config,
    )

    df = results.to_pandas()
    print("\n=== Baseline Ragas Evaluation Results ===")
    print(df)
    df.to_csv(ROOT_DIR / "evaluation" / "baseline_ragas_score.csv", index=False)
    print("\nSaved Baseline benchmark to evaluation/baseline_ragas_score.csv")


if __name__ == "__main__":
    main()