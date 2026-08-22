import sys
import types
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Compatibility shim for legacy VertexAI import in Ragas 0.4.3
vertex_stub = types.ModuleType("langchain_community.chat_models.vertexai")
vertex_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = vertex_stub


import json
import pandas as pd
from src.config import settings
import instructor
from ragas import evaluate, SingleTurnSample, EvaluationDataset
from ragas.metrics.collections import (
    Faithfulness,
    ResponseGroundedness,
    ContextPrecision,
    ContextPrecisionWithReference,
    ContextRecall
)

from openai import AsyncOpenAI
from ragas.metrics.base import Metric
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.retrieval.vector_store import VectorStore
from src.pipeline import rag_pipeline
from src.generation.llm_service import get_llm
from src.retrieval.embedding_manager import get_embeddings_model
from groq import AsyncGroq


client = AsyncOpenAI(
    api_key="dummy",      # can be any string for this test
    base_url="https://api.groq.com/openai/v1"
)

llm = llm_factory(
    model="llama-3.3-70b-versatile",   # or your model
    client=client,
)

metrics = [
    Faithfulness(llm=llm),
    ContextPrecisionWithReference(llm=llm),
    ContextRecall(llm=llm),
]

for m in metrics:
    print(type(m))
    print(isinstance(m, Metric))

def load_test_cases(file_path:str = "evaluation/test_dataset.json"):
    with open(file_path,"r",encoding="utf-8") as f:
        return json.load(f)
    
def main():
    print("--- Initializing Baseline Ragas Evaluation---")
    
    
    # Load retrievers and components
    vector_store = VectorStore()
    retriever = vector_store.as_retriever()
    chain = rag_pipeline(vector_store=vector_store)
    
    test_cases = load_test_cases()
    samples = []
    
    for item in test_cases:
        question = item['question']
        ground_truth = item['ground_truth']
        
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
        
    # configure groq and huggigface
    client = AsyncOpenAI(api_key=settings.GROQ_API_KEY,base_url="https://api.groq.com/openai/v1")
    evaluate_llm = llm_factory(
        model = settings.GROQ_MODEL_NAME,
        client=client,
    )
    evaluate_embeddings = LangchainEmbeddingsWrapper(get_embeddings_model())
    
    #define evaluation metrices
    
    metrics = [
        Faithfulness(llm=evaluate_llm),
        #ResponseGroundedness(llm=evaluate_llm, embeddings=evaluate_embeddings),
        ContextPrecisionWithReference(llm=evaluate_llm),
        ContextRecall(llm=evaluate_llm),
    ]
    print(metrics)

    for i, metric in enumerate(metrics):
        print(f"{i}: {metric}")
        print(type(metric))
        
    eval_dataset = EvaluationDataset(samples=samples)
    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
    )
    
    df = results.to_pandas()
    print("\n=== Baseline Ragas Evaluation Results ===")
    print(df)
    df.to_csv("evaluation/baseline_ragas_score.csv",index=False)
    print("\nSaved Baseline benchmark to evaluation/baseline_ragas_score.csv")
    
    
if __name__=="__main__":
    main()