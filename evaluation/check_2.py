import pandas as pd

df = pd.read_csv("evaluation/baseline_ragas_score_2.csv")

# Resolve column name variations across Ragas builds
relevancy_col = next(
    (c for c in ["answer_relevancy", "response_relevancy", "answer_relevance"] if c in df.columns),
    None,
)
precision_col = next(
    (c for c in ["llm_context_precision_with_reference", "context_precision"] if c in df.columns),
    None,
)

print("=" * 85)
print("     BASELINE RAGAS EVALUATION: RELEVANCY & PRECISION BREAKDOWN")
print("=" * 85)

for idx, row in df.iterrows():
    print(f"[{idx+1}] Question: {row['user_input']}")
    print(f"    Generated Answer  : {row.get('response', 'N/A')}")
    print(f"    Answer Relevancy  : {row.get(relevancy_col, 'N/A')}")
    print(f"    Context Precision : {row.get(precision_col, 'N/A')}")
    print("-" * 85)

# Calculate summary averages
print("\n" + "=" * 30 + " SUMMARY AVERAGES " + "=" * 30)
if relevancy_col:
    print(f"Average Answer Relevancy : {df[relevancy_col].mean():.4f}")
if precision_col:
    print(f"Average Context Precision: {df[precision_col].mean():.4f}")
print("=" * 78)