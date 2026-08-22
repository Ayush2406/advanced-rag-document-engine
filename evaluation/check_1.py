import pandas as pd

# Load the baseline benchmark results
df = pd.read_csv("evaluation/baseline_ragas_score.csv")

# Detect available metric column names
context_recall_col = (
    "context_recall"
    if "context_recall" in df.columns
    else "llm_context_recall"
)
faithfulness_col = "faithfulness"

print("=" * 80)
print("             BASELINE RAGAS EVALUATION METRIC REPORT")
print("=" * 80)

for idx, row in df.iterrows():
    print(f"[{idx+1}] Question: {row['user_input']}")
    print(f"    Expected Answer : {row.get('reference', 'N/A')}")
    print(f"    Generated Answer: {row.get('response', 'N/A')}")
    print(f"    Faithfulness    : {row.get(faithfulness_col, 'N/A')}")
    print(f"    Context Recall  : {row.get(context_recall_col, 'N/A')}")
    print("-" * 80)

# Compute and display aggregate benchmark averages
print("\n" + "=" * 30 + " SUMMARY AVERAGES " + "=" * 30)
if faithfulness_col in df.columns:
    print(f"Average Faithfulness   : {df[faithfulness_col].mean():.4f}")
if context_recall_col in df.columns:
    print(f"Average Context Recall : {df[context_recall_col].mean():.4f}")
print("=" * 78)