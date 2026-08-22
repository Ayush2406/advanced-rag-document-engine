import pandas as pd

df = pd.read_csv("evaluation/baseline_ragas_score.csv")

for idx, row in df.iterrows():
    print(f"[{idx+1}] Question: {row['user_input']}")
    print(f"    Generated Answer: {row['response']}")
    print(f"    Faithfulness Score: {row.get('faithfulness')}")
    print("-" * 60)