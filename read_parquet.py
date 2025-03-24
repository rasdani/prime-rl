import pandas as pd
from transformers import AutoTokenizer


tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

df = pd.read_parquet("data_rollout/step_0/9a2f0ceb-97e0-40d4-9ca9-0805cc9dd686.parquet")

print(list(df.loc[0, "input_tokens"]))
print(tok.decode(list(df.loc[0, "input_tokens"]) + list(df.loc[0, "output_tokens"])))


print(df.columns)
