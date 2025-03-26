import os
import glob
import pandas as pd

# Get list of all parquet files in the step directories
base_dir = "data"
parquet_files = glob.glob(os.path.join(base_dir, "step_0", "*.parquet"))

# Set to store unique token sequences (first 30 tokens from the first row)
unique_token_sequences = set()

total = 0
for file in parquet_files:
    df = pd.read_parquet(file)
    
    for i in range(len(df)):
        total += 1
        # Extract only the first 30 tokens from the first row's "input_tokens"
        tokens = df.loc[i, "input_tokens"][:2]
        token_tuple = tuple(tokens)  # Convert to tuple to make it hashable
        unique_token_sequences.add(token_tuple)

# Compute the number of unique token sequences
num_unique_sequences = len(unique_token_sequences)
print("Number of unique input token sequences (first 30 tokens):", num_unique_sequences)

print("total", total)
