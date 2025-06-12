import pandas as pd

# Path to your original parquet file
parquet_file = "/Users/danielbodner/Desktop/scanner_project/vertex_training_data.parquet"

# Load parquet
df = pd.read_parquet(parquet_file)

# Drop timestamp only
df_clean = df.drop(columns=['timestamp'])

# Output path
output_parquet = "/Users/danielbodner/Desktop/scanner_project/vertex_training_data_clean.parquet"

# Save as Parquet
df_clean.to_parquet(output_parquet, index=False)

print(f"✅ Parquet exported: {output_parquet}")
