import pandas as pd
import glob

# Correct folder
data_folder = 'final_training_data_with_results/'

# Find all CSV files
files = glob.glob(f'{data_folder}/*.csv')

dfs = []

for file in files:
    print(f'Loading {file}')
    df = pd.read_csv(file)

    # Drop rows with any missing values
    df = df.dropna()

    # Safety check
    if 'long_result' not in df.columns:
        raise ValueError(f"Missing long_result column in {file}")

    dfs.append(df)

# Combine everything
combined_df = pd.concat(dfs, ignore_index=True)

print(f"✅ Total rows after combining: {len(combined_df)}")

# Save to Parquet
combined_df.to_parquet('vertex_training_data.parquet', index=False)
print("🚀 Export complete: vertex_training_data.parquet")
