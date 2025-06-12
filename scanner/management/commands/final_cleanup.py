import os
import pandas as pd

# Create output folder if it doesn't exist
os.makedirs('final_training_data', exist_ok=True)

# Process each enriched file
for filename in os.listdir('exported_data_enriched'):
    if filename.endswith('.csv'):
        symbol = filename.replace('.csv', '')
        print(f"Processing {symbol}")

        file_path = os.path.join('exported_data_enriched', filename)
        df = pd.read_csv(file_path)

        # Drop first 288 rows
        df_clean = df.iloc[288:].copy()

        # Check for nulls
        null_count = df_clean.isnull().sum().sum()
        if null_count == 0:
            print(f"✅ {symbol}: Clean after dropping first 288 rows.")
        else:
            print(f"⚠ {symbol}: {null_count} missing values still remain after trimming.")

        # Export to final_training_data
        out_path = os.path.join('final_training_data', filename)
        df_clean.to_csv(out_path, index=False)

print("🚀 ALL FILES CLEANED AND SAVED TO final_training_data/")
