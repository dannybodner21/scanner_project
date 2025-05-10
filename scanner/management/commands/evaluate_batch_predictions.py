import pandas as pd
import json

# === Step 1: Load the full holdout dataset (with features + long_result) ===
full_df = pd.read_csv("full_holdout.csv")

# Select a few identifying features to compare (must match model inputs)
sample_features = ["price", "rsi", "macd", "volume", "change_5m"]
sample_features = [f for f in sample_features if f in full_df.columns]

# === Step 2: Load predictions from results.jsonl ===
pred_input_rows = []
with open("results.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        instance = data.get("instance", {})
        pred_input_rows.append(instance)

pred_input_df = pd.DataFrame(pred_input_rows)

# === Step 3: Compare row-by-row ===
n = min(len(full_df), len(pred_input_df))

mismatches = 0
for i in range(n):
    for feature in sample_features:
        full_val = full_df.iloc[i][feature]
        pred_val = pred_input_df.iloc[i].get(feature)
        if pd.isna(full_val) or pd.isna(pred_val):
            continue
        if abs(float(full_val) - float(pred_val)) > 1e-6:
            mismatches += 1
            print(f"❌ Mismatch at row {i}, feature '{feature}': full={full_val}, pred={pred_val}")
            if mismatches > 10:
                print("❗Too many mismatches — stopping check.")
                break
    if mismatches > 10:
        break

# === Final summary ===
if mismatches == 0:
    print("✅ Input alignment confirmed. You can safely evaluate the predictions.")
else:
    print(f"⚠️ Detected {mismatches} mismatches — batch prediction may not align with full_holdout.csv")
