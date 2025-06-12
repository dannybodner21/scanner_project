import pandas as pd
import numpy as np
import os

# Correct folders
DATA_DIR = "final_training_data"
SAVE_DIR = "final_training_data_with_results"
os.makedirs(SAVE_DIR, exist_ok=True)

TAKE_PROFIT = 0.04  # 4% profit
STOP_LOSS = 0.02    # 2% loss

def evaluate_long_trade(df):
    results = []
    prices = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    for i in range(len(df)):
        entry = prices[i]
        target = entry * (1 + TAKE_PROFIT)
        stop = entry * (1 - STOP_LOSS)

        trade_result = None

        for j in range(i+1, len(df)):
            high_j = highs[j]
            low_j = lows[j]

            if high_j >= target and low_j <= stop:
                trade_result = 0
                break
            elif high_j >= target:
                trade_result = 1
                break
            elif low_j <= stop:
                trade_result = 0
                break

        if trade_result is None:
            trade_result = 0

        results.append(trade_result)

    return results

for filename in os.listdir(DATA_DIR):
    if filename.endswith('.csv'):
        print(f"Processing {filename}...")
        file_path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(file_path)
        df['long_result'] = evaluate_long_trade(df)
        output_path = os.path.join(SAVE_DIR, filename)
        df.to_csv(output_path, index=False)

print("✅ All files processed and saved with long trade results.")
