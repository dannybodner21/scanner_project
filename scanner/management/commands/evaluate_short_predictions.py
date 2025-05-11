from django.core.management.base import BaseCommand
import os
import json
import pandas as pd
from google.cloud import aiplatform

# === CONFIG ===
PROJECT_ID = 'bodner-main-project'
REGION = 'us-central1'
ENDPOINT_ID = '5322327871249711104'  # short model endpoint
INPUT_FILE = 'full_holdout.csv'
CONFIDENCE_THRESHOLD = 0.5
BATCH_SIZE = 100

# === MODEL INPUT FEATURES ===
features = [
    "price", "volume", "change_5m", "change_1h", "change_24h",
    "high_24h", "low_24h", "avg_volume_1h", "relative_volume",
    "rsi", "macd", "macd_signal", "stochastic_k", "stochastic_d",
    "support_level", "resistance_level", "sma_5", "sma_20",
    "stddev_1h", "atr_1h", "obv", "change_since_high", "change_since_low",
    "fib_distance_0_236", "fib_distance_0_382", "fib_distance_0_5",
    "fib_distance_0_618", "fib_distance_0_786", "open", "close"
]

def run_short_prediction_evaluation():
    # === Handle JSON credentials ===
    raw_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if raw_creds:
        with open("/tmp/adc.json", "w") as f:
            f.write(raw_creds)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/adc.json"

    # === Init Vertex ===
    aiplatform.init(project=PROJECT_ID, location=REGION)
    endpoint = aiplatform.Endpoint(ENDPOINT_ID)

    # === Load and clean data ===
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=features + ["short_result"])

    # === Batch predict ===
    predictions = []
    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i:i+BATCH_SIZE]
        instances = [{k: float(row[k]) for k in features} for _, row in batch.iterrows()]
        try:
            response = endpoint.predict(instances)
            preds = response.predictions
            for pred in preds:
                print("🔍 Raw prediction:", pred)
                if isinstance(pred, dict) and "classes" in pred and "scores" in pred:
                    class_index = pred["classes"].index("true")
                    predictions.append(float(pred["scores"][class_index]))
                elif isinstance(pred, list) and len(pred) == 2:
                    predictions.append(float(pred[1]))
                elif isinstance(pred, (int, float)):
                    predictions.append(1.0 if int(pred) == 1 else 0.0)
                else:
                    predictions.append(None)
        except Exception as e:
            print(f"❌ Error on batch {i}-{i+BATCH_SIZE}: {e}")
            predictions.extend([None] * len(instances))

    # === Store results and filter trades
    df["probability"] = predictions
    df_live = df[df["probability"] > CONFIDENCE_THRESHOLD].copy()

    trades_taken = len(df_live)
    wins = df_live["short_result"].sum()
    losses = trades_taken - wins

    # === Report
    print(f"\n📊 Short Model Evaluation (All entries, prob > {CONFIDENCE_THRESHOLD}):")
    print(f"✅ Total rows evaluated: {len(df)}")
    print(f"✅ Trades taken:         {trades_taken}")
    print(f"🏁 Wins:                 {wins}")
    print(f"❌ Losses:               {losses}")
    if trades_taken > 0:
        print(f"🎯 Win Rate:             {wins / trades_taken:.2%}")
    else:
        print("⚠️ No trades met the confidence threshold.")

    # === Save live trades only
    df_live.to_csv("scored_short_results_live.csv", index=False)
    print("📁 Saved: scored_short_results_live.csv")

# === Django command wrapper ===
class Command(BaseCommand):
    help = 'Evaluate short model predictions on all entries and summarize confident trades'

    def handle(self, *args, **kwargs):
        run_short_prediction_evaluation()
