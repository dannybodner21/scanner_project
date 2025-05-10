from google.cloud import aiplatform
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === CONFIG ===
PROJECT_ID = 'bodner-main-project'
ENDPOINT_ID = '8508061657660915712'  # long model endpoint
REGION = 'us-central1'

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

# === Init Vertex AI ===
aiplatform.init(project=PROJECT_ID, location=REGION)
endpoint = aiplatform.Endpoint(ENDPOINT_ID)

# === Load input ===
df = pd.read_csv("full_holdout.csv")
X = df[features]
y_true = df["long_result"]

# === Predict ===
probabilities = []

for i, row in X.iterrows():
    instance = {k: float(row[k]) for k in features}
    try:
        response = endpoint.predict([instance])
        prediction = response.predictions[0]
        probabilities.append(prediction["probability"])
    except Exception as e:
        print(f"❌ Error on row {i}: {e}")
        probabilities.append(None)

# === Keep only confident live trades (prob > 0.5) ===
df["probability"] = probabilities
df_live_trades = df[df["probability"] > 0.7].copy()

# These are the trades we would take
y_true_live = df_live_trades["long_result"]
y_pred_live = [1] * len(df_live_trades)  # every confident prediction is treated as "go long"

# === Evaluate only trades taken
print("📊 Long Model (Live Trades Only, prob > 0.5):")
print(f"✅ Trades taken: {len(df_live_trades)}")
print(f"✅ Accuracy:  {accuracy_score(y_true_live, y_pred_live):.4f}")
print(f"✅ Precision: {precision_score(y_true_live, y_pred_live):.4f}")
print(f"✅ Recall:    {recall_score(y_true_live, y_pred_live):.4f}")
print(f"✅ F1 Score:  {f1_score(y_true_live, y_pred_live):.4f}")

df_live_trades.to_csv("scored_long_results_live.csv", index=False)
print("📁 Saved: scored_long_results_live.csv")
