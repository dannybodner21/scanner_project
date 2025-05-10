from google.cloud import aiplatform
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === CONFIG ===


PROJECT_ID = 'bodner-main-project'
ENDPOINT_ID = '8508061657660915712'
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
predicted_labels = []
probabilities = []

for i, row in X.iterrows():
    instance = {k: float(row[k]) for k in features}
    try:
        response = endpoint.predict([instance])
        prediction = response.predictions[0]
        predicted_labels.append(prediction["predicted_label"])
        probabilities.append(prediction["probability"])
    except Exception as e:
        print(f"❌ Error on row {i}: {e}")
        predicted_labels.append(None)
        probabilities.append(None)

# === Score ===
df["predicted_label"] = predicted_labels
df["probability"] = probabilities

valid_rows = df["predicted_label"].notnull()
y_pred = df.loc[valid_rows, "predicted_label"]
y_true_valid = df.loc[valid_rows, "long_result"]

print("📊 Long Model Evaluation:")
print("✅ Accuracy:  ", accuracy_score(y_true_valid, y_pred))
print("✅ Precision: ", precision_score(y_true_valid, y_pred))
print("✅ Recall:    ", recall_score(y_true_valid, y_pred))
print("✅ F1 Score:  ", f1_score(y_true_valid, y_pred))

df.to_csv("scored_long_results.csv", index=False)
print("📁 Saved: scored_long_results.csv")
