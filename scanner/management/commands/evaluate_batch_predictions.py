import json
import pandas as pd
from datetime import datetime
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scanner.models import RickisMetrics

class Command(BaseCommand):
    help = 'Evaluate Vertex AI batch predictions against RickisMetrics ground truth'

    def handle(self, *args, **kwargs):
        # === 1. Load JSONL predictions ===
        jsonl_path = "results.jsonl"  # Adjust if needed
        predictions = []

        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                instance = data.get("instance", {})
                prediction = data.get("prediction", {})

                predictions.append({
                    "timestamp": instance.get("timestamp"),
                    "price": instance.get("price"),
                    "coin_symbol": instance.get("coin_symbol"),
                    "probability": prediction.get("probability"),
                    "predicted_label": prediction.get("predicted_label"),
                })

        pred_df = pd.DataFrame(predictions)

        # Ensure timestamp is aware and parsed
        pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], errors="coerce")
        pred_df = pred_df.dropna(subset=["timestamp"])
        pred_df["timestamp"] = pred_df["timestamp"].apply(make_aware)

        self.stdout.write(f"✅ Loaded {len(pred_df)} predictions")

        # === 2. Load ground truth ===
        start = make_aware(datetime(2025, 4, 30))
        end = make_aware(datetime(2025, 5, 3))

        truth_qs = RickisMetrics.objects.filter(
            timestamp__gte=start,
            timestamp__lt=end
        ).values("price", "timestamp", "coin__symbol", "result")

        truth_df = pd.DataFrame.from_records(truth_qs)
        truth_df.rename(columns={"coin__symbol": "coin_symbol"}, inplace=True)

        self.stdout.write(f"✅ Loaded {len(truth_df)} ground truth rows")

        # === 3. Merge on price, timestamp, coin_symbol ===
        merged_df = pd.merge(
            pred_df,
            truth_df,
            on=["price", "timestamp", "coin_symbol"],
            how="inner"
        )

        self.stdout.write(f"🔗 Merged {len(merged_df)} records")

        if merged_df.empty:
            self.stdout.write("❌ No matching rows between predictions and ground truth.")
            return

        # === 4. Evaluate ===
        y_true = merged_df["result"]
        y_pred = merged_df["predicted_label"]

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        self.stdout.write("📊 Evaluation Metrics:")
        self.stdout.write(f"   ✅ Accuracy:  {acc:.4f}")
        self.stdout.write(f"   ✅ Precision: {precision:.4f}")
        self.stdout.write(f"   ✅ Recall:    {recall:.4f}")
        self.stdout.write(f"   ✅ F1 Score:  {f1:.4f}")
