from django.core.management.base import BaseCommand
import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Command(BaseCommand):
    help = 'Evaluate the short model predictions against short_result in full_holdout.csv'

    def handle(self, *args, **kwargs):
        self.stdout.write("📊 Evaluating Short Model...")

        full_df = pd.read_csv("full_holdout.csv")
        if "short_result" not in full_df.columns:
            raise ValueError("❌ 'short_result' column is missing in full_holdout.csv")

        with open("results_short.jsonl", "r") as f:
            results = [
                {
                    "predicted_label": json.loads(line)["prediction"].get("predicted_label"),
                    "probability": json.loads(line)["prediction"].get("probability")
                }
                for line in f
            ]

        pred_df = pd.DataFrame(results)
        if len(full_df) != len(pred_df):
            raise ValueError("❌ Row count mismatch between full_holdout.csv and results_short.jsonl")

        merged_df = pd.concat([full_df.reset_index(drop=True), pred_df], axis=1)

        y_true = merged_df["short_result"]
        y_pred = merged_df["predicted_label"]

        self.stdout.write(f"✅ Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
        self.stdout.write(f"✅ Precision: {precision_score(y_true, y_pred):.4f}")
        self.stdout.write(f"✅ Recall:    {recall_score(y_true, y_pred):.4f}")
        self.stdout.write(f"✅ F1 Score:  {f1_score(y_true, y_pred):.4f}")

        merged_df.to_csv("scored_short_results.csv", index=False)
        self.stdout.write("📁 Saved: scored_short_results.csv")
