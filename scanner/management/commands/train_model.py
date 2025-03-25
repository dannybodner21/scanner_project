import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from scanner.models import SuccessfulMove
from django.core.management.base import BaseCommand
from pathlib import Path

class Command(BaseCommand):
    help = "Train a model using successful vs non-successful metric patterns"

    def handle(self, *args, **kwargs):
        # Pull successful examples
        success_data = SuccessfulMove.objects.filter(move_type="long")
        success_rows = []
        for row in success_data:
            m = row.metrics
            success_rows.append({
                "price_change_5min": m.get("price_change_5min", 0),
                "five_min_relative_volume": m.get("five_min_relative_volume", 0),
                "price_change_1hr": m.get("price_change_1hr", 0),
                "market_cap": m.get("market_cap", 0),
                "label": 1
            })

        # Now pull negative examples (from Metrics that didn’t fire success signals)
        from scanner.models import Metrics
        from random import sample

        all_metrics = list(Metrics.objects.all().order_by("?")[:len(success_rows)*2])
        fail_rows = []
        for m in all_metrics:
            if m.price_change_5min is None: continue
            fail_rows.append({
                "price_change_5min": m.price_change_5min,
                "five_min_relative_volume": m.five_min_relative_volume,
                "price_change_1hr": m.price_change_1hr,
                "market_cap": float(m.market_cap or 0),
                "label": 0
            })

        # Combine and train
        df = pd.DataFrame(success_rows + fail_rows)
        print(f"Before drop: {len(df)} rows")
        df.dropna(inplace=True)
        print(f"After drop: {len(df)} rows")
        X = df.drop(columns=["label"])
        y = df["label"]

        model = LogisticRegression()
        model.fit(X, y)


        # Save the model inside the scanner directory
        BASE_DIR = Path(__file__).resolve().parent.parent.parent  # adjust based on your file location
        joblib.dump(model, "/tmp/ml_model.pkl")
        print(f"✅ Model trained and saved to /tmp/ml_model.pkl")
