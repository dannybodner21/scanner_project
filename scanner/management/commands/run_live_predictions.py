from django.core.management.base import BaseCommand
from scanner.models import Metrics
from django.utils.timezone import now, timedelta
import joblib
import os
import numpy as np

MODEL_PATH = os.path.join("scanner", "model", "ml_model.pkl")

REQUIRED_FIELDS = [
    "price_change_5min",
    "price_change_10min",
    "price_change_1hr",
    "five_min_relative_volume",
    "rolling_relative_volume",
    "twenty_min_relative_volume",
    "price_change_24hr",
    "price_change_7d",
    "volume_24h",
]

class Command(BaseCommand):
    help = "Run live predictions on latest metrics"

    def handle(self, *args, **options):
        if not os.path.exists(MODEL_PATH):
            self.stderr.write(f"❌ Model not found at {MODEL_PATH}")
            return

        model = joblib.load(MODEL_PATH)

        cutoff = now() - timedelta(minutes=5)
        latest_metrics = (
            Metrics.objects.filter(timestamp__gte=cutoff)
            .order_by("coin", "-timestamp")
            .distinct("coin")
        )

        signals = []

        for metric in latest_metrics:
            features = []

            for field in REQUIRED_FIELDS:
                value = getattr(metric, field)
                if value is None:
                    break
                features.append(float(value))
            else:
                X = np.array([features])
                prediction = model.predict(X)[0]
                confidence = model.predict_proba(X)[0][1]  # probability of success

                if prediction == 1 and confidence >= 0.6:
                    signals.append((metric.coin.symbol, confidence, metric.timestamp))

        if signals:
            self.stdout.write("💰 BUY SIGNALS:")
            for symbol, conf, ts in signals:
                self.stdout.write(f"➡️ {symbol} | Confidence: {conf:.2f} | Time: {ts}")
        else:
            self.stdout.write("📭 No high-confidence signals found.")
