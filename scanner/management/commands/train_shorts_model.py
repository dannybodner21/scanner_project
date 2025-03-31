from django.core.management.base import BaseCommand
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scanner.models import BacktestResult, Metrics
import joblib
import os
from django.conf import settings

MODEL_DIR = os.path.join(settings.BASE_DIR, "scanner", "model")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "ml_model_short.pkl")

class Command(BaseCommand):
    help = "Train ML model to detect short setups"

    def handle(self, *args, **kwargs):
        results = BacktestResult.objects.select_related('entry_metrics') \
            .filter(success__isnull=False) \
            .order_by('?')[:10000]



        X = []
        y = []

        for r in results:
            m = r.entry_metrics
            if not m:
                continue

            features = [
                m.price_change_5min,
                m.price_change_10min,
                m.price_change_1hr,
                m.price_change_24hr,
                m.price_change_7d,
                m.five_min_relative_volume,
                m.rolling_relative_volume,
                m.twenty_min_relative_volume,
                m.volume_24h,
            ]

            if any(f is None for f in features):
                continue

            label = 1 if r.success else 0  # This now reflects proper short success
            X.append(features)
            y.append(label)

        if not X:
            print("❌ No training data found.")
            return

        if len(set(y)) < 2:
            print(f"❌ Not enough class diversity. Classes: {set(y)}")
            return

        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X, y)

        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"✅ SHORT model trained. Accuracy: {acc:.2f}")

        joblib.dump(model, MODEL_PATH)
        print(f"📦 SHORT model saved to {MODEL_PATH}")
