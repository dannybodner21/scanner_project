from django.core.management.base import BaseCommand
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scanner.models import BacktestResult, Metrics
import pandas as pd
import joblib
import os
from pathlib import Path
from django.conf import settings


# 🔐 Set model directory and create it BEFORE anything else
MODEL_DIR = os.path.join(settings.BASE_DIR, "scanner", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "ml_model.pkl")

class Command(BaseCommand):
    help = "Train ML model using BacktestResult (includes wins and losses)"

    def handle(self, *args, **kwargs):
        results = BacktestResult.objects.select_related('entry_metrics').filter(success__isnull=False)

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

            X.append(features)
            y.append(1 if r.success else 0)

        if not X:
            print("❌ No training data found. Run backtest first.")
            return

        if len(set(y)) < 2:
            print(f"❌ Not enough class diversity in training data. Classes: {set(y)}")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Model trained. Accuracy: {acc:.2f}")

        print(f"💾 Saving model to: {MODEL_PATH}")
        print(f"📁 Directory exists? {os.path.exists(os.path.dirname(MODEL_PATH))}")
        print(f"📄 Will overwrite file? {os.path.exists(MODEL_PATH)}")

        joblib.dump(model, MODEL_PATH)
        print(f"📦 Model saved to {MODEL_PATH}")
