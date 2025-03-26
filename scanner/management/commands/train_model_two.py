from django.core.management.base import BaseCommand
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scanner.models import SuccessfulMove, Metrics
import pandas as pd
import joblib
import os
from decimal import Decimal

class Command(BaseCommand):
    help = "Train improved ML model using RandomForest"

    def handle(self, *args, **kwargs):
        # FIXED: removed select_related
        moves = SuccessfulMove.objects.all()

        X = []
        y = []

        for move in moves:
            m = move.entry_metrics
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
            y.append(1)

        if not X:
            print("❌ No training data found. Check SuccessfulMove and Metrics")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ RandomForest trained. Accuracy: {acc:.2f}")

        model_path = os.path.join("/workspace/tmp", "ml_model.pkl")
        joblib.dump(model, model_path)
        print(f"📦 Model saved to {model_path}")
