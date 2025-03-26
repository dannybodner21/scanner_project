import joblib
import os
import numpy as np
from pathlib import Path
from django.conf import settings


MODEL_PATH = os.path.join(settings.BASE_DIR, "scanner", "ml_model.pkl")
model = joblib.load(MODEL_PATH)

def score_metrics(metrics_dict):
    try:
        X = np.array([[
            metrics_dict["price_change_5min"],
            metrics_dict["price_change_10min"],
            metrics_dict["price_change_1hr"],
            metrics_dict["price_change_24hr"],
            metrics_dict["price_change_7d"],
            metrics_dict["five_min_relative_volume"],
            metrics_dict["rolling_relative_volume"],
            metrics_dict["twenty_min_relative_volume"],
            metrics_dict["volume_24h"],
        ]])
        proba = model.predict_proba(X)[0]
        if len(proba) < 2:
            print(f"⚠️ Only one class in prediction: {proba}")
            return float(proba[0]) if model.classes_[0] == 1 else 0.0
        return float(proba[1])
    except Exception as e:
        print(f"❌ Error scoring metrics: {e}")
        return 0.0
