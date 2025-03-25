import joblib
import os
import numpy as np

from pathlib import Path

# Get absolute path to model file
BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "ml_model.pkl"
model = joblib.load(model_path)

def score_metrics(metrics_dict):
    """
    Input: dict with keys matching training data
    Output: probability (float) that this is a successful setup
    """
    try:
        X = np.array([[
            metrics_dict["price_change_5min"],
            metrics_dict["five_min_relative_volume"],
            metrics_dict["price_change_1hr"],
            metrics_dict["market_cap"]
        ]])
        prob = model.predict_proba(X)[0][1]  # probability of label=1
        return prob
    except Exception as e:
        print(f"❌ Error scoring metrics: {e}")
        return 0.0
