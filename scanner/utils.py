import joblib
import os
import numpy as np
from pathlib import Path
from django.conf import settings


_model = None

def get_model():
    global _model
    if _model is None:
        try:
            from django.conf import settings

            #MODEL_PATH = os.path.join(settings.BASE_DIR, "scanner", "model", "ml_model.pkl")

            MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/scanner/model")  # Default fallback
            MODEL_PATH = os.path.join(MODEL_DIR, "ml_model.pkl")

            _model = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"❌ Could not load model: {e}")
            _model = None
    return _model

def score_metrics(metrics_dict):
    model = get_model()
    if model is None:
        return 0.0

    try:
        X = np.array([[
            metrics_dict["price_change_5min"],
            metrics_dict["price_change_10min"],
            metrics_dict["price_change_1hr"],
            metrics_dict["price_change_24hr"],
            metrics_dict["price_change_7d"],
            metrics_dict["five_min_relative_volume"],
            metrics_dict["rolling_relative_volume"],
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

import requests

def send_telegram_alert(message: str):
    chat_ids = ['1077594551']  # Add more if needed
    bot_token = '7672687080:AAFWvkwzp-LQE92XdO9vcVa5yWJDUxO17yE'
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    for chat_id in chat_ids:
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }
        response = requests.post(url, data=payload)

        if response.status_code == 200:
            print(f"✅ Telegram alert sent to {chat_id}")
        else:
            print(f"❌ Telegram error: {response.status_code} — {response.content}")


_short_model = None

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(settings.BASE_DIR, "scanner", "model"))
SHORT_MODEL_PATH = os.path.join(MODEL_DIR, "ml_model_short.pkl")


def get_short_model():
    global _short_model
    if _short_model is None:
        try:
            _short_model = joblib.load(SHORT_MODEL_PATH)
        except Exception as e:
            print(f"❌ Could not load SHORT model: {e}")
            _short_model = None
    return _short_model


def score_metrics_short(metrics_dict):
    model = get_short_model()
    if model is None:
        return 0.0

    try:
        X = np.array([[
            metrics_dict["price_change_5min"],
            metrics_dict["price_change_10min"],
            metrics_dict["price_change_1hr"],
            metrics_dict["price_change_24hr"],
            metrics_dict["price_change_7d"],
            metrics_dict["five_min_relative_volume"],
            metrics_dict["rolling_relative_volume"],
            metrics_dict["volume_24h"],
        ]])
        proba = model.predict_proba(X)[0]
        return float(proba[1])
    except Exception as e:
        print(f"❌ Error scoring SHORT metrics: {e}")
        return 0.0
