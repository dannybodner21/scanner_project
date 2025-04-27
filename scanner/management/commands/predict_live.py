from scanner.models import RickisMetrics
from django.utils.timezone import now
from datetime import timedelta
import joblib
import numpy as np
import threading
import requests

MODEL_PATH = '/workspace/scanner/xgboost_long_model.pkl'
BOT_TOKEN = '7672687080:AAFWvkwzp-LQE92XdO9vcVa5yWJDUxO17yE'
CHAT_ID = '1077594551'

def predict_live_logic():
    print("\n🚀 Loading trained model...")
    model = joblib.load(MODEL_PATH)

    print("🚀 Loading latest RickisMetrics...")
    metrics = load_latest_rickismetrics()
    if not metrics.exists():
        print("⚠️ No recent RickisMetrics found.")
        return

    X = preprocess_metrics(metrics)
    preds = model.predict_proba(X)[:, 1]

    print("🚀 Sending alerts...")
    for metric, confidence in zip(metrics, preds):
        print(f"🔎 {metric.coin.symbol} — Confidence: {confidence:.2f}")
        if confidence > 0.70:
            post_metrics_to_bot(metric, confidence)

    print("✅ Live prediction complete.")

def load_latest_rickismetrics():
    timestamp_cutoff = now() - timedelta(minutes=10)
    return RickisMetrics.objects.filter(timestamp__gte=timestamp_cutoff)

def preprocess_metrics(metrics_queryset):
    features = []
    for metric in metrics_queryset:
        features.append([
            float(metric.price),
            float(metric.volume),
            float(metric.change_5m),
            float(metric.change_1h),
            float(metric.change_24h),
            float(metric.high_24h),
            float(metric.low_24h),
            float(metric.avg_volume_1h),
            float(metric.relative_volume),
            float(metric.sma_5),
            float(metric.sma_20),
            float(metric.ema_12),
            float(metric.ema_26),
            float(metric.macd),
            float(metric.macd_signal),
            float(metric.rsi),
            float(metric.stochastic_k),
            float(metric.stochastic_d),
            float(metric.support_level),
            float(metric.resistance_level),
            float(metric.stddev_1h),
            float(metric.price_slope_1h),
            float(metric.atr_1h),
        ])
    return np.array(features)

def post_metrics_to_bot(metric, confidence):
    message = f"🚀 {metric.coin.symbol} | {metric.timestamp.strftime('%Y-%m-%d %H:%M')}\nLong Confidence: {confidence:.2f}"
    async_post_to_bot(message)

def async_post_to_bot(text):
    def post():
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": text
        }
        try:
            requests.post(url, json=payload)
        except Exception as e:
            print(f"❌ Error sending Telegram alert: {e}")

    threading.Thread(target=post).start()
