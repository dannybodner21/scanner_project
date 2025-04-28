from scanner.models import RickisMetrics
from django.utils.timezone import now
from datetime import timedelta
import joblib
import numpy as np
import threading
import requests

LONG_MODEL_PATH = '/workspace/scanner/xgboost_long_model.pkl'
SHORT_MODEL_PATH = '/workspace/scanner/xgboost_short_model.pkl'
BOT_TOKEN = '7672687080:AAFWvkwzp-LQE92XdO9vcVa5yWJDUxO17yE'
CHAT_ID = '1077594551'

def predict_live_logic():
    print("\n🚀 Loading trained models...")
    long_model = joblib.load(LONG_MODEL_PATH)
    short_model = joblib.load(SHORT_MODEL_PATH)

    print("🚀 Loading latest RickisMetrics...")
    metrics = load_latest_rickismetrics()
    if not metrics.exists():
        print("⚠️ No recent RickisMetrics found.")
        return

    X = preprocess_metrics(metrics)

    long_preds = long_model.predict_proba(X)[:, 1]
    short_preds = short_model.predict_proba(X)[:, 1]

    print("🚀 Sending alerts...")
    for metric, long_confidence, short_confidence in zip(metrics, long_preds, short_preds):
        print(f"🔎 {metric.coin.symbol} — Long: {long_confidence:.2f} | Short: {short_confidence:.2f}")

        if long_confidence > 0.70:
            post_metrics_to_bot(metric, long_confidence, signal_type='LONG')

        if short_confidence > 0.70:
            post_metrics_to_bot(metric, short_confidence, signal_type='SHORT')

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

def post_metrics_to_bot(metric, confidence, signal_type='LONG'):
    signal_emoji = "📈" if signal_type == 'LONG' else "📉"
    message = (
        f"{signal_emoji} {metric.coin.symbol} | {metric.timestamp.strftime('%Y-%m-%d %H:%M')}\n"
        f"{signal_type.capitalize()} Confidence: {confidence:.2f}"
    )
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
