from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from django.utils.timezone import now
from datetime import timedelta
import joblib
import numpy as np
import threading
import requests

# === Config ===
MODEL_PATH = '/workspace/scanner/short_model.joblib'

# === Command ===
class Command(BaseCommand):
    help = 'Predict short trades using AWS short model and send alerts'

    def handle(self, *args, **kwargs):
        print("\n📦 Loading AWS short model...")
        model = joblib.load(MODEL_PATH)

        print("🕒 Loading recent RickisMetrics...")
        cutoff = now() - timedelta(minutes=5)
        metrics = RickisMetrics.objects.filter(timestamp__gte=cutoff)

        if not metrics.exists():
            print("⚠️ No recent metrics found.")
            return

        instances = []
        symbols = []
        timestamps = []

        for metric in metrics:
            try:
                row = [
                    float(metric.price),
                    float(metric.volume),
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
                ]
                instances.append(row)
                symbols.append(metric.coin.symbol)
                timestamps.append(metric.timestamp.strftime('%Y-%m-%d %H:%M'))
            except:
                continue

        if not instances:
            print("❌ No valid entries to predict.")
            return

        print("🧠 Predicting...")
        preds = model.predict_proba(np.array(instances))[:, 1]

        for sym, conf, ts in zip(symbols, preds, timestamps):
            print(f"🔻 {sym} — Short Confidence: {conf:.4f}")
            if conf > 0.70:
                message = [f"🔻 SHORT ALERT: {sym} | {ts}\nConfidence: {conf:.2f}"]
                send_text(message)

        print("✅ Prediction complete.")


# bot message notificagtions
def send_text(true_triggers_two):

    if len(true_triggers_two) > 0:

        # telegram bot information
        chat_id_danny = '1077594551'
        #chat_id_ricki = '1054741134'
        #chat_ids = [chat_id_danny, chat_id_ricki]
        chat_ids = [chat_id_danny]
        bot_token = '7672687080:AAFWvkwzp-LQE92XdO9vcVa5yWJDUxO17yE'
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"


        # send message to myself and Ricki
        message = ""
        for chat_id in chat_ids:
            for trigger in true_triggers_two:

                message += trigger + " "

            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown",
            }

            response = requests.post(url, data=payload)

            if response.status_code == 200:
                print("Message sent successfully.")
            else:
                print(f"Failed to send message: {response.content}")

    return
