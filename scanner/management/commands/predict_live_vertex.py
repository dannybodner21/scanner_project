from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from django.utils.timezone import now
from datetime import timedelta
import requests
import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import numpy as np
import json

class Command(BaseCommand):
    help = 'Send latest RickisMetrics to Vertex AI endpoint and get live predictions'

    def handle(self, *args, **kwargs):
        project = 'bodner-main-project'        # <<< your real project id
        endpoint_id = '1612984812077842432'    # <<< your real endpoint id
        region = 'us-central1'                 # <<< Example: 'us-central1'

        # Authenticate and get token
        credentials, _ = google.auth.default()
        credentials.refresh(Request())
        access_token = credentials.token

        # Prediction endpoint URL
        url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/endpoints/{endpoint_id}:predict"

        # Load latest RickisMetrics
        cutoff = now() - timedelta(minutes=10)
        metrics = RickisMetrics.objects.filter(timestamp__gte=cutoff)

        if not metrics.exists():
            self.stdout.write(self.style.WARNING("⚠️ No recent RickisMetrics data found."))
            return

        # Prepare payload
        instances = []
        for metric in metrics:
            instances.append({
                "price": float(metric.price),
                "volume": float(metric.volume),
                "change_5m": float(metric.change_5m),
                "change_1h": float(metric.change_1h),
                "change_24h": float(metric.change_24h),
                "high_24h": float(metric.high_24h),
                "low_24h": float(metric.low_24h),
                "avg_volume_1h": float(metric.avg_volume_1h),
                "relative_volume": float(metric.relative_volume),
                "sma_5": float(metric.sma_5),
                "sma_20": float(metric.sma_20),
                "ema_12": float(metric.ema_12),
                "ema_26": float(metric.ema_26),
                "macd": float(metric.macd),
                "macd_signal": float(metric.macd_signal),
                "rsi": float(metric.rsi),
                "stochastic_k": float(metric.stochastic_k),
                "stochastic_d": float(metric.stochastic_d),
                "support_level": float(metric.support_level),
                "resistance_level": float(metric.resistance_level),
                "stddev_1h": float(metric.stddev_1h),
                "price_slope_1h": float(metric.price_slope_1h),
                "atr_1h": float(metric.atr_1h),
            })

        payload = {
            "instances": instances
        }

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        # Send POST request
        response = requests.post(url, headers=headers, json=payload)

        # Print prediction results
        if response.status_code == 200:
            predictions = response.json()
            self.stdout.write(self.style.SUCCESS(f"✅ Predictions received:"))
            self.stdout.write(json.dumps(predictions, indent=2))
        else:
            self.stdout.write(self.style.ERROR(f"❌ Error {response.status_code}: {response.text}"))
