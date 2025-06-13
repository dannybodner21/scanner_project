import os
import json
import requests
import numpy as np
from datetime import datetime
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# CONFIG
PROJECT_ID = "healthy-mark-446922-p8"
LOCATION = "us-central1"
ENDPOINT_ID = "1878894947566878720"
REGION = LOCATION

FEATURES = [
    "open", "high", "low", "close", "volume", "sma_5", "sma_20",
    "ema_12", "ema_26", "ema_crossover_flag", "rsi", "macd", "macd_signal",
    "stochastic_k", "stochastic_d", "bollinger_upper", "bollinger_middle",
    "bollinger_lower", "adx", "atr_1h", "stddev_1h", "momentum_10",
    "momentum_50", "roc", "rolling_volatility_5h", "rolling_volatility_24h",
    "high_low_ratio", "price_position", "candle_body_size", "candle_body_pct",
    "wick_upper", "wick_lower", "slope_5h", "slope_24h", "trend_acceleration",
    "fib_distance_0_236", "fib_distance_0_382", "fib_distance_0_618", "vwap",
    "volume_price_ratio", "volume_change_5m", "volume_surge", "overbought_rsi",
    "oversold_rsi", "upper_bollinger_break", "lower_bollinger_break",
    "atr_normalized", "short_vs_long_strength"
]

TEXT_FEATURES = [
    "ema_crossover_flag", "overbought_rsi", "oversold_rsi", "upper_bollinger_break", "lower_bollinger_break"
]

# Fake template instance (values match your model feature order)
FAKE_INSTANCE = [
    4.02, 4.03, 4.01, 4.02, 49208.11,
    4.023, 4.02335, 4.02326, 4.02441, 0,
    49.94, -5.44e-10, -1.40e-9, 50.0, 50.0,
    4.04889, 4.02335, 3.99409,
    22.44, 0.01514, 0.01225, 0.0, -0.01247,
    0.0, 0.02629, 0.06073,
    1.00324, 0.49999, 0.0052, 0.49999,
    0.00109, 0.00129, -3.33e-10, -3.81e-10, 6.94e-11,
    -0.01507, -0.00230, 0.00186,
    4.93459, 14108.15, -0.02054, 1.0,
    0.0, 0.0, 0.0, 0.0,
    0.00349, 0.99997
]

def safe_value(val, feature):
    try:
        if val is None or np.isnan(val) or np.isinf(val):
            return 0.0
        if feature in TEXT_FEATURES:
            return int(val)
        return float(val)
    except:
        return 0.0

def get_google_jwt_token():
    service_account_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())
    return credentials.token

def run_live_pipeline():
    print(f"⏱ Live pipeline started: {datetime.now()}")

    try:
        # Use fake instance here
        instance = [safe_value(val, feature) for val, feature in zip(FAKE_INSTANCE, FEATURES)]

        jwt_token = get_google_jwt_token()
        vertex_url = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict"
        headers = {"Authorization": f"Bearer {jwt_token}", "Content-Type": "application/json"}
        payload = {"instances": [instance]}
        response = requests.post(vertex_url, headers=headers, json=payload)
        response.raise_for_status()

        predictions = response.json().get("predictions", [])
        if predictions:
            pred = predictions[0]
            class_idx = pred["classes"].index("true")
            confidence = pred["scores"][class_idx]
            print(f"LONG | FAKE DATA — Confidence: {confidence:.4f}")

    except Exception as e:
        print(f"❌ Error: {e}")

    print("🚀 Live pipeline run complete.")

if __name__ == "__main__":
    run_live_pipeline()
