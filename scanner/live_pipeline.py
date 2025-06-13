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

# ✅ FIXED FEATURE ORDER - exact schema Vertex AI expects
FEATURES = [
    "adx", "atr_1h", "atr_normalized", "bollinger_lower", "bollinger_middle", "bollinger_upper",
    "candle_body_pct", "candle_body_size", "close", "ema_12", "ema_26", "ema_crossover_flag",
    "fib_distance_0_236", "fib_distance_0_382", "fib_distance_0_618", "high", "high_low_ratio",
    "low", "lower_bollinger_break", "macd", "macd_signal", "momentum_10", "momentum_50", "open",
    "overbought_rsi", "oversold_rsi", "price_position", "roc", "rolling_volatility_24h", "rolling_volatility_5h",
    "rsi", "short_vs_long_strength", "slope_24h", "slope_5h", "sma_20", "sma_5", "stddev_1h",
    "stochastic_d", "stochastic_k", "trend_acceleration", "upper_bollinger_break", "volume",
    "volume_change_5m", "volume_price_ratio", "volume_surge", "vwap", "wick_lower", "wick_upper"
]

TEXT_FEATURES = [
    "ema_crossover_flag", "overbought_rsi", "oversold_rsi", "upper_bollinger_break", "lower_bollinger_break"
]


FAKE_INSTANCE = [
    22.44, 0.01514, 0.00349, 3.99409, 4.02335, 4.04889,
    0.49999, 0.0052, 4.02, 4.02326, 4.02441, 0,
    -0.01507, -0.00230, 0.00186, 4.03, 1.00324,
    4.01, 0, -5.44e-10, -1.40e-9, 0.0, -0.01247, 4.02,
    0, 0, 0.49999, 0.0, 0.06073, 0.02629,
    49.94, 0.99997, -3.81e-10, -3.33e-10, 4.02335, 4.023, 0.01225,
    50.0, 50.0, 6.94e-11, 0, 49208.11,
    -0.02054, 14108.15, 1.0, 4.93459, 0.00129, 0.00109
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
