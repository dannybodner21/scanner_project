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
    "adx", "atr_1h", "atr_normalized", "bollinger_lower", "bollinger_middle", "bollinger_upper",
    "candle_body_pct", "candle_body_size", "close", "ema_12", "ema_26", "ema_crossover_flag",
    "fib_distance_0_236", "fib_distance_0_382", "fib_distance_0_618", "high", "high_low_ratio",
    "low", "lower_bollinger_break", "macd", "macd_signal", "momentum_10", "momentum_50", "open",
    "overbought_rsi", "oversold_rsi", "price_position", "roc", "rolling_volatility_24h", "rolling_volatility_5h",
    "rsi", "short_vs_long_strength", "slope_24h", "slope_5h", "sma_20", "sma_5", "stddev_1h",
    "stochastic_d", "stochastic_k", "trend_acceleration", "upper_bollinger_break", "volume",
    "volume_change_5m", "volume_price_ratio", "volume_surge", "vwap", "wick_lower", "wick_upper"
]

FAKE_INSTANCE = {
    "adx": 22.44,
    "atr_1h": 0.01514,
    "atr_normalized": 0.00349,
    "bollinger_lower": 3.99409,
    "bollinger_middle": 4.02335,
    "bollinger_upper": 4.04889,
    "candle_body_pct": 0.49999,
    "candle_body_size": 0.0052,
    "close": 4.02,
    "ema_12": 4.02326,
    "ema_26": 4.02441,
    "ema_crossover_flag": "0",
    "fib_distance_0_236": -0.01507,
    "fib_distance_0_382": -0.00230,
    "fib_distance_0_618": 0.00186,
    "high": 4.03,
    "high_low_ratio": 1.00324,
    "low": 4.01,
    "lower_bollinger_break": "0",
    "macd": -5.44e-10,
    "macd_signal": -1.40e-9,
    "momentum_10": 0.0,
    "momentum_50": -0.01247,
    "open": 4.02,
    "overbought_rsi": "0",
    "oversold_rsi": "0",
    "price_position": 0.49999,
    "roc": 0.0,
    "rolling_volatility_24h": 0.06073,
    "rolling_volatility_5h": 0.02629,
    "rsi": 49.94,
    "short_vs_long_strength": 0.99997,
    "slope_24h": -3.81e-10,
    "slope_5h": -3.33e-10,
    "sma_20": 4.02335,
    "sma_5": 4.023,
    "stddev_1h": 0.01225,
    "stochastic_d": 50.0,
    "stochastic_k": 50.0,
    "trend_acceleration": 6.94e-11,
    "upper_bollinger_break": "0",
    "volume": 49208.11,
    "volume_change_5m": -0.02054,
    "volume_price_ratio": 14108.15,
    "volume_surge": 1.0,
    "vwap": 4.93459,
    "wick_lower": 0.00129,
    "wick_upper": 0.00109
}

def get_google_jwt_token():
    service_account_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())
    return credentials.token

def run_live_pipeline(request=None):
    print(f"⏱ Running fake test at: {datetime.now()}")

    try:
        jwt_token = get_google_jwt_token()
        vertex_url = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict"
        headers = {"Authorization": f"Bearer {jwt_token}", "Content-Type": "application/json"}

        payload = {"instances": [FAKE_INSTANCE]}
        response = requests.post(vertex_url, headers=headers, json=payload)
        response.raise_for_status()

        predictions = response.json().get("predictions", [])
        if predictions:
            pred = predictions[0]
            confidence = pred["scores"][0]
            print(f"✅ FAKE TEST SUCCESS — Confidence: {confidence:.4f}")
        else:
            print("⚠ No predictions returned.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_live_pipeline()
