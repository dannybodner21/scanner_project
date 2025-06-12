import os
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from django.core.management.base import BaseCommand
from scanner.models import Coin, LiveModelMetrics
from google.auth.transport.requests import Request
from google.oauth2 import service_account

# ---------------- CONFIG ----------------

COINAPI_KEY = "01293e2a-dcf1-4e81-8310-c6aa9d0cb743"
PROJECT_ID = "healthy-mark-446922-p8"
REGION = "us-central1"
ENDPOINT_ID = "000000000000000"

# Your 16 coin list
SYMBOLS = [
    "BTC", "ETH", "XRP", "LTC", "SOL", "DOGE", "PEPE", "ADA",
    "XLM", "SUI", "LINK", "AVAX", "DOT", "SHIB", "HBAR", "UNI"
]

COINAPI_MAP = {symbol: f"BINANCE_SPOT_{symbol}_USDT" for symbol in SYMBOLS}

# ----------------------------------------

def get_google_jwt_token():
    audience = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict"
    service_account_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())
    return credentials.token

def fetch_ohlcv(symbol):
    coinapi_symbol = COINAPI_MAP[symbol]
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    end = now_utc.replace(second=0, microsecond=0)
    start = end - timedelta(minutes=5)

    url = (
        f"https://rest.coinapi.io/v1/ohlcv/{coinapi_symbol}/history"
        f"?period_id=5MIN&time_start={start.isoformat()}&time_end={end.isoformat()}&limit=1"
    )

    headers = {"X-CoinAPI-Key": COINAPI_KEY}
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"❌ Failed fetch for {symbol}: {resp.text}")
        return None

    data = resp.json()
    if not data:
        return None

    item = data[0]
    return {
        "timestamp": datetime.fromisoformat(item['time_period_start']),
        "open": Decimal(item['price_open']),
        "high": Decimal(item['price_high']),
        "low": Decimal(item['price_low']),
        "close": Decimal(item['price_close']),
        "volume": Decimal(item['volume_traded'])
    }

# ------------ CALCULATIONS ------------

def calculate_features(prices):
    df = pd.DataFrame(prices)

    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['ema_crossover_flag'] = (df['ema_12'] > df['ema_26']).astype(int)

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    high_roll = df['high'].rolling(14).max()
    low_roll = df['low'].rolling(14).min()
    df['stochastic_k'] = 100 * (df['close'] - low_roll) / (high_roll - low_roll)
    df['stochastic_d'] = df['stochastic_k'].rolling(3).mean()

    df['bollinger_middle'] = df['close'].rolling(20).mean()
    df['bollinger_std'] = df['close'].rolling(20).std()
    df['bollinger_upper'] = df['bollinger_middle'] + 2 * df['bollinger_std']
    df['bollinger_lower'] = df['bollinger_middle'] - 2 * df['bollinger_std']

    df['atr_1h'] = (df['high'] - df['low']).rolling(12).mean()
    df['stddev_1h'] = df['close'].rolling(12).std()

    df.fillna(0, inplace=True)
    latest = df.iloc[-1]

    return {
        "sma_5": latest['sma_5'],
        "sma_20": latest['sma_20'],
        "ema_12": latest['ema_12'],
        "ema_26": latest['ema_26'],
        "ema_crossover_flag": latest['ema_crossover_flag'],
        "rsi": latest['rsi'],
        "macd": latest['macd'],
        "macd_signal": latest['macd_signal'],
        "stochastic_k": latest['stochastic_k'],
        "stochastic_d": latest['stochastic_d'],
        "bollinger_upper": latest['bollinger_upper'],
        "bollinger_middle": latest['bollinger_middle'],
        "bollinger_lower": latest['bollinger_lower'],
        "atr_1h": latest['atr_1h'],
        "stddev_1h": latest['stddev_1h']
    }

# ------------ PREDICT ---------------

def call_vertex_api(instances):
    jwt_token = get_google_jwt_token()
    endpoint_url = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict"
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json"
    }
    body = {"instances": instances}
    response = requests.post(endpoint_url, headers=headers, json=body)
    response.raise_for_status()
    return response.json()

# ------------ PIPELINE ---------------

def run_pipeline():
    for symbol in SYMBOLS:
        print(f"\n🟢 Processing {symbol}...")

        # Fetch prices (you should really load 50+ candles for indicators, simplify here for demo)
        prices = []
        for i in range(50, 0, -1):
            time_end = datetime.utcnow().replace(second=0, microsecond=0, tzinfo=timezone.utc) - timedelta(minutes=5 * i)
            time_start = time_end
            url = (
                f"https://rest.coinapi.io/v1/ohlcv/{COINAPI_MAP[symbol]}/history"
                f"?period_id=5MIN&time_start={time_start.isoformat()}&time_end={time_end.isoformat()}&limit=1"
            )
            headers = {"X-CoinAPI-Key": COINAPI_KEY}
            resp = requests.get(url, headers=headers)
            if resp.status_code == 200 and resp.json():
                item = resp.json()[0]
                prices.append({
                    "timestamp": datetime.fromisoformat(item['time_period_start']),
                    "open": Decimal(item['price_open']),
                    "high": Decimal(item['price_high']),
                    "low": Decimal(item['price_low']),
                    "close": Decimal(item['price_close']),
                    "volume": Decimal(item['volume_traded'])
                })
            else:
                print(f"⚠ Failed candle at {time_end} for {symbol}")

        if len(prices) < 20:
            print(f"❌ Not enough data for {symbol}")
            continue

        features = calculate_features(prices)
        print(f"✅ Features calculated for {symbol}")

        # Predict
        instance = {k: float(v) for k, v in features.items()}
        prediction = call_vertex_api([instance])
        confidence = prediction['predictions'][0]
        print(f"🟣 Prediction for {symbol}: {confidence}")

# ------------- Django mgmt command ------------

class Command(BaseCommand):
    help = "Run full live pipeline"

    def handle(self, *args, **kwargs):
        run_pipeline()
