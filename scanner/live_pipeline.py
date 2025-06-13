import os
import json
import requests
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime
from django.utils.timezone import make_aware
from scanner.models import Coin, LiveModelMetrics
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# ----------------------------------------
# CONFIG
# ----------------------------------------
PROJECT_ID = "healthy-mark-446922-p8"
LOCATION = "us-central1"
ENDPOINT_ID = "1878894947566878720"
REGION = LOCATION

COINAPI_KEY = "01293e2a-dcf1-4e81-8310-c6aa9d0cb743"
BASE_URL = "https://rest.coinapi.io/v1/ohlcv"
SYMBOL_MAP = {
    "BTC": "BINANCE_SPOT_BTC_USDT",
    "ETH": "BINANCE_SPOT_ETH_USDT",
    "XRP": "BINANCE_SPOT_XRP_USDT",
    "LTC": "BINANCE_SPOT_LTC_USDT",
    "SOL": "BINANCE_SPOT_SOL_USDT",
    "DOGE": "BINANCE_SPOT_DOGE_USDT",
    "PEPE": "BINANCE_SPOT_PEPE_USDT",
    "ADA": "BINANCE_SPOT_ADA_USDT",
    "XLM": "BINANCE_SPOT_XLM_USDT",
    "SUI": "BINANCE_SPOT_SUI_USDT",
    "LINK": "BINANCE_SPOT_LINK_USDT",
    "AVAX": "BINANCE_SPOT_AVAX_USDT",
    "DOT": "BINANCE_SPOT_DOT_USDT",
    "SHIB": "BINANCE_SPOT_SHIB_USDT",
    "HBAR": "BINANCE_SPOT_HBAR_USDT",
    "UNI": "BINANCE_SPOT_UNI_USDT"
}

# ----------------------------------------
# JWT TOKEN FUNCTION
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

# ----------------------------------------
# Main pipeline logic
# ----------------------------------------

def run_live_pipeline():
    print(f"⏱ Live pipeline started: {datetime.now()}")

    for symbol, coinapi_symbol in SYMBOL_MAP.items():
        try:
            coin = Coin.objects.get(symbol=symbol)

            url = f"{BASE_URL}/{coinapi_symbol}/latest?period_id=5MIN&limit=50"
            headers = {"X-CoinAPI-Key": COINAPI_KEY}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                raise ValueError("No data returned")

            df = pd.DataFrame([{
                "timestamp": datetime.fromisoformat(candle['time_period_start'].replace("Z", "+00:00")),
                "open": float(candle['price_open']),
                "high": float(candle['price_high']),
                "low": float(candle['price_low']),
                "close": float(candle['price_close']),
                "volume": float(candle['volume_traded'])
            } for candle in data])

            df.sort_values(by="timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)

            # All metrics
            df["sma_5"] = df["close"].rolling(window=5).mean()
            df["sma_20"] = df["close"].rolling(window=20).mean()
            df["ema_12"] = df["close"].ewm(span=12).mean()
            df["ema_26"] = df["close"].ewm(span=26).mean()
            df["ema_crossover_flag"] = (df["ema_12"] > df["ema_26"])

            delta = df["close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df["rsi"] = 100 - (100 / (1 + rs))

            exp12 = df["close"].ewm(span=12).mean()
            exp26 = df["close"].ewm(span=26).mean()
            df["macd"] = exp12 - exp26
            df["macd_signal"] = df["macd"].ewm(span=9).mean()

            high_14 = df["high"].rolling(window=14).max()
            low_14 = df["low"].rolling(window=14).min()
            df["stochastic_k"] = ((df["close"] - low_14) / (high_14 - low_14)) * 100
            df["stochastic_d"] = df["stochastic_k"].rolling(window=3).mean()

            df["bollinger_middle"] = df["close"].rolling(window=20).mean()
            df["bollinger_std"] = df["close"].rolling(window=20).std()
            df["bollinger_upper"] = df["bollinger_middle"] + (df["bollinger_std"] * 2)
            df["bollinger_lower"] = df["bollinger_middle"] - (df["bollinger_std"] * 2)

            df["momentum_10"] = df["close"].diff(periods=10)
            df["momentum_50"] = df["close"].diff(periods=50)
            df["roc"] = df["close"].pct_change(periods=10)

            df["rolling_volatility_5h"] = df["close"].rolling(window=60).std()
            df["rolling_volatility_24h"] = df["close"].rolling(window=288).std()

            df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

            df["high_low_ratio"] = df["high"] / df["low"]
            df["price_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
            df["candle_body_size"] = abs(df["close"] - df["open"])
            df["candle_body_pct"] = df["candle_body_size"] / (df["high"] - df["low"])
            df["wick_upper"] = df["high"] - df[["open", "close"]].max(axis=1)
            df["wick_lower"] = df[["open", "close"]].min(axis=1) - df["low"]

            df["volume_price_ratio"] = df["volume"] / df["close"]
            df["volume_change_5m"] = df["volume"].pct_change()
            df["volume_surge"] = (df["volume"] > df["volume"].rolling(window=20).mean()).astype(int)

            recent_high = df["high"].rolling(window=50).max()
            recent_low = df["low"].rolling(window=50).min()
            diff = recent_high - recent_low
            df["fib_distance_0_236"] = (df["close"] - (recent_high - 0.236 * diff)) / diff
            df["fib_distance_0_382"] = (df["close"] - (recent_high - 0.382 * diff)) / diff
            df["fib_distance_0_618"] = (df["close"] - (recent_high - 0.618 * diff)) / diff

            df["atr_1h"] = (df["high"] - df["low"]).rolling(window=12).mean()
            df["atr_normalized"] = df["atr_1h"] / df["close"]
            df["stddev_1h"] = df["close"].rolling(window=12).std()

            df["overbought_rsi"] = (df["rsi"] >= 70)
            df["oversold_rsi"] = (df["rsi"] <= 30)
            df["upper_bollinger_break"] = (df["close"] > df["bollinger_upper"])
            df["lower_bollinger_break"] = (df["close"] < df["bollinger_lower"])

            df["slope_5h"] = df["close"].rolling(window=60).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
            df["slope_24h"] = df["close"].rolling(window=288).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
            df["trend_acceleration"] = df["slope_5h"] - df["slope_24h"]

            df["short_vs_long_strength"] = df["ema_12"] / df["ema_26"]

            # ADX Calculation (matching your training)
            high_diff = df["high"].diff()
            low_diff = df["low"].diff()

            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

            tr1 = df["high"] - df["low"]
            tr2 = abs(df["high"] - df["close"].shift())
            tr3 = abs(df["low"] - df["close"].shift())
            tr = np.maximum.reduce([tr1, tr2, tr3])

            atr = pd.Series(tr).rolling(window=14).sum()
            plus_di = 100 * pd.Series(plus_dm).rolling(window=14).sum() / atr
            minus_di = 100 * pd.Series(minus_dm).rolling(window=14).sum() / atr
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            adx = dx.rolling(window=14).mean()

            df["adx"] = adx.fillna(0)
            df.fillna(0, inplace=True)

            # Save latest row:
            row = df.iloc[-1]

            LiveModelMetrics.objects.create(
                coin=coin,
                timestamp=row["timestamp"],
                open=Decimal(row["open"]),
                high=Decimal(row["high"]),
                low=Decimal(row["low"]),
                close=Decimal(row["close"]),
                volume=Decimal(row["volume"]),
                sma_5=Decimal(row["sma_5"]),
                sma_20=Decimal(row["sma_20"]),
                ema_12=Decimal(row["ema_12"]),
                ema_26=Decimal(row["ema_26"]),
                ema_crossover_flag=bool(row["ema_crossover_flag"]),
                rsi=float(row["rsi"]),
                macd=float(row["macd"]),
                macd_signal=float(row["macd_signal"]),
                stochastic_k=float(row["stochastic_k"]),
                stochastic_d=float(row["stochastic_d"]),
                bollinger_upper=float(row["bollinger_upper"]),
                bollinger_middle=float(row["bollinger_middle"]),
                bollinger_lower=float(row["bollinger_lower"]),
                adx=float(row["adx"]),
                atr_1h=Decimal(row["atr_1h"]),
                stddev_1h=float(row["stddev_1h"]),
                momentum_10=float(row["momentum_10"]),
                momentum_50=float(row["momentum_50"]),
                roc=float(row["roc"]),
                rolling_volatility_5h=float(row["rolling_volatility_5h"]),
                rolling_volatility_24h=float(row["rolling_volatility_24h"]),
                high_low_ratio=float(row["high_low_ratio"]),
                price_position=float(row["price_position"]),
                candle_body_size=float(row["candle_body_size"]),
                candle_body_pct=float(row["candle_body_pct"]),
                wick_upper=float(row["wick_upper"]),
                wick_lower=float(row["wick_lower"]),
                slope_5h=float(row["slope_5h"]),
                slope_24h=float(row["slope_24h"]),
                trend_acceleration=float(row["trend_acceleration"]),
                fib_distance_0_236=float(row["fib_distance_0_236"]),
                fib_distance_0_382=float(row["fib_distance_0_382"]),
                fib_distance_0_618=float(row["fib_distance_0_618"]),
                vwap=float(row["vwap"]),
                volume_price_ratio=float(row["volume_price_ratio"]),
                volume_change_5m=float(row["volume_change_5m"]),
                volume_surge=bool(row["volume_surge"]),
                overbought_rsi=bool(row["overbought_rsi"]),
                oversold_rsi=bool(row["oversold_rsi"]),
                upper_bollinger_break=bool(row["upper_bollinger_break"]),
                lower_bollinger_break=bool(row["lower_bollinger_break"]),
                atr_normalized=float(row["atr_normalized"]),
                short_vs_long_strength=float(row["short_vs_long_strength"])
            )

            # Build instance for prediction
            instance = row.to_dict()
            instance.pop("timestamp")
            instance.pop("bollinger_std")
            for k in instance:
                if isinstance(instance[k], np.generic):
                    instance[k] = float(instance[k])

            jwt_token = get_google_jwt_token()
            vertex_url = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict"
            headers = {"Authorization": f"Bearer {jwt_token}", "Content-Type": "application/json"}
            payload = {"instances": [instance]}
            response = requests.post(vertex_url, headers=headers, json=payload)
            prediction = response.json()
            print(f"✅ {symbol}: Model Prediction: {prediction}")

        except Exception as e:
            print(f"❌ {symbol}: {e}")

    print("🚀 Live pipeline run complete.")
