import os
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# CONFIG
PROJECT_ID = "healthy-mark-446922-p8"
LOCATION = "us-central1"
ENDPOINT_ID = "1878894947566878720"
REGION = LOCATION

COINAPI_KEY = "01293e2a-dcf1-4e81-8310-c6aa9d0cb743"
BASE_URL = "https://rest.coinapi.io/v1/ohlcv"

SYMBOL_MAP = {
    "BTC": "BINANCE_SPOT_BTC_USDT", "ETH": "BINANCE_SPOT_ETH_USDT",
    "XRP": "BINANCE_SPOT_XRP_USDT", "LTC": "BINANCE_SPOT_LTC_USDT",
    "SOL": "BINANCE_SPOT_SOL_USDT", "DOGE": "BINANCE_SPOT_DOGE_USDT",
    "PEPE": "BINANCE_SPOT_PEPE_USDT", "ADA": "BINANCE_SPOT_ADA_USDT",
    "XLM": "BINANCE_SPOT_XLM_USDT", "SUI": "BINANCE_SPOT_SUI_USDT",
    "LINK": "BINANCE_SPOT_LINK_USDT", "AVAX": "BINANCE_SPOT_AVAX_USDT",
    "DOT": "BINANCE_SPOT_DOT_USDT", "SHIB": "BINANCE_SPOT_SHIB_USDT",
    "HBAR": "BINANCE_SPOT_HBAR_USDT", "UNI": "BINANCE_SPOT_UNI_USDT"
}

# These fields require string conversion
TEXT_FEATURES = [
    "ema_crossover_flag", "overbought_rsi", "oversold_rsi",
    "upper_bollinger_break", "lower_bollinger_break"
]

# Your features (columns in model)
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

def get_google_jwt_token():
    service_account_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())
    return credentials.token

# main entry point
def run_live_pipeline(request=None):
    print(f"⏱ Running live predictions: {datetime.now()}")

    try:
        jwt_token = get_google_jwt_token()
        vertex_url = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict"
        headers = {"Authorization": f"Bearer {jwt_token}", "Content-Type": "application/json"}

        for symbol, coinapi_symbol in SYMBOL_MAP.items():
            try:
                # Pull 100 latest 5m candles
                url = f"{BASE_URL}/{coinapi_symbol}/latest?period_id=5MIN&limit=100"
                resp = requests.get(url, headers={"X-CoinAPI-Key": COINAPI_KEY}, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                if not data or len(data) < 50:
                    print(f"❌ {symbol}: Not enough candles")
                    continue

                # Build dataframe
                df = pd.DataFrame([{
                    "timestamp": pd.to_datetime(candle['time_period_start']),
                    "open": candle['price_open'],
                    "high": candle['price_high'],
                    "low": candle['price_low'],
                    "close": candle['price_close'],
                    "volume": candle['volume_traded']
                } for candle in data])

                df.sort_values("timestamp", inplace=True)
                df.reset_index(drop=True, inplace=True)

                # Full feature calculations (same as your model)
                df["sma_5"] = df["close"].rolling(5).mean()
                df["sma_20"] = df["close"].rolling(20).mean()
                df["ema_12"] = df["close"].ewm(span=12).mean()
                df["ema_26"] = df["close"].ewm(span=26).mean()
                df["ema_crossover_flag"] = (df["ema_12"] > df["ema_26"]).astype(str)

                delta = df["close"].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(14).mean()
                avg_loss = loss.rolling(14).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                df["rsi"] = 100 - (100 / (1 + rs))

                df["macd"] = df["ema_12"] - df["ema_26"]
                df["macd_signal"] = df["macd"].ewm(span=9).mean()

                high_14 = df["high"].rolling(14).max()
                low_14 = df["low"].rolling(14).min()
                df["stochastic_k"] = ((df["close"] - low_14) / (high_14 - low_14)) * 100
                df["stochastic_d"] = df["stochastic_k"].rolling(3).mean()

                df["bollinger_middle"] = df["close"].rolling(20).mean()
                df["bollinger_std"] = df["close"].rolling(20).std()
                df["bollinger_upper"] = df["bollinger_middle"] + 2 * df["bollinger_std"]
                df["bollinger_lower"] = df["bollinger_middle"] - 2 * df["bollinger_std"]

                df["momentum_10"] = df["close"].diff(10)
                df["momentum_50"] = df["close"].diff(50)
                df["roc"] = df["close"].pct_change(10)

                df["rolling_volatility_5h"] = df["close"].rolling(60).std()
                df["rolling_volatility_24h"] = df["close"].rolling(288).std()

                df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
                df["high_low_ratio"] = df["high"] / df["low"]
                df["price_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
                df["candle_body_size"] = abs(df["close"] - df["open"])
                df["candle_body_pct"] = df["candle_body_size"] / (df["high"] - df["low"])
                df["wick_upper"] = df["high"] - df[["open", "close"]].max(axis=1)
                df["wick_lower"] = df[["open", "close"]].min(axis=1) - df["low"]

                df["volume_price_ratio"] = df["volume"] / df["close"]
                df["volume_change_5m"] = df["volume"].pct_change()
                df["volume_surge"] = (df["volume"] > df["volume"].rolling(20).mean()).astype(int)

                recent_high = df["high"].rolling(50).max()
                recent_low = df["low"].rolling(50).min()
                diff = recent_high - recent_low
                df["fib_distance_0_236"] = (df["close"] - (recent_high - 0.236 * diff)) / diff
                df["fib_distance_0_382"] = (df["close"] - (recent_high - 0.382 * diff)) / diff
                df["fib_distance_0_618"] = (df["close"] - (recent_high - 0.618 * diff)) / diff

                df["atr_1h"] = (df["high"] - df["low"]).rolling(12).mean()
                df["atr_normalized"] = df["atr_1h"] / df["close"]
                df["stddev_1h"] = df["close"].rolling(12).std()

                df["overbought_rsi"] = (df["rsi"] >= 70).astype(str)
                df["oversold_rsi"] = (df["rsi"] <= 30).astype(str)
                df["upper_bollinger_break"] = (df["close"] > df["bollinger_upper"]).astype(str)
                df["lower_bollinger_break"] = (df["close"] < df["bollinger_lower"]).astype(str)

                df["slope_5h"] = df["close"].rolling(60).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
                df["slope_24h"] = df["close"].rolling(288).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
                df["trend_acceleration"] = df["slope_5h"] - df["slope_24h"]
                df["short_vs_long_strength"] = df["ema_12"] / df["ema_26"]

                # ADX calculation:
                high_diff = df["high"].diff()
                low_diff = df["low"].diff()
                plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
                minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
                tr1 = df["high"] - df["low"]
                tr2 = abs(df["high"] - df["close"].shift())
                tr3 = abs(df["low"] - df["close"].shift())
                tr = np.maximum.reduce([tr1, tr2, tr3])
                atr = pd.Series(tr).rolling(14).sum()
                plus_di = 100 * pd.Series(plus_dm).rolling(14).sum() / atr
                minus_di = 100 * pd.Series(minus_dm).rolling(14).sum() / atr
                dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
                df["adx"] = dx.rolling(14).mean().fillna(0)

                df.fillna(0, inplace=True)

                row = df.iloc[-1]
                instance = {}

                for feature in FEATURES:
                    val = row[feature]
                    instance[feature] = str(val) if feature in TEXT_FEATURES else float(val)

                payload = {"instances": [instance]}
                response = requests.post(vertex_url, headers=headers, json=payload)
                response.raise_for_status()

                predictions = response.json().get("predictions", [])
                if predictions:
                    pred = predictions[0]
                    confidence = pred["scores"][0]
                    print(f"✅ {symbol} — Confidence: {confidence:.4f}")
                else:
                    print(f"⚠ {symbol}: No predictions returned")

            except Exception as e:
                print(f"❌ {symbol}: {e}")

    except Exception as e:
        print(f"❌ Fatal Error: {e}")

if __name__ == "__main__":
    run_live_pipeline()
