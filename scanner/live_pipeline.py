import os
import requests
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime, timedelta, timezone

from django.utils.timezone import make_aware
from scanner.models import Coin, LiveModelMetrics

# ----------------------------------------
# CONFIG
# ----------------------------------------

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
# Main pipeline logic
# ----------------------------------------

def run_live_pipeline():
    print(f"⏱ Live pipeline started: {datetime.now()}")

    for symbol, coinapi_symbol in SYMBOL_MAP.items():
        try:
            coin = Coin.objects.get(symbol=symbol)

            now_utc = datetime.utcnow().replace(second=0, microsecond=0, tzinfo=timezone.utc)
            start_time = (now_utc - timedelta(minutes=5)).isoformat()
            end_time = now_utc.isoformat()

            url = f"{BASE_URL}/{coinapi_symbol}/history?period_id=5MIN&time_start={start_time}&time_end={end_time}&limit=1"
            headers = {"X-CoinAPI-Key": COINAPI_KEY}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()

            data = resp.json()
            if not data:
                raise ValueError("No data returned")

            candle = data[0]
            ts = datetime.fromisoformat(candle['time_period_start'].replace("Z", "+00:00"))
            price = Decimal(candle['price_close'])
            high = Decimal(candle['price_high'])
            low = Decimal(candle['price_low'])
            open_price = Decimal(candle['price_open'])
            close_price = Decimal(candle['price_close'])
            volume = Decimal(candle['volume_traded'])

            df = pd.DataFrame([{
                "timestamp": ts,
                "open": float(open_price),
                "high": float(high),
                "low": float(low),
                "close": float(close_price),
                "volume": float(volume)
            }])

            # -----------------------------------
            # Calculations start here
            # -----------------------------------

            df["sma_5"] = df["close"].rolling(window=5).mean().fillna(df["close"])
            df["sma_20"] = df["close"].rolling(window=20).mean().fillna(df["close"])
            df["ema_12"] = df["close"].ewm(span=12).mean()
            df["ema_26"] = df["close"].ewm(span=26).mean()
            df["ema_crossover_flag"] = (df["ema_12"] > df["ema_26"]).astype(int)

            delta = df["close"].diff().fillna(0)
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=14).mean().fillna(0)
            avg_loss = loss.rolling(window=14).mean().fillna(0)
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df["rsi"] = 100 - (100 / (1 + rs.fillna(0)))

            exp12 = df["close"].ewm(span=12, adjust=False).mean()
            exp26 = df["close"].ewm(span=26, adjust=False).mean()
            df["macd"] = exp12 - exp26
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

            high_14 = df["high"].rolling(window=14).max().fillna(df["high"])
            low_14 = df["low"].rolling(window=14).min().fillna(df["low"])
            df["stochastic_k"] = ((df["close"] - low_14) / (high_14 - low_14)) * 100
            df["stochastic_d"] = df["stochastic_k"].rolling(window=3).mean()

            df["bollinger_middle"] = df["close"].rolling(window=20).mean()
            df["bollinger_std"] = df["close"].rolling(window=20).std()
            df["bollinger_upper"] = df["bollinger_middle"] + (df["bollinger_std"] * 2)
            df["bollinger_lower"] = df["bollinger_middle"] - (df["bollinger_std"] * 2)

            # Save last row to DB:
            row = df.iloc[-1]

            LiveModelMetrics.objects.create(
                coin=coin,
                timestamp=make_aware(ts),
                price=price,
                high_24h=high,
                low_24h=low,
                open=open_price,
                close=close_price,
                volume=volume,
                sma_5=Decimal(row["sma_5"]),
                sma_20=Decimal(row["sma_20"]),
                ema_12=Decimal(row["ema_12"]),
                ema_26=Decimal(row["ema_26"]),
                ema_crossover_flag=int(row["ema_crossover_flag"]),
                rsi=float(row["rsi"]),
                macd=float(row["macd"]),
                macd_signal=float(row["macd_signal"]),
                stochastic_k=float(row["stochastic_k"]),
                stochastic_d=float(row["stochastic_d"]),
                bollinger_upper=float(row["bollinger_upper"]),
                bollinger_middle=float(row["bollinger_middle"]),
                bollinger_lower=float(row["bollinger_lower"]),
            )

            print(f"✅ {symbol}: Calculations and save succeeded.")

        except Exception as e:
            print(f"❌ {symbol}: {e}")

    print("🚀 Live pipeline run complete.")
