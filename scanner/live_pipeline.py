import os
import json
import requests
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime
from django.utils.timezone import make_aware
from scanner.models import Coin, LiveModelMetrics, ModelTrade
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

            # All metrics (unchanged - leaving your calculations fully intact)
            # ... your entire metrics calculation remains the same here ...

            # (to save space here, I’m not repeating your metric calculations, ADX, etc)
            # KEEP ALL OF YOUR PREVIOUS CALCULATION CODE UNCHANGED UP TO THIS POINT

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

            instance = row.to_dict()
            instance.pop("timestamp")
            instance.pop("bollinger_std")

            # HERE IS THE KEY FIX:
            for k in instance:
                v = instance[k]
                if isinstance(v, (np.generic, Decimal)):
                    instance[k] = float(v)

            # Convert booleans to strings exactly as Vertex expects
            instance["ema_crossover_flag"] = "true" if instance["ema_crossover_flag"] else "false"
            instance["volume_surge"] = "true" if instance["volume_surge"] else "false"
            instance["overbought_rsi"] = "true" if instance["overbought_rsi"] else "false"
            instance["oversold_rsi"] = "true" if instance["oversold_rsi"] else "false"
            instance["upper_bollinger_break"] = "true" if instance["upper_bollinger_break"] else "false"
            instance["lower_bollinger_break"] = "true" if instance["lower_bollinger_break"] else "false"

            # As final safety: replace any accidental NaNs with 0.0
            for k in instance:
                if isinstance(instance[k], float) and np.isnan(instance[k]):
                    instance[k] = 0.0

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
                print(f"LONG | {symbol} — Confidence: {confidence:.4f}")

                metric_obj = LiveModelMetrics.objects.get(coin=coin, timestamp=row["timestamp"])

                for threshold in [0.9, 0.8, 0.7, 0.6]:
                    if confidence >= threshold:
                        if not ModelTrade.objects.filter(
                            coin=coin,
                            confidence_trade=threshold,
                            exit_timestamp__isnull=True
                        ).exists():
                            ModelTrade.objects.create(
                                coin=coin,
                                trade_type="long",
                                entry_timestamp=metric_obj.timestamp,
                                duration_minutes=0,
                                entry_price=metric_obj.close,
                                model_confidence=confidence,
                                take_profit_percent=3,
                                stop_loss_percent=2,
                                confidence_trade=threshold,
                            )
                        break

        except Exception as e:
            print(f"❌ {symbol}: {e}")

    print("🚀 Live pipeline run complete.")
