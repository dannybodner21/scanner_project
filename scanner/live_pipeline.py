import os
import json
import requests
import numpy as np
import pandas as pd
import ta
import xgboost as xgb
import time
import base64
import hmac
import hashlib
from datetime import datetime
from scanner.models import Coin, ModelTrade

COINAPI_SYMBOL_MAP = {
    "BTCUSDT": "BINANCE_SPOT_BTC_USDT",
    "ETHUSDT": "BINANCE_SPOT_ETH_USDT",
    "XRPUSDT": "BINANCE_SPOT_XRP_USDT",
    "LTCUSDT": "BINANCE_SPOT_LTC_USDT",
    "SOLUSDT": "BINANCE_SPOT_SOL_USDT",
    "DOGEUSDT": "BINANCE_SPOT_DOGE_USDT",
    "LINKUSDT": "BINANCE_SPOT_LINK_USDT",
    "DOTUSDT": "BINANCE_SPOT_DOT_USDT",
    "SHIBUSDT": "BINANCE_SPOT_SHIB_USDT",
    "ADAUSDT": "BINANCE_SPOT_ADA_USDT",
}

COIN_SYMBOL_MAP_DB = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "XRPUSDT": "XRP",
    "LTCUSDT": "LTC",
    "SOLUSDT": "SOL",
    "DOGEUSDT": "DOGE",
    "LINKUSDT": "LINK",
    "DOTUSDT": "DOT",
    "SHIBUSDT": "SHIB",
    "ADAUSDT": "ADA",
}

FEATURES = [
    'returns_5m', 'returns_15m', 'returns_1h', 'returns_4h', 'momentum',
    'volume_ma_20', 'vol_spike', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_lower', 'atr_14', 'adx_14', 'obv', 'obv_slope',
    'ema_9', 'ema_21', 'ema_diff', 'volatility', 'ma_200',
    'bull_regime', 'bear_regime', 'sideways_regime'
]

CONFIDENCE_THRESHOLDS = [0.9, 0.8, 0.7, 0.6]


def fetch_ohlcv(coin, limit=300):
    coinapi_symbol = COINAPI_SYMBOL_MAP.get(coin)
    if not coinapi_symbol:
        raise ValueError(f"CoinAPI symbol mapping not found for {coin}")

    url = f"{BASE_URL}/{coinapi_symbol}/latest?period_id=5MIN&limit={limit}"
    headers = {"X-CoinAPI-Key": COINAPI_KEY}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame([{
        'timestamp': pd.to_datetime(candle['time_period_start']),
        'open': candle['price_open'],
        'high': candle['price_high'],
        'low': candle['price_low'],
        'close': candle['price_close'],
        'volume': candle['volume_traded']
    } for candle in data])

    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def add_features(df):
    df['returns_5m'] = df['close'].pct_change(1)
    df['returns_15m'] = df['close'].pct_change(3)
    df['returns_1h'] = df['close'].pct_change(12)
    df['returns_4h'] = df['close'].pct_change(48)
    df['momentum'] = df['close'] - df['close'].shift(5)

    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['vol_spike'] = df['volume'] / df['volume_ma_20']

    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    bollinger = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_lower'] = bollinger.bollinger_lband()

    df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['adx_14'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)

    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['obv_slope'] = df['obv'].diff()

    df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
    df['ema_diff'] = df['ema_9'] - df['ema_21']

    df['volatility'] = df['close'].rolling(20).std()
    df['ma_200'] = ta.trend.sma_indicator(df['close'], window=200)

    df['bull_regime'] = ((df['adx_14'] > 25) & (df['close'] > df['ma_200'])).astype(int)
    df['bear_regime'] = ((df['adx_14'] > 25) & (df['close'] < df['ma_200'])).astype(int)
    df['sideways_regime'] = (df['adx_14'] <= 25).astype(int)

    df = df.dropna()

    if df.empty:
        print("⚠ Warning: DataFrame empty after dropna in add_features")
    return df

def prepare_instance(df):
    row = df.iloc[-1]
    instance = {
        'open': row['open'],
        'high': row['high'],
        'low': row['low'],
        'close': row['close'],
        'volume': row['volume'],
    }
    for feature in FEATURES:
        instance[feature] = row[feature]
    return instance, row

def print_feature_stats(df, coin):
    print(f"--- Feature stats for {coin} ---")
    for feature in FEATURES:
        if feature in df.columns:
            series = df[feature]
            print(f"{feature} stats:")
            print(f"  count: {series.count()}")
            print(f"  mean: {series.mean()}")
            print(f"  std: {series.std()}")
            print(f"  min: {series.min()}")
            print(f"  25%: {series.quantile(0.25)}")
            print(f"  50%: {series.median()}")
            print(f"  75%: {series.quantile(0.75)}")
            print(f"  max: {series.max()}")
        else:
            print(f"⚠ Feature {feature} missing in dataframe for {coin}")

def run_live_pipeline(request=None):
    import django
    django.setup()

    model = xgb.Booster()
    model.load_model("best_xgb_model.bin")

    for coin in COINS:
        try:
            print(f"Processing {coin}...")
            df = fetch_ohlcv(coin, limit=300)

            if df.empty or len(df) < 220:
                print(f"⚠ Skipping {coin} due to insufficient raw data length")
                continue

            df = add_features(df)

            # Verify all features exist in df columns and are not NaN in last row
            missing_features = [f for f in FEATURES if f not in df.columns]
            if missing_features:
                print(f"⚠ Missing features {missing_features} for {coin}, skipping.")
                continue

            last_row = df.iloc[-1]
            nan_features = [f for f in FEATURES if pd.isna(last_row[f])]
            if nan_features:
                print(f"⚠ Features with NaN in last row: {nan_features} for {coin}, skipping.")
                continue

            instance, row = prepare_instance(df)
            feature_df = pd.DataFrame([instance])

            dmatrix = xgb.DMatrix(feature_df)
            proba = model.predict(dmatrix)[0]

            print(f"{coin}: Confidence = {proba:.4f}")

            db_symbol = COIN_SYMBOL_MAP_DB.get(coin)
            coin_obj = Coin.objects.get(symbol=db_symbol)

            for threshold in CONFIDENCE_THRESHOLDS:
                if proba >= threshold:
                    open_trade_exists = ModelTrade.objects.filter(
                        coin=coin_obj,
                        confidence_trade=threshold,
                        exit_timestamp__isnull=True
                    ).exists()

                    if not open_trade_exists:
                        ModelTrade.objects.create(
                            coin=coin_obj,
                            trade_type="long",
                            entry_timestamp=row["timestamp"],
                            duration_minutes=0,
                            entry_price=row["close"],
                            model_confidence=proba,
                            take_profit_percent=6,
                            stop_loss_percent=3,
                            confidence_trade=threshold
                        )
                        print(f"💰 Trade placed for {coin} at confidence threshold {threshold}")
                    else:
                        print(f"⚠ Open trade exists for {coin} at confidence threshold {threshold}")
                    break

        except Exception as e:
            print(f"❌ Error processing {coin}: {e}")

if __name__ == "__main__":
    run_live_pipeline()
