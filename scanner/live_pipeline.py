import os
import json
import requests
import numpy as np
import pandas as pd
import ta
import xgboost as xgb
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

COINAPI_KEY = "01293e2a-dcf1-4e81-8310-c6aa9d0cb743"
BASE_URL = "https://rest.coinapi.io/v1/ohlcv"
COINS = list(COINAPI_SYMBOL_MAP.keys())

FEATURES = [
    'returns_5m', 'returns_15m', 'returns_1h', 'returns_4h', 'momentum',
    'volume_ma_20', 'vol_spike', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_lower', 'atr_14', 'adx_14', 'obv', 'obv_slope',
    'ema_9', 'ema_21', 'ema_diff', 'volatility', 'ma_200',
    'bull_regime', 'bear_regime', 'sideways_regime'
]

CONFIDENCE_THRESHOLDS = [0.9, 0.8, 0.7, 0.6]

def fetch_ohlcv(coin, limit=100):
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
    print(f"returns_5m stats:\n{df['returns_5m'].describe()}")

    df['returns_15m'] = df['close'].pct_change(3)
    print(f"returns_15m stats:\n{df['returns_15m'].describe()}")

    df['returns_1h'] = df['close'].pct_change(12)
    print(f"returns_1h stats:\n{df['returns_1h'].describe()}")

    df['returns_4h'] = df['close'].pct_change(48)
    print(f"returns_4h stats:\n{df['returns_4h'].describe()}")

    df['momentum'] = df['close'] - df['close'].shift(5)
    print(f"momentum stats:\n{df['momentum'].describe()}")

    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    print(f"volume_ma_20 stats:\n{df['volume_ma_20'].describe()}")

    df['vol_spike'] = df['volume'] / df['volume_ma_20']
    print(f"vol_spike stats:\n{df['vol_spike'].describe()}")

    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
    print(f"rsi_14 stats:\n{df['rsi_14'].describe()}")

    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    print(f"macd stats:\n{df['macd'].describe()}")
    print(f"macd_signal stats:\n{df['macd_signal'].describe()}")
    print(f"macd_hist stats:\n{df['macd_hist'].describe()}")

    bollinger = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_lower'] = bollinger.bollinger_lband()
    print(f"bb_upper stats:\n{df['bb_upper'].describe()}")
    print(f"bb_lower stats:\n{df['bb_lower'].describe()}")

    df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    print(f"atr_14 stats:\n{df['atr_14'].describe()}")

    df['adx_14'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    print(f"adx_14 stats:\n{df['adx_14'].describe()}")

    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['obv_slope'] = df['obv'].diff()
    print(f"obv stats:\n{df['obv'].describe()}")
    print(f"obv_slope stats:\n{df['obv_slope'].describe()}")

    df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
    df['ema_diff'] = df['ema_9'] - df['ema_21']
    print(f"ema_9 stats:\n{df['ema_9'].describe()}")
    print(f"ema_21 stats:\n{df['ema_21'].describe()}")
    print(f"ema_diff stats:\n{df['ema_diff'].describe()}")

    df['volatility'] = df['close'].rolling(20).std()
    print(f"volatility stats:\n{df['volatility'].describe()}")

    df['ma_200'] = ta.trend.sma_indicator(df['close'], window=200)
    print(f"ma_200 stats:\n{df['ma_200'].describe()}")

    df['bull_regime'] = ((df['adx_14'] > 25) & (df['close'] > df['ma_200'])).astype(int)
    df['bear_regime'] = ((df['adx_14'] > 25) & (df['close'] < df['ma_200'])).astype(int)
    df['sideways_regime'] = (df['adx_14'] <= 25).astype(int)
    print(f"bull_regime stats:\n{df['bull_regime'].describe()}")
    print(f"bear_regime stats:\n{df['bear_regime'].describe()}")
    print(f"sideways_regime stats:\n{df['sideways_regime'].describe()}")

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

def run_live_pipeline(request=None):
    import django
    django.setup()

    # Load XGBoost model properly
    model = xgb.Booster()
    model.load_model("best_xgb_model.bin")

    for coin in COINS:
        try:
            print(f"Processing {coin}...")
            df = fetch_ohlcv(coin, limit=300)

            if df.empty or len(df) < 220:
                print(f"⚠ Skipping {coin} due to insufficient data after feature engineering")
                continue

            df = add_features(df)

            # Check that all features exist in dataframe columns
            missing_features = [f for f in FEATURES if f not in df.columns]
            if missing_features:
                print(f"⚠ Missing features {missing_features} for {coin}, skipping.")
                continue

            # Check last row for NaNs in features
            row = df.iloc[-1]
            nan_features = [f for f in FEATURES if pd.isna(row[f])]
            if nan_features:
                print(f"⚠ Features with NaN values {nan_features} for {coin}, skipping.")
                continue

            instance, row = prepare_instance(df)
            feature_df = pd.DataFrame([instance])

            print(f"{coin} feature stats:")
            print(feature_df.describe().T[['mean','std','min','max']])

            # Convert to DMatrix for prediction
            dmatrix = xgb.DMatrix(feature_df)
            proba = model.predict(dmatrix)[0]  # probability of positive class

            print(f"{coin}: Confidence = {proba:.4f}")

            # Get Coin instance from DB
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
