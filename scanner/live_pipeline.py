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
import urllib.parse

from datetime import datetime, timedelta
from scanner.models import Coin, ModelTrade, RealTrade
from scipy.stats import linregress
from django.utils.timezone import now


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
    "UNIUSDT": "BINANCE_SPOT_UNI_USDT",
    "AVAXUSDT": "BINANCE_SPOT_AVAX_USDT",
    "XLMUSDT": "BINANCE_SPOT_XLM_USDT",
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
    "UNIUSDT": "UNI",
    "AVAXUSDT": "AVAX",
    "XLMUSDT": "XLM",
}


COINS = list(COIN_SYMBOL_MAP_DB.keys())
COINAPI_KEY = "01293e2a-dcf1-4e81-8310-c6aa9d0cb743"
BASE_URL = "https://rest.coinapi.io/v1/ohlcv"




FEATURES = [
    'open', 'high', 'low', 'close', 'volume',
    'adx_14', 'ma_200', 'returns_5m', 'returns_15m', 'returns_1h', 'returns_4h',
    'momentum', 'volume_ma_20', 'vol_spike', 'rsi_14', 'macd', 'macd_signal',
    'macd_hist', 'bb_upper', 'bb_lower', 'atr_14', 'obv', 'obv_slope',
    'ema_9', 'ema_21', 'ema_diff', 'volatility', 'bull_regime', 'bear_regime',
    'sideways_regime', 'slope_1h', 'dist_from_high_24h', 'dist_from_low_24h',
    'stoch_k', 'stoch_d', 'price_change_5', 'volume_change_5',
    'high_1h', 'low_1h', 'pos_in_range_1h', 'vwap_1h', 'pos_vs_vwap'
]


INPUT_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'adx_14', 'ma_200', 'returns_5m', 'returns_15m', 'returns_1h', 'returns_4h',
    'momentum', 'volume_ma_20', 'vol_spike', 'rsi_14', 'macd', 'macd_signal',
    'macd_hist', 'bb_upper', 'bb_lower', 'atr_14', 'obv', 'obv_slope',
    'ema_9', 'ema_21', 'ema_diff', 'volatility', 'bull_regime', 'bear_regime',
    'sideways_regime', 'slope_1h', 'dist_from_high_24h', 'dist_from_low_24h',
    'stoch_k', 'stoch_d', 'price_change_5', 'volume_change_5',
    'high_1h', 'low_1h', 'pos_in_range_1h', 'vwap_1h', 'pos_vs_vwap'
]


CONFIDENCE_THRESHOLD = 0.99


def send_text(messages):

    if len(messages) > 0:

        chat_id_danny = '1077594551'
        chat_ids = [chat_id_danny]
        bot_token = '7672687080:AAFWvkwzp-LQE92XdO9vcVa5yWJDUxO17yE'
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

        message = ""
        for chat_id in chat_ids:
            for trigger in messages:

                message += trigger + " "

            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown",
            }

            response = requests.post(url, data=payload)

            if response.status_code == 200:
                print("Message sent successfully.")
            else:
                print(f"Failed to send message: {response.content}")

    return


def calculate_trend_slope(prices):
    if len(prices) < 12:
        return np.nan
    x = np.arange(len(prices))
    slope, _, _, _, _ = linregress(x, prices)
    return slope


def fetch_ohlcv_old(coin, limit=300):
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


def fetch_ohlcv(coin, limit=300, end_time=None):
    coinapi_symbol = COINAPI_SYMBOL_MAP.get(coin)
    if not coinapi_symbol:
        raise ValueError(f"CoinAPI symbol mapping not found for {coin}")

    # Default to now if no end_time is provided
    if end_time is None:
        end_time = datetime.utcnow()

    time_end = end_time.isoformat()

    url = f"{BASE_URL}/{coinapi_symbol}/history?period_id=5MIN&time_end={time_end}&limit={limit}"
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

    initial_len = len(df)



    try:

        df['returns_5m'] = df['close'].pct_change(1).clip(-1, 1)
        df['returns_15m'] = df['close'].pct_change(3).clip(-1, 1)
        df['returns_1h'] = df['close'].pct_change(12).clip(-1, 1)
        df['returns_4h'] = df['close'].pct_change(48).clip(-1, 1)
        df['momentum'] = df['close'] - df['close'].shift(5)

    except Exception as e:
        raise RuntimeError(f"❌ Failed to calculate returns or momentum: {e}")


    try:

        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['vol_spike'] = df['volume'] / df['volume_ma_20']

    except Exception as e:
        raise RuntimeError(f"❌ Failed to calculate volume spike: {e}")


    try:

        df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()

    except Exception as e:
        raise RuntimeError(f"❌ Failed to calculate RSI or MACD: {e}")

    try:

        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()

    except Exception as e:
        raise RuntimeError(f"❌ Failed to calculate Bollinger Bands: {e}")

    try:

        df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

    except Exception as e:
        raise RuntimeError(f"❌ Failed to calculate ATR: {e}")

    try:

        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['obv_slope'] = df['obv'].diff()

    except Exception as e:
        raise RuntimeError(f"❌ Failed to calculate OBV: {e}")

    try:

        df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['ema_diff'] = df['ema_9'] - df['ema_21']

    except Exception as e:
        raise RuntimeError(f"❌ Failed to calculate EMA: {e}")

    try:

        df['volatility'] = df['close'].rolling(20).std()

    except Exception as e:
        raise RuntimeError(f"❌ Failed to calculate volatility: {e}")


    try:

        df['adx_14'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df['ma_200'] = ta.trend.sma_indicator(df['close'], window=200)

        print(f"{df['adx_14'].notna().sum()} rows with adx_14")
        print(f"{df['ma_200'].notna().sum()} rows with ma_200")

        # df = df[df['adx_14'].notna() & df['ma_200'].notna()].copy()

        # print(f"Remaining rows after adx_14 and ma_200 filter: {len(df)}")

    except Exception as e:
        raise RuntimeError(f"❌ Failed to calculate adx_14 or ma_200: {e}")


    try:

        adx_thresh = df['adx_14'].quantile(0.5)
        df['bull_regime'] = ((df['adx_14'] > adx_thresh) & (df['close'] > df['ma_200'])).astype(int)
        df['bear_regime'] = ((df['adx_14'] > adx_thresh) & (df['close'] < df['ma_200'])).astype(int)
        df['sideways_regime'] = (df['adx_14'] <= adx_thresh).astype(int)

    except Exception as e:
        raise RuntimeError(f"❌ Failed to calculate regimes: {e}")

    try:

        df['slope_1h'] = df['close'].rolling(12).apply(calculate_trend_slope, raw=False)

        df['dist_from_high_24h'] = ((df['close'] - df['high'].rolling(288).max()) / df['high'].rolling(288).max()).clip(-1, 1)
        df['dist_from_low_24h'] = ((df['close'] - df['low'].rolling(288).min()) / df['low'].rolling(288).min()).clip(-1, 5)

    except Exception as e:
        raise RuntimeError(f"❌ Failed to calculate distance from 24h high/low: {e}")


    try:

        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

    except Exception as e:
        raise RuntimeError(f"❌ Failed to calculate Stochastic Oscillator: {e}")


    try:

        df['price_change_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        df['volume_change_5'] = (df['volume'] - df['volume'].shift(5)) / df['volume'].shift(5)

    except Exception as e:
        raise RuntimeError(f"❌ Failed to calculate price/volume change: {e}")

    try:

        df['high_1h'] = df['high'].rolling(12).max()
        df['low_1h'] = df['low'].rolling(12).min()
        df['pos_in_range_1h'] = (df['close'] - df['low_1h']) / (df['high_1h'] - df['low_1h'])
        df['vwap_1h'] = (df['close'] * df['volume']).rolling(12).sum() / df['volume'].rolling(12).sum()
        df['pos_vs_vwap'] = df['close'] - df['vwap_1h']

    except Exception as e:
        raise RuntimeError(f"❌ Failed to calculate high/low/vwap metrics: {e}")


    print(f"Rows before final check: {len(df)}")

    # ✅ Only check the last row for missing values
    for col in INPUT_COLUMNS:
        if pd.isna(df.iloc[-1][col]):
            raise RuntimeError(f"❌ Missing value in feature '{col}' for last row.")

    print(f"✅ Last row is valid for prediction: {df.iloc[-1]['timestamp']}")


    return df


def prepare_instance(df):
    row = df.iloc[-1]
    instance = {col: row[col] for col in INPUT_COLUMNS}
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

    long_model = xgb.Booster()
    long_model.load_model("eight_long_xgb_model.bin")

    short_model = xgb.Booster()
    short_model.load_model("seven_short_xgb_model.bin")

    for coin in COINS:
        try:
            print(f"Processing {coin}...")

            # df = fetch_ohlcv(coin, limit=350)

            aligned_now = datetime.utcnow().replace(second=0, microsecond=0)
            df = fetch_ohlcv(coin, limit=350, end_time=aligned_now)


            print("\n📈 LIVE CANDLE:")
            print(df.tail(5)[['timestamp', 'open', 'high', 'low', 'close', 'volume']])


            print(df.tail(10))
            print(df.describe())
            print(df.isna().sum())
            print(f"Fetched {len(df)} rows for {coin}")

            if df.empty or len(df) < 288:
                print(f"⚠ Skipping {coin} due to insufficient raw data length")
                continue

            df = add_features(df)

            if df.empty or len(df) < 1:
                print(f"⚠ Skipping {coin} after feature engineering — no valid rows")
                continue

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
            feature_df = feature_df[INPUT_COLUMNS]


            # 🔍 Print input features for investigation
            #print(f"\n📊 Features sent to model for {coin}:")
            #for col in INPUT_COLUMNS:
                #print(f"  {col}: {feature_df.iloc[0][col]}")


            dmatrix = xgb.DMatrix(feature_df[INPUT_COLUMNS], feature_names=INPUT_COLUMNS)

            #proba = model.predict(dmatrix)[0]
            long_proba = long_model.predict(dmatrix)[0]
            short_proba = short_model.predict(dmatrix)[0]

            print(f"{coin}: Long = {long_proba:.4f}")
            print(f"{coin}: Short = {short_proba:.4f}")






            log_entry = {
                "timestamp": row["timestamp"].isoformat(),
                "coin": coin,
                "predicted_long_prob": float(long_proba),
                "predicted_short_prob": float(short_proba),
                "threshold": CONFIDENCE_THRESHOLD,
                "decision_long": "LONG" if long_proba >= CONFIDENCE_THRESHOLD else "NO TRADE",
                "decision_short": "SHORT" if short_proba >= CONFIDENCE_THRESHOLD else "NO TRADE",
                "features": {k: float(row[k]) for k in INPUT_COLUMNS}
            }

            with open("live_predictions_log.jsonl", "a") as f:
                f.write(json.dumps(log_entry) + "\n")




            db_symbol = COIN_SYMBOL_MAP_DB.get(coin)
            coin_obj = Coin.objects.get(symbol=db_symbol)

            existing_long_trade = ModelTrade.objects.filter(
                coin=coin_obj,
                exit_timestamp__isnull=True,
                trade_type="long"
            ).first()

            existing_short_trade = ModelTrade.objects.filter(
                coin=coin_obj,
                exit_timestamp__isnull=True,
                trade_type="short"
            ).first()

            if not existing_long_trade:
                if long_proba >= CONFIDENCE_THRESHOLD:
                    ModelTrade.objects.create(
                        coin=coin_obj,
                        trade_type="long",
                        entry_timestamp=row["timestamp"],
                        duration_minutes=0,
                        entry_price=row["close"],
                        model_confidence=long_proba,
                        take_profit_percent=4,
                        stop_loss_percent=2,
                        confidence_trade=CONFIDENCE_THRESHOLD
                    )
                    print(f"💰 LONG trade placed for {coin} @ {CONFIDENCE_THRESHOLD}")

            else:
                print(f"⚠ Long trade already open for {coin}: {existing_long_trade.trade_type} @ {existing_long_trade.entry_timestamp}")

            if not existing_short_trade:
                if short_proba >= CONFIDENCE_THRESHOLD:
                    ModelTrade.objects.create(
                        coin=coin_obj,
                        trade_type="short",
                        entry_timestamp=row["timestamp"],
                        duration_minutes=0,
                        entry_price=row["close"],
                        model_confidence=short_proba,
                        take_profit_percent=4,
                        stop_loss_percent=2,
                        confidence_trade=CONFIDENCE_THRESHOLD
                    )
                    print(f"🔻 SHORT trade placed for {coin} @ {CONFIDENCE_THRESHOLD}")

            else:
                print(f"⚠ Short trade already open for {coin}: {existing_short_trade.trade_type} @ {existing_short_trade.entry_timestamp}")


        except Exception as e:
            print(f"❌ Error processing {coin}: {e}")

    #send_real_trade_updates()

if __name__ == "__main__":
    run_live_pipeline()
