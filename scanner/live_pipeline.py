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
import django
import joblib

from datetime import datetime, timedelta
from scanner.models import Coin, ModelTrade, RealTrade
from scipy.stats import linregress
from joblib import load

from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

from django.utils.timezone import now, make_aware

import joblib
from scanner.management.commands.one_dataset import add_enhanced_features  # import your actual function
from sklearn.preprocessing import StandardScaler


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



MODEL_PATH = "enhanced_model.joblib"
SCALER_PATH = "feature_scaler.joblib"
FEATURES_PATH = "selected_features.joblib"
CONFIDENCE_THRESHOLD = 0.70


FEATURES = [
    'ema_9', 'ema_21', 'ema_50', 'ema_200', 'rsi_14', 'macd', 'macd_signal',
    'macd_hist', 'bb_upper', 'bb_lower', 'bb_width', 'obv', 'atr_14',
    'adx_14', 'ema_dist_50_norm', 'candle_size_norm', 'macd_hist_norm',
    'rsi_x_vol_change', 'bb_width_x_adx', 'hour', 'day_of_week',
    'day_of_year', 'month', 'rsi_lag_1', 'close_change_lag_1',
    'rsi_lag_3', 'close_change_lag_3', 'rsi_lag_6', 'close_change_lag_6',
    'rsi_lag_12', 'close_change_lag_12', 'rolling_mean_close_12',
    'rolling_std_close_12', 'rolling_mean_volume_12',
    'rolling_mean_close_24', 'rolling_std_close_24',
    'rolling_mean_volume_24', 'rolling_mean_close_144',
    'rolling_std_close_144', 'rolling_mean_volume_144',
    'market_avg_change', 'market_avg_rsi', 'coin_ADAUSDT',
    'coin_AVAXUSDT', 'coin_BTCUSDT', 'coin_DOGEUSDT', 'coin_DOTUSDT',
    'coin_ETHUSDT', 'coin_LINKUSDT', 'coin_LTCUSDT', 'coin_SHIBUSDT',
    'coin_SOLUSDT', 'coin_UNIUSDT', 'coin_XLMUSDT', 'coin_XRPUSDT'
]


INPUT_COLUMNS = [
    'ema_9', 'ema_21', 'ema_50', 'ema_200', 'rsi_14', 'macd', 'macd_signal',
    'macd_hist', 'bb_upper', 'bb_lower', 'bb_width', 'obv', 'atr_14',
    'adx_14', 'ema_dist_50_norm', 'candle_size_norm', 'macd_hist_norm',
    'rsi_x_vol_change', 'bb_width_x_adx', 'hour', 'day_of_week',
    'day_of_year', 'month', 'rsi_lag_1', 'close_change_lag_1',
    'rsi_lag_3', 'close_change_lag_3', 'rsi_lag_6', 'close_change_lag_6',
    'rsi_lag_12', 'close_change_lag_12', 'rolling_mean_close_12',
    'rolling_std_close_12', 'rolling_mean_volume_12',
    'rolling_mean_close_24', 'rolling_std_close_24',
    'rolling_mean_volume_24', 'rolling_mean_close_144',
    'rolling_std_close_144', 'rolling_mean_volume_144',
    'market_avg_change', 'market_avg_rsi', 'coin_ADAUSDT',
    'coin_AVAXUSDT', 'coin_BTCUSDT', 'coin_DOGEUSDT', 'coin_DOTUSDT',
    'coin_ETHUSDT', 'coin_LINKUSDT', 'coin_LTCUSDT', 'coin_SHIBUSDT',
    'coin_SOLUSDT', 'coin_UNIUSDT', 'coin_XLMUSDT', 'coin_XRPUSDT'
]



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






def fetch_ohlcv(coin, limit=350):
    symbol = COINAPI_SYMBOL_MAP[coin]
    url = f"{BASE_URL}/{symbol}/history"
    params = {
        'period_id': '5MIN',
        'limit': limit,
        'time_end': datetime.utcnow().replace(microsecond=0).isoformat()
    }
    headers = {"X-CoinAPI-Key": COINAPI_KEY}
    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame([{
        'timestamp': pd.to_datetime(x['time_period_start']),
        'open': x['price_open'], 'high': x['price_high'],
        'low': x['price_low'], 'close': x['price_close'],
        'volume': x['volume_traded'], 'coin': coin
    } for x in data])
    return df.sort_values("timestamp").reset_index(drop=True)




def fetch_ohlcv_for_coin(coin_symbol, limit=350):
    """Fetches historical OHLCV data for a single coin from CoinAPI."""
    api_symbol = COINAPI_SYMBOL_MAP.get(coin_symbol)
    if not api_symbol:
        print(f"Warning: No CoinAPI symbol found for {coin_symbol}")
        return None

    url = f"https://rest.coinapi.io/v1/ohlcv/{api_symbol}/history"

    # Format the timestamp correctly for the API (no microseconds)
    time_end = datetime.utcnow().replace(microsecond=0).isoformat()

    params = {
        'period_id': '5MIN',
        'limit': limit,
        'time_end': time_end
    }
    headers = {"X-CoinAPI-Key": COINAPI_KEY}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame([{
            'timestamp': pd.to_datetime(x['time_period_start']),
            'open': x['price_open'], 'high': x['price_high'],
            'low': x['price_low'], 'close': x['price_close'],
            'volume': x['volume_traded']
        } for x in data])
        df['coin'] = coin_symbol # Add coin identifier
        return df.sort_values("timestamp").reset_index(drop=True)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {coin_symbol}: {e}")
        return None



def one_hot_encode_coin(df, coin):
    for c in COINS:
        df[f"coin_{c}"] = 1 if c == coin else 0
    return df


def add_features(df):

    df['open'] = df['open']
    df['high'] = df['high']
    df['low'] = df['low']
    df['close'] = df['close']
    df['volume'] = df['volume']

    df['rsi_14'] = RSIIndicator(df['close']).rsi()

    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    df['macd_hist_slope'] = df['macd_hist'].diff()

    from ta.momentum import StochasticOscillator
    stoch = StochasticOscillator(df['high'], df['low'], df['close'])
    df['stochastic_k'] = stoch.stoch()
    df['stochastic_d'] = stoch.stoch_signal()

    df['atr_14'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

    df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['obv_slope'] = df['obv'].diff()

    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_spike'] = (df['volume'] > 1.5 * df['volume_ma_20']).astype(int)

    df['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
    df['vwap_delta'] = df['close'] - df['vwap']

    df['price_slope_6'] = df['close'].diff(6)

    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek




    #df['ema_9'] = EMAIndicator(df['close'], window=9).ema_indicator()
    #df['ema_21'] = EMAIndicator(df['close'], window=21).ema_indicator()
    #df['ema_50'] = EMAIndicator(df['close'], window=50).ema_indicator()
    #df['ema_200'] = EMAIndicator(df['close'], window=200).ema_indicator()
    #bb = BollingerBands(df['close'])
    #df['bb_upper'] = bb.bollinger_hband()
    #df['bb_lower'] = bb.bollinger_lband()
    #df['bb_width'] = bb.bollinger_wband()
    #df['adx_14'] = ADXIndicator(df['high'], df['low'], df['close']).adx()
    #df['day_of_year'] = df['timestamp'].dt.dayofyear
    #df['month'] = df['timestamp'].dt.month


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







def run_live_pipeline_old(request=None):

    #import django
    #django.setup()

    long_model = load('gemini_model.joblib')

    # long_model = xgb.Booster()
    # long_model.load_model("eight_long_xgb_model.bin")

    #short_model = xgb.Booster()
    #short_model.load_model("seven_short_xgb_model.bin")

    for coin in COINS:
        try:
            print(f"Processing {coin}...")

            # df = fetch_ohlcv(coin, limit=350)

            #aligned_now = datetime.utcnow().replace(second=0, microsecond=0)
            #df = fetch_ohlcv(coin, limit=350, end_time=aligned_now)


            df = fetch_ohlcv(coin)
            if df.empty or len(df) < 200:
                continue


            print("\n📈 LIVE CANDLE:")
            print(df.tail(5)[['timestamp', 'open', 'high', 'low', 'close', 'volume']])


            print(df.tail(10))
            print(df.describe())
            print(df.isna().sum())
            print(f"Fetched {len(df)} rows for {coin}")

            df = add_features(df, coin)
            latest = df.iloc[-1]
            row = latest[INPUT_COLUMNS]

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

            # long_proba = long_model.predict(dmatrix)[0]
            long_proba = float(long_model.predict_proba(X)[0][1])

            #short_proba = short_model.predict(dmatrix)[0]

            print(f"{coin}: Long = {long_proba:.4f}")
            #print(f"{coin}: Short = {short_proba:.4f}")






            log_entry = {
                "timestamp": row["timestamp"].isoformat(),
                "coin": coin,
                "predicted_long_prob": float(long_proba),
                #"predicted_short_prob": float(short_proba),
                "threshold": CONFIDENCE_THRESHOLD,
                "decision_long": "LONG" if long_proba >= CONFIDENCE_THRESHOLD else "NO TRADE",
                #"decision_short": "SHORT" if short_proba >= CONFIDENCE_THRESHOLD else "NO TRADE",
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

            '''
            existing_short_trade = ModelTrade.objects.filter(
                coin=coin_obj,
                exit_timestamp__isnull=True,
                trade_type="short"
            ).first()
            '''

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




            '''
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
            '''



        except Exception as e:
            print(f"❌ Error processing {coin}: {e}")



def run_live_pipeline():
    print("🚀 Running live LightGBM pipeline")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    selected_features = joblib.load(FEATURES_PATH)

    for coin in COINS:
        try:
            df = fetch_ohlcv(coin)
            if df is None or len(df) < 288:
                print(f"⚠️ Skipping {coin} (not enough data)")
                continue

            df['coin'] = coin  # Ensure 'coin' column exists for cross-coin features
            df = add_enhanced_features(df)

            latest = df.sort_values('timestamp').iloc[-1:]
            feature_df = latest[selected_features].copy()

            if feature_df.isnull().values.any():
                print(f"⚠️ Skipping {coin} due to NaN in features")
                continue

            feature_scaled = scaler.transform(feature_df)
            prob = model.predict_proba(feature_scaled)[0][1]

            print(f"{coin}: confidence = {prob:.4f}")

            if prob >= CONFIDENCE_THRESHOLD:
                coin_symbol = COIN_SYMBOL_MAP_DB[coin]
                coin_obj = Coin.objects.get(symbol=coin_symbol)

                exists = ModelTrade.objects.filter(
                    coin=coin_obj,
                    exit_timestamp__isnull=True,
                    trade_type='long'
                ).exists()

                if not exists:
                    ModelTrade.objects.create(
                        coin=coin_obj,
                        trade_type='long',
                        entry_timestamp=make_aware(latest['timestamp'].values[0].astype('M8[ms]').astype(datetime)),
                        entry_price=latest['close'].values[0],
                        model_confidence=prob,
                        take_profit_percent=6.0,
                        stop_loss_percent=3.0,
                        confidence_trade=CONFIDENCE_THRESHOLD
                    )
                    print(f"✅ LONG trade opened for {coin} @ {latest['close'].values[0]:.4f}")
                else:
                    print(f"ℹ️ Long trade already open for {coin}")

        except Exception as e:
            print(f"❌ Error with {coin}: {e}")

    print("✅ Pipeline complete")





    

if __name__ == "__main__":
    run_live_pipeline()
