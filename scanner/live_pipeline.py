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
from scanner.models import Coin, ModelTrade, RealTrade, ConfidenceHistory, LivePriceSnapshot, LiveChart
from scipy.stats import linregress
from joblib import load

from django.utils.timezone import now, make_aware
from sklearn.preprocessing import StandardScaler
from decimal import Decimal, InvalidOperation


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

MODEL_PATH = "ten_model.joblib"
SCALER_PATH = "ten_feature_scaler.joblib"
FEATURES_PATH = "ten_selected_features.joblib"
CONFIDENCE_THRESHOLD = 0.8

#TWO_MODEL_PATH = "seven_model.joblib"
#TWO_SCALER_PATH = "seven_feature_scaler.joblib"
#TWO_FEATURES_PATH = "seven_selected_features.joblib"
#TWO_CONFIDENCE_THRESHOLD = 0.8

SHORT_MODEL_PATH = "short_four_model.joblib"
SHORT_SCALER_PATH = "short_four_feature_scaler.joblib"
SHORT_FEATURES_PATH = "short_four_selected_features.joblib"
SHORT_CONFIDENCE_THRESHOLD = 0.62

selected_features = joblib.load(FEATURES_PATH)



# Chart model setup
LABELS = ['bearish', 'bullish', 'neutral']
_vision_model = None

def load_vision_model(model_path='chart_model.pth'):
    import torch
    from torchvision import models

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()

def get_vision_model():
    global _vision_model
    if _vision_model is None:
        _vision_model = load_vision_model()
    return _vision_model


def classify_chart(image_buf):
    import torch
    from torchvision import transforms
    from PIL import Image
    try:
        model = get_vision_model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        img = Image.open(image_buf).convert("RGB")
        img_tensor = image_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_tensor)
            pred_idx = torch.argmax(logits, dim=1).item()
            return LABELS[pred_idx]
    except Exception as e:
        print(f"Failed to classify image: {e}")
        return 'neutral'


def generate_chart_image(df, coin, timestamp):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import io
    import mplfinance as mpf

    try:
        
        # Fix timestamp type if needed
        if isinstance(timestamp, np.datetime64):
            timestamp = pd.to_datetime(timestamp).to_pydatetime()

        df_plot = df.tail(60).copy()
        df_plot.set_index('timestamp', inplace=True)
        df_plot['MA20'] = df_plot['close'].rolling(20).mean()
        df_plot['MA50'] = df_plot['close'].rolling(50).mean()

        mpf_style = mpf.make_mpf_style(base_mpf_style='nightclouds', rc={'font.size': 6})
        addplots = [
            mpf.make_addplot(df_plot['MA20'], color='orange'),
            mpf.make_addplot(df_plot['MA50'], color='purple')
        ]

        fig, axlist = mpf.plot(
            df_plot,
            type='candle',
            style=mpf_style,
            title=f"{coin} - {timestamp.strftime('%Y-%m-%d %H:%M')}",
            ylabel='Price (USDT)',
            volume=True,
            addplot=addplots,
            returnfig=True,
            figsize=(6, 4)
        )

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        # Run classification on the image buffer
        chart_label = classify_chart(buf)
        print(f"🧠 Chart classification for {coin}: {chart_label}")
        buf.seek(0)  # Rewind buffer for saving
        return buf, chart_label
    except Exception as e:
        print(f"Chart generation failed for {coin} at {timestamp}: {e}")
        return None, 'neutral'



def safe_decimal(value):
    try:
        # Replace curly quotes or weird characters
        value_str = str(value).replace("“", "").replace("”", "").replace(",", "").strip()
        return Decimal(value_str)
    except (InvalidOperation, TypeError, ValueError):
        return None


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


def fetch_ohlcv(coin, limit=2100):
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


def fetch_all_ohlcv(coins):
    all_data = {}
    for coin in coins:
        try:
            df = fetch_ohlcv(coin)
            if df is not None and len(df) >= 288:
                all_data[coin] = df
        except Exception as e:
            print(f"❌ Error fetching {coin}: {e}")
    return all_data


def add_enhanced_features(df):
    import numpy as np
    import pandas as pd
    from ta.trend import EMAIndicator, SMAIndicator
    from ta.momentum import RSIIndicator, StochRSIIndicator
    from ta.volume import OnBalanceVolumeIndicator
    from ta.volatility import AverageTrueRange, BollingerBands

    df = df.copy()
    df.set_index('timestamp', inplace=True)

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_price = df['open']

    # EMAs & SMAs
    for period in [9, 21, 50, 100, 200]:
        df[f'ema_{period}'] = EMAIndicator(close, window=period).ema_indicator()
        df[f'sma_{period}'] = SMAIndicator(close, window=period).sma_indicator()

    # EMA ratios
    df['ema_9_21_ratio'] = df['ema_9'] / df['ema_21']
    df['ema_21_50_ratio'] = df['ema_21'] / df['ema_50']
    df['ema_50_200_ratio'] = df['ema_50'] / df['ema_200']
    df['price_above_ema_200'] = (close > df['ema_200']).astype(int)

    # RSI
    df['rsi_14'] = RSIIndicator(close, window=14).rsi()
    df['rsi_14_oversold'] = (df['rsi_14'] < 30).astype(int)
    df['rsi_14_overbought'] = (df['rsi_14'] > 70).astype(int)

    df['rsi_21'] = RSIIndicator(close, window=21).rsi()
    df['rsi_21_overbought'] = (df['rsi_21'] > 70).astype(int)
    df['rsi_21_oversold'] = (df['rsi_21'] < 30).astype(int)

    # MACD
    macd_line = EMAIndicator(close, window=12).ema_indicator() - EMAIndicator(close, window=26).ema_indicator()
    macd_signal = macd_line.rolling(9).mean()
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal
    df['macd_histogram'] = macd_line - macd_signal
    df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)

    # Stochastic RSI
    stoch = StochRSIIndicator(close, window=14)
    df['stoch_k'] = stoch.stochrsi_k()
    df['stoch_d'] = stoch.stochrsi_d()
    df['stoch_oversold'] = (df['stoch_k'] < 0.2).astype(int)
    df['stoch_overbought'] = (df['stoch_k'] > 0.8).astype(int)

    # OBV and volume
    df['obv'] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df['volume_sma_20'] = volume.rolling(window=20).mean()
    df['volume_ratio'] = volume / df['volume_sma_20']
    df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
    df['volume_trend'] = df['volume_sma_20'].pct_change(5)

    # ATR
    df['atr_14'] = AverageTrueRange(high, low, close, window=14).average_true_range()
    df['atr_21'] = AverageTrueRange(high, low, close, window=21).average_true_range()

    # Bollinger Bands
    bb = BollingerBands(close, window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.8).astype(int)

    # Returns (abs)
    for p in [1, 3, 6, 12, 24, 48]:
        df[f'returns_{p}p'] = close.pct_change(p)
        df[f'returns_{p}p_abs'] = np.abs(df[f'returns_{p}p'])

    # Candle structure
    df['body_size'] = np.abs(close - open_price) / open_price
    df['upper_shadow'] = (high - np.maximum(close, open_price)) / open_price
    df['lower_shadow'] = (np.minimum(close, open_price) - low) / open_price
    df['is_green'] = (close > open_price).astype(int)
    df['high_low_ratio'] = high / low
    df['close_position'] = (close - low) / (high - low)

    # Distance from 24h/7d highs and lows
    df['dist_from_high_24h'] = (close / high.rolling(288).max()) - 1
    df['dist_from_low_24h'] = (close / low.rolling(288).min()) - 1
    df['dist_from_high_7d'] = (close / high.rolling(2016).max()) - 1
    df['dist_from_low_7d'] = (close / low.rolling(2016).min()) - 1

    # Trend slopes
    for w in [6, 12, 24]:
        df[f'slope_{w}p'] = close.rolling(w).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == w else np.nan,
            raw=False
        )

    # Time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['is_us_hours'] = ((df.index.hour >= 13) & (df.index.hour <= 21)).astype(int)
    df['is_asia_hours'] = ((df.index.hour >= 0) & (df.index.hour <= 8)).astype(int)

    # Lag features
    for lag in [1, 2, 3, 6]:
        df[f'close_lag_{lag}'] = close.shift(lag)
        df[f'volume_lag_{lag}'] = volume.shift(lag)
        df[f'rsi_14_lag_{lag}'] = df['rsi_14'].shift(lag)

    df.reset_index(inplace=True)
    return df.tail(1)  # latest row only


def prepare_instance(df):
    row = df.iloc[-1]
    instance = {col: row[col] for col in selected_features}
    return instance, row


def print_feature_stats(df, coin):

    print(f"--- Feature stats for {coin} ---")

    for feature in selected_features:
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


def run_live_pipeline():

    print("🚀 Running live LightGBM pipeline")

    long_model = joblib.load(MODEL_PATH)
    long_scaler = joblib.load(SCALER_PATH)
    long_features = joblib.load(FEATURES_PATH)

    #two_long_model = joblib.load(TWO_MODEL_PATH)
    #two_long_scaler = joblib.load(TWO_SCALER_PATH)
    #two_long_features = joblib.load(TWO_FEATURES_PATH)

    short_model = joblib.load(SHORT_MODEL_PATH)
    short_scaler = joblib.load(SHORT_SCALER_PATH)
    short_features = joblib.load(SHORT_FEATURES_PATH)

    all_ohlcv = fetch_all_ohlcv(COINS)

    btc_df = add_enhanced_features(all_ohlcv["BTCUSDT"])

    btc_df['btc_bull_trend'] = (btc_df['ema_21'] > btc_df['ema_50']).astype(int)
    btc_df['btc_strong_trend'] = (btc_df['ema_9'] > btc_df['ema_21']).astype(int)

    btc_bull_trend_value = btc_df.iloc[-1]['btc_bull_trend']
    btc_strong_trend_value = btc_df.iloc[-1]['btc_strong_trend']

    for coin in COINS:

        if coin not in all_ohlcv:
            continue

        try:

            df = add_enhanced_features(all_ohlcv[coin])
            df['btc_bull_trend'] = btc_bull_trend_value
            df['btc_strong_trend'] = btc_strong_trend_value
            df['id'] = 0

            latest = df.sort_values('timestamp').iloc[-1:]
            coin_symbol = COIN_SYMBOL_MAP_DB[coin]
            coin_obj = Coin.objects.get(symbol=coin_symbol)


            # save latest price data
            LivePriceSnapshot.objects.update_or_create(
                coin=coin_symbol,
                defaults={
                    "open": safe_decimal(latest["open"].values[0]),
                    "high": safe_decimal(latest["high"].values[0]),
                    "low": safe_decimal(latest["low"].values[0]),
                    "close": safe_decimal(latest["close"].values[0]),
                    "volume": safe_decimal(latest["volume"].values[0]),
                }
            )
            

            # -------- LONG MODEL --------
            long_missing = [f for f in long_features if f not in df.columns or df[f].isnull().any()]
            if not long_missing:

                long_feature_df = latest[long_features].copy()
                long_scaled = long_scaler.transform(long_feature_df)

                long_scaled_df = pd.DataFrame(long_scaled, columns=long_features)

                long_prob = round(long_model.predict_proba(long_scaled)[0][1], 2)
                print(f"📈 {coin} LONG confidence: {long_prob:.4f}")

                ConfidenceHistory.objects.create(
                    coin=coin_obj,
                    model_name=MODEL_PATH,
                    confidence=long_prob,
                )

                oldestLongConfidence = ConfidenceHistory.objects.filter(
                    coin=coin_obj,
                    model_name=MODEL_PATH
                ).order_by("timestamp").first()

                if ConfidenceHistory.objects.filter(coin=coin_obj, model_name=MODEL_PATH).count() > 12 and oldestLongConfidence:
                    oldestLongConfidence.delete()

                if long_prob >= CONFIDENCE_THRESHOLD:

                    exists = ModelTrade.objects.filter(
                        coin=coin_obj, exit_timestamp__isnull=True, trade_type='long'
                    ).exists()

                    if not exists:

                        image_path = generate_chart_image(df, coin, latest['timestamp'].values[0])
                        if image_path:
                            chart_prediction = classify_chart(image_path)
                            print(f"🧠 Chart prediction: {chart_prediction} for {coin}")
                            if chart_prediction != 'bullish':
                                print(f"🚫 Chart model rejected trade for {coin} (label: {chart_prediction})")
                                continue

                        recent_confs = list(
                            ConfidenceHistory.objects
                            .filter(coin=coin_obj, model_name=MODEL_PATH)
                            .order_by('-timestamp')
                            .values_list('confidence', flat=True)[:6]
                        )[::-1]

                        while len(recent_confs) < 6:
                            recent_confs.insert(0, None) 

                        ModelTrade.objects.create(
                            coin=coin_obj,
                            trade_type='long',
                            entry_timestamp=make_aware(latest['timestamp'].values[0].astype('M8[ms]').astype(datetime)),
                            entry_price=safe_decimal(latest['close'].values[0]),
                            model_confidence=long_prob,
                            take_profit_percent=1.5,
                            stop_loss_percent=2.0,
                            confidence_trade=CONFIDENCE_THRESHOLD,
                            recent_confidences=recent_confs,
                        )
                        print(f"✅ LONG trade opened for {coin} @ {latest['close'].values[0]:.4f}")
                        send_text([f"LONG trade opened for {coin} @ {latest['close'].values[0]:.4f}"])
                    else:
                        print(f"ℹ️ Long trade already open for {coin}")
            else:
                print(f"❌ {coin} missing LONG features: {long_missing}")

            # -------- SHORT MODEL --------
            short_missing = [f for f in short_features if f not in df.columns or df[f].isnull().any()]

            if not short_missing:
                short_feature_df = latest[short_features].copy()
                short_scaled = short_scaler.transform(short_feature_df)
                short_prob = round(short_model.predict_proba(short_scaled)[0][1], 2)
                print(f"📉 {coin} SHORT confidence: {short_prob:.4f}")

                ConfidenceHistory.objects.create(
                    coin=coin_obj,
                    model_name=SHORT_MODEL_PATH,
                    confidence=short_prob,
                )

                oldestShortConfidence = ConfidenceHistory.objects.filter(
                    coin=coin_obj,
                    model_name=SHORT_MODEL_PATH
                ).order_by("timestamp").first()

                if ConfidenceHistory.objects.filter(coin=coin_obj, model_name=SHORT_MODEL_PATH).count() > 12 and oldestShortConfidence:
                    oldestShortConfidence.delete()

                if short_prob >= SHORT_CONFIDENCE_THRESHOLD:
                    exists = ModelTrade.objects.filter(
                        coin=coin_obj, exit_timestamp__isnull=True, trade_type='short'
                    ).exists()
                    if not exists:

                        recent_confs_short = list(
                            ConfidenceHistory.objects
                            .filter(coin=coin_obj, model_name=SHORT_MODEL_PATH)
                            .order_by('-timestamp')
                            .values_list('confidence', flat=True)[:6]
                        )[::-1]

                        while len(recent_confs_short) < 6:
                            recent_confs_short.insert(0, None)

                        ModelTrade.objects.create(
                            coin=coin_obj,
                            trade_type='short',
                            entry_timestamp=make_aware(latest['timestamp'].values[0].astype('M8[ms]').astype(datetime)),
                            entry_price=safe_decimal(latest['close'].values[0]),
                            model_confidence=short_prob,
                            take_profit_percent=1.0,
                            stop_loss_percent=2.0,
                            confidence_trade=SHORT_CONFIDENCE_THRESHOLD,
                            recent_confidences=recent_confs_short,
                        )
                        print(f"✅ SHORT trade opened for {coin} @ {latest['close'].values[0]:.4f}")
                        send_text([f"SHORT trade opened for {coin} @ {latest['close'].values[0]:.4f}"])
                    else:
                        print(f"ℹ️ Short trade already open for {coin}")
            else:
                print(f"❌ {coin} missing SHORT features: {short_missing}")


            # -------- LONG MODEL 2 --------
            '''
            two_missing = [f for f in two_long_features if f not in df.columns or df[f].isnull().any()]
            if not two_missing:
                two_feature_df = latest[two_long_features].copy()
                two_scaled = two_long_scaler.transform(two_feature_df)
                two_prob = round(two_long_model.predict_proba(two_scaled)[0][1], 2)
                print(f"📈 {coin} LONG (Model 2) confidence: {two_prob:.4f}")

                ConfidenceHistory.objects.create(
                    coin=coin_obj,
                    model_name=TWO_MODEL_PATH,
                    confidence=two_prob,
                )

                oldestTwoConfidence = ConfidenceHistory.objects.filter(
                    coin=coin_obj,
                    model_name=TWO_MODEL_PATH
                ).order_by("timestamp").first()

                if ConfidenceHistory.objects.filter(coin=coin_obj, model_name=TWO_MODEL_PATH).count() > 12 and oldestTwoConfidence:
                    oldestTwoConfidence.delete()

                if two_prob >= TWO_CONFIDENCE_THRESHOLD:

                    exists = ModelTrade.objects.filter(
                        coin=coin_obj, exit_timestamp__isnull=True, trade_type='long'
                    ).filter(model_confidence__gte=TWO_CONFIDENCE_THRESHOLD).exists()

                    if not exists:
                        recent_confs_two = list(
                            ConfidenceHistory.objects
                            .filter(coin=coin_obj, model_name=TWO_MODEL_PATH)
                            .order_by('-timestamp')
                            .values_list('confidence', flat=True)[:6]
                        )[::-1]

                        while len(recent_confs_two) < 6:
                            recent_confs_two.insert(0, None)

                        ModelTrade.objects.create(
                            coin=coin_obj,
                            trade_type='long',
                            entry_timestamp=make_aware(latest['timestamp'].values[0].astype('M8[ms]').astype(datetime)),
                            entry_price=safe_decimal(latest['close'].values[0]),
                            model_confidence=two_prob,
                            take_profit_percent=2.0,
                            stop_loss_percent=2.0,
                            confidence_trade=TWO_CONFIDENCE_THRESHOLD,
                            recent_confidences=recent_confs_two,
                        )
                        print(f"✅ LONG (Model 2) trade opened for {coin} @ {latest['close'].values[0]:.4f}")
                        send_text([f"LONG (Model 2) trade opened for {coin} @ {latest['close'].values[0]:.4f}"])
                    else:
                        print(f"ℹ️ Long (Model 2) trade already open for {coin}")
            else:
                print(f"❌ {coin} missing LONG Model 2 features: {two_missing}")
            '''

        except Exception as e:
            print(f"❌ Error with {coin}: {e}")




    print("\n🔍 Evaluating open trades...")
    open_trades = ModelTrade.objects.filter(exit_timestamp__isnull=True)

    for trade in open_trades:
        try:
            price_entry = float(trade.entry_price)
            coin_symbol = trade.coin.symbol + "USDT"
            df = all_ohlcv.get(coin_symbol)

            if df is None or df.empty:
                print(f"⚠️ No price data for {coin_symbol}, skipping")
                continue

            price_now = float(df.iloc[-1]['close'])
            result = True

            if trade.trade_type == "long":
                if price_now >= price_entry * 1.015:
                    status = "💰 TAKE PROFIT"
                elif price_now <= price_entry * 0.98:
                    status = "🛑 STOP LOSS"
                    result = False
                else:
                    continue
            else:
                if price_now <= price_entry * 0.99:
                    status = "💰 TAKE PROFIT"
                elif price_now >= price_entry * 1.02:
                    status = "🛑 STOP LOSS"
                    result = False
                else:
                    continue

            trade.exit_price = price_now
            trade.exit_timestamp = now()
            trade.result = result
            trade.save()

            print(f"{status} | {trade.trade_type.upper()} {trade.coin.symbol} @ {price_now:.6f}")

        except Exception as e:
            print(f"❌ Error closing trade for {trade.coin.symbol}: {e}")


    print("✅ Pipeline complete")
    

if __name__ == "__main__":
    run_live_pipeline()
