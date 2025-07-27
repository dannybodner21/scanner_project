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
import pytz

from datetime import datetime, timedelta
from scanner.models import Coin, MemoryTrade, ModelTrade, RealTrade, ConfidenceHistory, LivePriceSnapshot, LiveChart, CoinAPIPrice
from scipy.stats import linregress
from joblib import load

from django.utils.timezone import now, make_aware
from sklearn.preprocessing import StandardScaler
from decimal import Decimal, InvalidOperation

import openai
from openai import OpenAI
from io import BytesIO
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files import File


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

MODEL_PATH = "four_model.joblib"  # UPDATED: Latest LONG model
SCALER_PATH = "four_feature_scaler.joblib"
FEATURES_PATH = "four_selected_features.joblib"
CONFIDENCE_THRESHOLD = 0.80  # OPTIMIZED: Raised for better selectivity

#TWO_MODEL_PATH = "seven_model.joblib"
#TWO_SCALER_PATH = "seven_feature_scaler.joblib"
#TWO_FEATURES_PATH = "seven_selected_features.joblib"
#TWO_CONFIDENCE_THRESHOLD = 0.8

SHORT_MODEL_PATH = "short_three_model.joblib"  # UPDATED: Latest SHORT model
SHORT_SCALER_PATH = "short_three_feature_scaler.joblib"
SHORT_FEATURES_PATH = "short_three_selected_features.joblib"
SHORT_CONFIDENCE_THRESHOLD = 0.68  # OPTIMIZED: Raised for better selectivity

selected_features = joblib.load(FEATURES_PATH)

# ask Chat GPT
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# === ADVANCED PROFITABILITY FILTERS ===

def passes_advanced_long_filters(row, ml_confidence, coin):
    """
    Advanced filtering system for LONG trades to maximize profitability
    Returns True if trade should proceed, False to reject
    """
    
    # FILTER 1: Volume Confirmation
    volume_ratio = getattr(row, 'volume_ratio', 1.0)
    if volume_ratio < 1.2:  # Require above-average volume
        print(f"🚫 {coin} LONG: Low volume ({volume_ratio:.2f})")
        return False
    
    # FILTER 2: Momentum Confluence Check
    rsi_14 = getattr(row, 'rsi_14', 50)
    ema_9_21_ratio = getattr(row, 'ema_9_21_ratio', 1.0)
    macd = getattr(row, 'macd', 0)
    macd_signal = getattr(row, 'macd_signal', 0)
    
    bullish_momentum = (
        rsi_14 < 35 and  # Oversold condition
        ema_9_21_ratio > 1.001 and  # Short EMA above long EMA
        macd > macd_signal  # MACD bullish
    )
    
    if not bullish_momentum and ml_confidence < 0.85:
        print(f"🚫 {coin} LONG: Weak momentum confluence")
        return False
    
    # FILTER 3: Market Structure Check
    close = getattr(row, 'close', 0)
    ema_200 = getattr(row, 'ema_200', close)
    
    if close < ema_200 and ml_confidence < 0.80:  # Below major trend
        print(f"🚫 {coin} LONG: Below 200 EMA trend")
        return False
    
    # FILTER 4: Volatility Regime
    atr_14 = getattr(row, 'atr_14', 0)
    if atr_14 == 0:  # Avoid dead markets
        print(f"🚫 {coin} LONG: No volatility")
        return False
    
    # FILTER 5: Time-based filtering (avoid low-volume hours)
    current_hour = datetime.utcnow().hour
    if current_hour in [1, 2, 3, 4, 5, 6] and ml_confidence < 0.80:  # Asian low-volume hours
        print(f"🚫 {coin} LONG: Low-volume time period")
        return False
    
    # FILTER 6: Anti-correlation with other positions
    if has_too_many_correlated_positions(coin, 'long'):
        print(f"🚫 {coin} LONG: Too many correlated positions")
        return False
    
    print(f"✅ {coin} LONG: Passed all advanced filters")
    return True


def passes_advanced_short_filters(row, ml_confidence, coin):
    """
    Advanced filtering system for SHORT trades to maximize profitability
    Returns True if trade should proceed, False to reject
    """
    
    # FILTER 1: Volume Confirmation
    volume_ratio = getattr(row, 'volume_ratio', 1.0)
    if volume_ratio < 1.2:  # Require above-average volume
        print(f"🚫 {coin} SHORT: Low volume ({volume_ratio:.2f})")
        return False
    
    # FILTER 2: Bearish Momentum Confluence
    rsi_14 = getattr(row, 'rsi_14', 50)
    ema_9_21_ratio = getattr(row, 'ema_9_21_ratio', 1.0)
    macd = getattr(row, 'macd', 0)
    macd_signal = getattr(row, 'macd_signal', 0)
    
    bearish_momentum = (
        rsi_14 > 65 and  # Overbought condition
        ema_9_21_ratio < 0.999 and  # Short EMA below long EMA
        macd < macd_signal  # MACD bearish
    )
    
    if not bearish_momentum and ml_confidence < 0.75:
        print(f"🚫 {coin} SHORT: Weak bearish momentum")
        return False
    
    # FILTER 3: Resistance/Distribution Check
    close = getattr(row, 'close', 0)
    high = getattr(row, 'high', close)
    open_price = getattr(row, 'open', close)
    
    # Look for distribution patterns (long upper shadows)
    upper_shadow = (high - max(close, open_price)) / open_price if open_price > 0 else 0
    body_size = abs(close - open_price) / open_price if open_price > 0 else 0
    
    strong_distribution = (
        upper_shadow > body_size * 1.5 and  # Long upper shadow
        close < open_price and  # Red candle
        volume_ratio > 1.5  # High volume
    )
    
    if not strong_distribution and ml_confidence < 0.70:
        print(f"🚫 {coin} SHORT: No distribution pattern")
        return False
    
    # FILTER 4: Volatility Expansion Check
    atr_14 = getattr(row, 'atr_14', 0)
    if atr_14 == 0:  # Avoid dead markets
        print(f"🚫 {coin} SHORT: No volatility")
        return False
    
    # FILTER 5: Market Regime Check (prefer range-bound for shorts)
    bb_position = getattr(row, 'bb_position', 0.5)
    if bb_position < 0.7 and ml_confidence < 0.70:  # Not near upper BB
        print(f"🚫 {coin} SHORT: Not overbought enough")
        return False
    
    # FILTER 6: Risk-off sentiment confirmation
    # In real implementation, this could check BTC trend, VIX equivalent, etc.
    
    # FILTER 7: Anti-correlation with other positions
    if has_too_many_correlated_positions(coin, 'short'):
        print(f"🚫 {coin} SHORT: Too many correlated positions")
        return False
    
    print(f"✅ {coin} SHORT: Passed all advanced filters")
    return True


def has_too_many_correlated_positions(coin, trade_type):
    """
    Prevent over-concentration in correlated assets
    """
    # Get current open positions
    open_positions = ModelTrade.objects.filter(
        exit_timestamp__isnull=True,
        trade_type=trade_type
    ).count()
    
    # Limit to maximum 3 positions per direction
    if open_positions >= 3:
        return True
    
    # Check for highly correlated coins (simplified)
    correlated_groups = [
        ['BTCUSDT', 'ETHUSDT'],  # Major cryptos
        ['DOGEUSDT', 'SHIBUSDT'],  # Meme coins
        ['ADAUSDT', 'DOTUSDT', 'XLMUSDT'],  # Alt coins
        ['UNIUSDT', 'LINKUSDT', 'AVAXUSDT']  # DeFi/utility
    ]
    
    for group in correlated_groups:
        if coin in group:
            # Check if we already have a position in this group
            group_positions = ModelTrade.objects.filter(
                exit_timestamp__isnull=True,
                trade_type=trade_type,
                coin__symbol__in=[COIN_SYMBOL_MAP_DB.get(c, c) for c in group]
            ).count()
            
            if group_positions >= 1:  # Already have one in this group
                return True
    
    return False


def get_market_regime():
    """
    Determine current market regime (trending/ranging/volatile)
    This affects our filtering thresholds
    """
    try:
        # Get BTC data for market regime
        btc_data = CoinAPIPrice.objects.filter(
            coin='BTCUSDT'
        ).order_by('-timestamp')[:48]  # Last 4 hours
        
        if len(btc_data) < 48:
            return 'unknown'
        
        df = pd.DataFrame.from_records(btc_data.values())
        df['close'] = pd.to_numeric(df['close'])
        
        # Calculate volatility
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * 100
        
        # Calculate trend strength
        price_change = (df['close'].iloc[0] - df['close'].iloc[-1]) / df['close'].iloc[-1] * 100
        
        if volatility > 2.0:
            return 'volatile'
        elif abs(price_change) > 3.0:
            return 'trending'
        else:
            return 'ranging'
            
    except Exception as e:
        print(f"Error determining market regime: {e}")
        return 'unknown'


# to format all the memory trades before sending to ChatGPT
def format_trade_memory(trade):
    feature_summary = ", ".join(
        f"{k}: {float(v):.3f}" if isinstance(v, (float, int, np.float64, np.float32, Decimal)) else f"{k}: {v}"
        for k, v in trade.features.items()
    )

    return (
        f"Coin: {trade.coin}, Time: {trade.timestamp.strftime('%Y-%m-%d %H:%M')}, "
        f"ML Confidence: {trade.ml_confidence:.2f}, "
        f"{'GPT Confidence: {:.2f}'.format(trade.gpt_confidence) if trade.gpt_confidence is not None else 'GPT Confidence: N/A'}, "
        f"Outcome: {trade.outcome}, Features: [{feature_summary}]"
    )


# things that can help improvement
# add a 1 hour chart image
# recent trade outcomes - record all trades by the current ml model with confidence from ml and agent
# BTC market context - 24h change, macd crossed, above ema 50
# could add written context about what the chart pattern shows
# only send the clean features - don't dilute the data
# make sure there are no NaN values

def gpt_filter_trade(coin, timestamp, features, chart_path, chart_path_30m):
    try:

        time.sleep(0.5)

        with open(chart_path, "rb") as f:
            image_bytes = f.read()

        with open(chart_path_30m, "rb") as f:
            chart_30m_bytes = f.read()

        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        base64_image_30min = base64.b64encode(chart_30m_bytes).decode("utf-8")

        wins = MemoryTrade.objects.filter(outcome="win").order_by('-timestamp')[:50]
        losses = MemoryTrade.objects.filter(outcome="loss").order_by('-timestamp')[:50]
        closed_trades = sorted(list(wins) + list(losses), key=lambda t: t.timestamp, reverse=True)

        memory_lines = [format_trade_memory(trade) for trade in closed_trades]
        memory_text = "\n\n".join(memory_lines)

        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                },
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image_30min}"
                },
            },
            {
                "type": "text",
                "text": f"Coin: {coin}\nTimestamp: {timestamp}\nFeatures:\n" +
                        "\n".join([
                            f"{k}: {float(v):.6f}" if isinstance(v, (int, float, np.float32, np.float64, Decimal)) else f"{k}: {v}"
                            for k, v in features.items()
                        ])
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an elite cryptocurrency day trader who has made millions trading 5-minute candlestick chart setups. Your job is to review each setup (chart + features) and predict the probability that the price will increase by at least 2 percent BEFORE decreasing by 1 percent. There should be both a 5 min chart pattern and a 30 min chart pattern attached. Use your deep knowledge of candlestick patterns, trend indicators, and ML predictions. A machine learning model has already pre-filtered these trades. We want to approve those with very high-quality signals. Take ML confidence into account but do not over-rely on it. Below are 100 recent trades with their inputs and outcomes. Learn from them.\n {memory_text} \n Use the following guidelines to create your score between 0.00 and 1.00: Above 0.95 = extremely likely winner, 0.90 to 0.94 = strong edge, probable winner, 0.85 to 0.89 = decent edge, cautious approval, Below 0.85 = you do not approve. Reject setups that show weakness, sideways chop, or unclear direction — even if ML score is high. You may be shown results of past trades as memory — use them to calibrate your decision-making. Respond with only your confidence score (0.00 to 1.00), rounded to two decimal places."},
                {"role": "user", "content": content}
            ],
            max_tokens=50,
        )

        decision = response.choices[0].message.content.lower().strip()
        gpt_conf = float(decision)
        print(f"🧠 GPT confidence: {gpt_conf:.2f}")
        return gpt_conf

    except Exception as e:
        print(f"❌ GPT filter failed: {e}")
        return 0.0


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
        if isinstance(image_buf, tuple):
            image_buf = image_buf[0]

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


def generate_chart_image(coin, timestamp, df, output_dir="chart_images"):
    import os
    import pandas as pd
    import mplfinance as mpf
    from datetime import datetime

    os.makedirs(output_dir, exist_ok=True)

    try:
        # Preprocess
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()

        # Force numeric columns to float
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)

        # Add moving averages
        df['MA20'] = df['close'].rolling(20).mean()
        df['MA50'] = df['close'].rolling(50).mean()

        # Get the last 60 candles
        chart_data = df.tail(60)

        # Validate data
        required_cols = ['open', 'high', 'low', 'close', 'MA20', 'MA50']
        if chart_data[required_cols].isnull().any().any():
            print(f"⚠️ Chart skipped for {coin} — missing values at {timestamp}")
            return None

        if chart_data.empty or chart_data[['open', 'high', 'low', 'close']].isna().any().any():
            print(f"⚠️ Insufficient data to generate chart for {coin} at {timestamp}")
            return None

        # MA Overlays
        addplots = [
            mpf.make_addplot(chart_data['MA20'], color='orange', width=1.2),
            mpf.make_addplot(chart_data['MA50'], color='blue', width=1.2)
        ]

        image_path = os.path.join(output_dir, f"{coin}.png")

        mpf.plot(
            chart_data,
            type='candle',
            style='charles',
            volume=False,
            addplot=addplots,
            title=f"{coin} - {timestamp.strftime('%Y-%m-%d %H:%M')}",
            savefig=image_path
        )
        return image_path

    except Exception as e:
        print(f"❌ Chart generation failed for {coin} at {timestamp}: {e}")
        return None


def generate_chart_image_30m(coin, timestamp, df_5m):
    import matplotlib.pyplot as plt
    import mplfinance as mpf
    import pandas as pd
    from io import BytesIO
    import os

    """
    Generate a 30-minute candlestick chart image for the given coin and timestamp.
    Uses the previous 48 30-min candles (~24 hours of data).
    """

    # Convert timestamp if needed
    df_5m = df_5m.copy()
    df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'])
    df_5m.set_index('timestamp', inplace=True)

    # Resample to 30-minute candles
    df_30m = df_5m.resample('30min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    for col in ["open", "high", "low", "close", "volume"]:
        df_30m[col] = pd.to_numeric(df_30m[col], errors="coerce")

    df_30m = df_30m.dropna(subset=["open", "high", "low", "close", "volume"])

    # Add moving averages (optional)
    df_30m['ma9'] = df_30m['close'].rolling(window=9).mean()
    df_30m['ma21'] = df_30m['close'].rolling(window=21).mean()

    # Limit to the last 48 candles (~24 hours)
    df_30m = df_30m.iloc[-48:]

    # Build filename and directory
    filename = f"aaa{coin}_30m_{timestamp.strftime('%Y%m%d_%H%M')}.png"
    save_path = os.path.join("chart_images", filename)
    os.makedirs("chart_images", exist_ok=True)

    # Generate plot
    mpf.plot(
        df_30m,
        type='candle',
        style='yahoo',
        mav=(9, 21),
        volume=True,
        figsize=(10, 6),
        savefig=save_path
    )

    return save_path






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


def fetch_latest_candle(coin):

    symbol = COINAPI_SYMBOL_MAP[coin]
    now = datetime.utcnow().replace(second=0, microsecond=0)
    start = now - timedelta(minutes=5)

    url = f"{BASE_URL}/{symbol}/history"
    params = {
        'period_id': '5MIN',
        'time_start': start.isoformat(),
        'time_end': now.isoformat(),
        'limit': 1
    }
    headers = {"X-CoinAPI-Key": COINAPI_KEY}

    try:
        r = requests.get(url, headers=headers, params=params)
        r.raise_for_status()
        data = r.json()
        if not data:
            print(f"⚠️ No data returned for {coin}")
            return None

        x = data[0]
        return {
            'coin': coin,
            'timestamp': pd.to_datetime(x['time_period_start']).replace(tzinfo=pytz.UTC),
            'open': x['price_open'],
            'high': x['price_high'],
            'low': x['price_low'],
            'close': x['price_close'],
            'volume': x['volume_traded']
        }
    except Exception as e:
        print(f"❌ Error fetching {coin}: {e}")
        return None


# -- Save if not already stored --
def save_candle_if_missing(candle):

    if not candle:
        return

    exists = CoinAPIPrice.objects.filter(
        coin=candle['coin'],
        timestamp=candle['timestamp']
    ).exists()

    if exists:
        print(f"⏩ {candle['coin']} @ {candle['timestamp']} already exists")
        return

    CoinAPIPrice.objects.create(
        coin=candle['coin'],
        timestamp=candle['timestamp'],
        open=candle['open'],
        high=candle['high'],
        low=candle['low'],
        close=candle['close'],
        volume=candle['volume']
    )
    print(f"✅ Saved: {candle['coin']} @ {candle['timestamp']}")


def get_latest_saved_timestamp(coin):

    obj = CoinAPIPrice.objects.filter(coin=coin).order_by('-timestamp').first()
    return obj.timestamp if obj else None


def fetch_missing_candles(coin, start_ts, end_ts):

    symbol = COINAPI_SYMBOL_MAP[coin]
    url = f"{BASE_URL}/{symbol}/history"
    headers = {"X-CoinAPI-Key": COINAPI_KEY}

    minutes_missing = int((end_ts - start_ts).total_seconds() / 60)
    candles_needed = max(1, minutes_missing // 5)

    if candles_needed > 1000:
        candles_needed = 1000  # avoid credit explosion

    params = {
        'period_id': '5MIN',
        'time_start': start_ts.isoformat(),
        'limit': candles_needed
    }

    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()
    return r.json()


def save_missing_candles(coin):

    now = datetime.utcnow().replace(second=0, microsecond=0, tzinfo=pytz.UTC)
    last_saved = get_latest_saved_timestamp(coin)

    if last_saved is None:
        print(f"⚠️ No candles saved yet for {coin}")
        return

    next_needed = last_saved + timedelta(minutes=5)
    if next_needed >= now:
        print(f"✅ {coin} is already up to date")
        return

    print(f"🔄 Fetching missing candles for {coin} from {next_needed} to {now}")

    try:
        candles = fetch_missing_candles(coin, next_needed, now)
        for x in candles:
            CoinAPIPrice.objects.update_or_create(
                coin=coin,
                timestamp=pd.to_datetime(x['time_period_start']),
                defaults={
                    'open': x['price_open'],
                    'high': x['price_high'],
                    'low': x['price_low'],
                    'close': x['price_close'],
                    'volume': x['volume_traded'],
                }
            )
        print(f"✅ Inserted {len(candles)} candles for {coin}")
    except Exception as e:
        print(f"❌ Error backfilling {coin}: {e}")


# expects coin as BTCUSDT format
def get_recent_candles(coin, limit=2016):

    qs = (
        CoinAPIPrice.objects
        .filter(coin=coin)
        .order_by('-timestamp')
        .values('timestamp', 'open', 'high', 'low', 'close', 'volume')
    )
    df = pd.DataFrame(list(qs[:limit]))
    if len(df) < limit:
        return None
    return df.sort_values('timestamp').reset_index(drop=True)


# used to backfill the last 72 candles so there are no gaps
def backfill_recent_candles(coin):

    print(f"🔁 Backfilling recent candles for {coin}...")
    symbol = COINAPI_SYMBOL_MAP[coin]
    headers = {"X-CoinAPI-Key": COINAPI_KEY}

    end = datetime.utcnow().replace(second=0, microsecond=0, tzinfo=pytz.UTC)
    start = end - timedelta(minutes=60)

    chunk_minutes = 5 * 100  # 100 candles max
    current = start

    while current < end:
        chunk_end = min(current + timedelta(minutes=chunk_minutes), end)

        params = {
            'period_id': '5MIN',
            'time_start': current.isoformat(),
            'time_end': chunk_end.isoformat(),
            'limit': 100
        }

        try:
            r = requests.get(f"{BASE_URL}/{symbol}/history", headers=headers, params=params)
            r.raise_for_status()
            candles = r.json()

            for x in candles:

                ts = pd.Timestamp(x['time_period_start'], tz='UTC')

                CoinAPIPrice.objects.update_or_create(
                    coin=coin,
                    timestamp=ts,
                    defaults={
                        'open': x['price_open'],
                        'high': x['price_high'],
                        'low': x['price_low'],
                        'close': x['price_close'],
                        'volume': x['volume_traded'],
                    }
                )
            print(f"✅ {len(candles)} candles inserted for {coin} from {current} to {chunk_end}")

        except Exception as e:
            print(f"❌ Error backfilling {coin} from {current} to {chunk_end}: {e}")

        current = chunk_end
        time.sleep(1)  # avoid rate limits


def add_enhanced_features(df):
    import numpy as np
    import pandas as pd
    from ta.trend import EMAIndicator, SMAIndicator
    from ta.momentum import RSIIndicator, StochRSIIndicator
    from ta.volume import OnBalanceVolumeIndicator
    from ta.volatility import AverageTrueRange, BollingerBands

    df = df.copy()
    df.set_index('timestamp', inplace=True)

    #close = df['close']
    #high = df['high']
    #low = df['low']
    #volume = df['volume']
    #open_price = df['open']

    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)
    open_price = df['open'].astype(float)

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


def has_recent_2016_candles(coin):

    candles = CoinAPIPrice.objects.filter(coin=coin).order_by('-timestamp')[:2016]
    if len(candles) < 2016:
        return False

    timestamps = sorted([c.timestamp for c in candles])
    expected_diff = timedelta(minutes=5)

    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i - 1] != expected_diff:
            return False

    return True


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



    # 🛠️ Ensure each coin has at least 2016 candles in DB
    for coin in COINS:
        coin_symbol = coin

        if not has_recent_2016_candles(coin_symbol):
            print(f"⚠️ {coin} is fcked -> backfilling to 7d...")
            backfill_recent_candles(coin_symbol)


    # Pull latest candle per coin and save
    latest_candles = {}
    for coin in COINS:

        candle = fetch_latest_candle(coin)
        save_candle_if_missing(candle)

        if candle:
            latest_candles[coin] = candle

    # Exit early if BTC candle is missing
    if "BTCUSDT" not in latest_candles:
        print("❌ BTCUSDT candle missing, cannot proceed")
        return

    btc_recent = get_recent_candles("BTCUSDT", limit=2016)
    if btc_recent is None:
        print("❌ Not enough BTCUSDT data for trend indicators")
        return

    btc_df = add_enhanced_features(btc_recent)

    btc_df['btc_bull_trend'] = (btc_df['ema_21'] > btc_df['ema_50']).astype(int)
    btc_df['btc_strong_trend'] = (btc_df['ema_9'] > btc_df['ema_21']).astype(int)

    btc_bull_trend_value = btc_df.iloc[-1]['btc_bull_trend']
    btc_strong_trend_value = btc_df.iloc[-1]['btc_strong_trend']


    for coin in COINS:

        if coin not in latest_candles:
            continue

        try:
            # Get 2016 most recent candles from DB
            recent_df = get_recent_candles(coin, limit=2016)

            if recent_df is None:
                print(f"⏭️ Skipping {coin}, not enough candles in DB")
                continue

            df = add_enhanced_features(recent_df)
                        
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

                    # === ENHANCED PROFITABILITY FILTERING ===
                    if not passes_advanced_long_filters(latest.iloc[0], long_prob, coin):
                        print(f"🚫 REJECTED by advanced filters: LONG trade for {coin}")
                        continue

                    exists = ModelTrade.objects.filter(
                        coin=coin_obj, exit_timestamp__isnull=True, trade_type='long'
                    ).exists()

                    if not exists:

                        latest_row = latest.iloc[0]
                        features = {
                            col: float(latest_row[col])
                            if isinstance(latest_row[col], (np.float64, np.float32, Decimal))
                            else int(latest_row[col])
                            if isinstance(latest_row[col], (np.int64, np.int32))
                            else latest_row[col]
                            for col in latest_row.index
                            if col not in ['timestamp', 'prediction', 'open']
                        }
                        
                        timestamp = pd.to_datetime(latest['timestamp'].values[0])
                        timestamp = make_aware(timestamp)

                        chart_path = generate_chart_image(coin, timestamp, recent_df)
                        chart_path_30m = generate_chart_image_30m(coin, timestamp, recent_df)
                        chart_file_30m = open(chart_path_30m, "rb") if chart_path_30m else None

                        if not chart_path:
                            continue

                        existing = LiveChart.objects.filter(coin=coin).first()

                        # delete old image if it exists, then save the new one
                        if existing and existing.image:
                            existing.image.delete(save=False)
                            
                        # Save to LiveChart model (overwrite existing entry per coin)
                        with open(chart_path, "rb") as img_file:
                            chart_bytes = img_file.read()
                            #file_name = os.path.basename(chart_path)
                            file_name = f"{coin}.png"

                            LiveChart.objects.update_or_create(
                                coin=coin,
                                defaults={
                                    "timestamp": timestamp,
                                    "image": ContentFile(chart_bytes, name=file_name),
                                }
                            )

                        decision = gpt_filter_trade(coin, timestamp, features, chart_path, chart_path_30m)

                        # save memory trade
                        chart_file = open(chart_path, "rb")
                        memory_trade = MemoryTrade.objects.create(
                            coin=coin,
                            timestamp=timestamp,
                            trade_type='long',
                            ml_confidence=long_prob,
                            gpt_confidence=decision,
                            leverage=10,
                            tp_percent=2.00,  # OPTIMIZED: Better risk/reward
                            sl_percent=1.00,  # OPTIMIZED: Tighter risk control
                            features=features,  # full feature dict
                            chart_image_5min=File(chart_file) if chart_file else None,
                            chart_image_30min=File(chart_file_30m) if chart_file_30m else None,
                            outcome='open',
                            entry_price = safe_decimal(latest['close'].values[0]) * Decimal("1.001"),
                        )

                        if chart_file:
                            chart_file.close()
                        if chart_file_30m:
                            chart_file_30m.close()


                        # ENHANCED DYNAMIC THRESHOLD based on market regime
                        market_regime = get_market_regime()
                        gpt_threshold = {
                            'volatile': 0.90,   # Higher threshold in volatile markets
                            'trending': 0.85,   # Standard threshold in trending markets  
                            'ranging': 0.88,    # Higher threshold in ranging markets
                            'unknown': 0.87     # Default threshold
                        }.get(market_regime, 0.87)
                        
                        if decision < gpt_threshold:
                            print(f"🚫 REJECTED by Chat GPT: {coin} ({decision:.2f} < {gpt_threshold:.2f}) [{market_regime} market]")
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
                            entry_price = safe_decimal(latest['close'].values[0]) * Decimal("1.001"),
                            model_confidence=long_prob,
                            take_profit_percent=2.5,
                            stop_loss_percent=1.5,
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
                    
                    # === ENHANCED PROFITABILITY FILTERING ===
                    if not passes_advanced_short_filters(latest.iloc[0], short_prob, coin):
                        print(f"🚫 REJECTED by advanced filters: SHORT trade for {coin}")
                        continue
                    
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
                            entry_price = safe_decimal(latest['close'].values[0]) * Decimal("0.999"),
                            model_confidence=short_prob,
                            take_profit_percent=2.0,  # OPTIMIZED: 2:1 risk/reward  
                            stop_loss_percent=1.0,   # OPTIMIZED: Tighter risk control
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



    # evaluate any open trades executed by ML model and ChatGPT
    print("\n🔍 Evaluating open trades...")
    open_trades = ModelTrade.objects.filter(exit_timestamp__isnull=True)

    for trade in open_trades:
        try:

            price_entry = float(trade.entry_price)
            coin_symbol = trade.coin.symbol + "USDT"

            df = get_recent_candles(coin_symbol, limit=1)
            if df is None or df.empty:
                print(f"⚠️ No price data for {coin_symbol}, skipping")
                continue

            price_high = float(df.iloc[-1]['high'])
            price_low = float(df.iloc[-1]['low'])
            price_now = float(df.iloc[-1]['close'])
            result = True

            if trade.trade_type == "long":

                if price_low <= price_entry * 0.985:
                    status = "🛑 STOP LOSS"
                    result = False
                elif price_high >= price_entry * 1.025:
                    status = "💰 TAKE PROFIT"
                else:
                    continue
            else:
                if price_high >= price_entry * 1.01:
                    status = "🛑 STOP LOSS"
                    result = False
                elif price_low <= price_entry * 0.98:
                    status = "💰 TAKE PROFIT"
                else:
                    continue

            trade.exit_price = price_now
            trade.exit_timestamp = now()
            trade.result = result
            trade.save()

            print(f"{status} | {trade.trade_type.upper()} {trade.coin.symbol} @ {price_now:.6f}")

        except Exception as e:
            print(f"❌ Error closing trade for {trade.coin.symbol}: {e}")


    # evaluate trades used for ChatGPT memory
    print("\n📚 Evaluating memory trades...")
    memory_trades = MemoryTrade.objects.filter(outcome='open')

    for trade in memory_trades:
        try:
            price_entry = float(trade.entry_price)
            coin_symbol = trade.coin

            df = get_recent_candles(trade.coin, limit=1)
            if df is None or df.empty:
                print(f"⚠️ No price data for {coin_symbol}, skipping")
                continue

            price_now = float(df.iloc[-1]['close'])
            price_high = float(df.iloc[-1]['high'])
            price_low = float(df.iloc[-1]['low'])

            if trade.trade_type == "long":

                if price_low <= price_entry * 0.985:
                    trade.outcome = "loss"
                elif price_high >= price_entry * 1.025:
                    trade.outcome = "win"
                else:
                    continue

            elif trade.trade_type == "short":

                if price_high >= price_entry * 1.01:
                    trade.outcome = "loss"
                elif price_low <= price_entry * 0.98:
                    trade.outcome = "win"
                else:
                    continue

            trade.exit_price = price_now
            trade.exit_timestamp = now()
            trade.save()

            print(f"📘 Memory {trade.outcome.upper()} | {trade.trade_type.upper()} {trade.coin} @ {price_now:.6f}")

        except Exception as e:
            print(f"❌ Error closing memory trade for {trade.coin}: {e}")



    print("✅ Pipeline complete")
    

if __name__ == "__main__":
    run_live_pipeline()
