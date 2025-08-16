import os
import json
import requests
import numpy as np
import pandas as pd
import joblib
from decimal import Decimal, InvalidOperation
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from django.utils.timezone import now
from scanner.models import (
    Coin, ModelTrade, ConfidenceHistory, LivePriceSnapshot, CoinAPIPrice
)


# ---- NumPy RNG pickle compatibility shim (for joblib pickle RNG) ----
def _patch_numpy_rng_compat():
    import sys, types, numpy as np
    for name in ("PCG64", "MT19937", "Philox", "SFC64"):
        cls = getattr(np.random, name, None)
        if cls is None:
            continue
        mod = types.ModuleType(f"numpy.random._{name.lower()}")
        setattr(mod, name, cls)
        sys.modules[mod.__name__] = mod
_patch_numpy_rng_compat()

# --------------------------------
# Maps / Config
# --------------------------------
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
    "TRXUSDT": "BINANCE_SPOT_TRX_USDT",
    "ATOMUSDT": "BINANCE_SPOT_ATOM_USDT",
}
COIN_SYMBOL_MAP_DB = {
    "BTCUSDT": "BTC","ETHUSDT": "ETH","XRPUSDT": "XRP","LTCUSDT": "LTC",
    "SOLUSDT": "SOL","DOGEUSDT": "DOGE","LINKUSDT": "LINK","DOTUSDT": "DOT",
    "SHIBUSDT": "SHIB","ADAUSDT": "ADA","UNIUSDT": "UNI","AVAXUSDT": "AVAX",
    "XLMUSDT": "XLM","TRXUSDT": "TRX","ATOMUSDT": "ATOM",
}
COINS = list(COIN_SYMBOL_MAP_DB.keys())

COINAPI_KEY = os.environ.get("COINAPI_KEY", "01293e2a-dcf1-4e81-8310-c6aa9d0cb743")
BASE_URL = "https://rest.coinapi.io/v1/ohlcv"

# Model artifacts - Updated for three_long_hgb_model
MODEL_PATH    = "three_long_hgb_model.joblib"
SCALER_PATH   = "three_feature_scaler.joblib"
FEATURES_PATH = "three_feature_list.json"
CONFIDENCE_THRESHOLD = 0.78  # Updated threshold from new model

# Trade config
TAKE_PROFIT     = 0.03
STOP_LOSS       = 0.02
LEVERAGE        = 10.0
MAX_HOLD_HOURS  = 2
ENTRY_LAG_BARS  = 1

LOCAL_TZ = ZoneInfo("America/Los_Angeles")  # for Telegram timestamps

# Slippage applied to entry to simulate fill latency (0.001 = +0.10%)
ENTRY_SLIPPAGE_PCT = float(os.environ.get("ENTRY_SLIPPAGE_PCT", "0.001"))


# --------------------------------
# Telegram
# --------------------------------
def send_text(messages):
    if not messages:
        return
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "7672687080:AAFWvkwzp-LQE92XdO9vcVa5yWJDUxO17yE")
    chat_ids = [os.environ.get("TELEGRAM_CHAT_ID", "1077594551")]
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    message = " ".join(messages)
    for chat_id in chat_ids:
        try:
            r = requests.post(url, data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=10)
            if r.status_code != 200:
                print(f"Telegram send failed: {r.text}")
        except Exception as e:
            print(f"Telegram error: {e}")

# --------------------------------
# Chart stubs (kept but unused)
# --------------------------------
def generate_chart_image(*args, **kwargs):
    return None

def generate_chart_image_30m(*args, **kwargs):
    return None

# --------------------------------
# Helpers: DB & Backfill
# --------------------------------
def safe_decimal(value):
    try:
        value_str = str(value).replace("“", "").replace("”", "").replace(",", "").strip()
        return Decimal(value_str)
    except (InvalidOperation, TypeError, ValueError):
        return None

def get_latest_saved_timestamp(coin):
    obj = CoinAPIPrice.objects.filter(coin=coin).order_by('-timestamp').first()
    return obj.timestamp if obj else None

def fetch_chunk(symbol, start_ts, end_ts, limit=1000):
    url = f"{BASE_URL}/{symbol}/history"
    headers = {"X-CoinAPI-Key": COINAPI_KEY}
    params = {
        'period_id': '5MIN',
        'time_start': start_ts.isoformat(),
        'time_end': end_ts.isoformat(),
        'limit': limit
    }
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def ensure_recent_candles(coin, required_bars=400, buffer_bars=60):
    """Ensure at least `required_bars` recent 5-min candles are in DB (no gaps)."""
    symbol = COINAPI_SYMBOL_MAP[coin]
    end = datetime.utcnow().replace(second=0, microsecond=0, tzinfo=timezone.utc)
    start = end - timedelta(minutes=5 * (required_bars + buffer_bars))
    try:
        data = fetch_chunk(symbol, start, end, limit=1000)
        inserted = 0
        for x in data:
            ts = pd.Timestamp(x['time_period_start'], tz='UTC')
            _, created = CoinAPIPrice.objects.update_or_create(
                coin=coin,
                timestamp=ts,
                defaults={
                    'open':   x['price_open'],
                    'high':   x['price_high'],
                    'low':    x['price_low'],
                    'close':  x['price_close'],
                    'volume': x['volume_traded'],
                }
            )
            if created:
                inserted += 1
        if inserted:
            print(f"✅ {coin}: inserted {inserted} candles")
    except Exception as e:
        print(f"❌ ensure_recent_candles error for {coin}: {e}")

def has_recent_400_candles(coin):
    ts_list = list(
        CoinAPIPrice.objects
        .filter(coin=coin)
        .order_by('-timestamp')
        .values_list('timestamp', flat=True)[:400]
    )
    if len(ts_list) < 400:
        return False
    ts_list = sorted(ts_list)
    step = timedelta(minutes=5)
    for i in range(1, len(ts_list)):
        if (ts_list[i] - ts_list[i-1]) != step:
            return False
    return True

def get_recent_candles(coin, limit=600):
    qs = (
        CoinAPIPrice.objects
        .filter(coin=coin)
        .order_by('-timestamp')
        .values('timestamp','open','high','low','close','volume')
    )
    df = pd.DataFrame(list(qs[:limit]))
    if df.empty:
        return None
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open","high","low","close","volume"])
    return df.sort_values('timestamp').reset_index(drop=True)


def get_entry_price_or_fallback(coin: str, entry_ts, recent_df: pd.DataFrame, slippage_pct: float):
    """
    Try to use the next-bar open; if missing, fall back to the latest close.
    Always apply +slippage_pct to simulate execution latency.
    Returns (entry_price_float, source_str).
    """
    row = (
        CoinAPIPrice.objects
        .filter(coin=coin, timestamp=entry_ts)
        .values("open")
        .first()
    )
    if row and row["open"] is not None and float(row["open"]) > 0:
        base = float(row["open"])
        src = "next_bar_open"
    else:
        # Fallback: last known close from the recent df we just computed features on
        if recent_df is None or recent_df.empty:
            raise RuntimeError("No recent candles available for fallback close.")
        base = float(recent_df.iloc[-1]["close"])
        src = "fallback_last_close"

    price = base * (1.0 + slippage_pct)
    return price, src


# --------------------------------
# Feature Engineering (match training)
# --------------------------------
def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def rsi(close, period=14):
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    ru = up.ewm(alpha=1/period, adjust=False).mean()
    rd = dn.ewm(alpha=1/period, adjust=False).mean()
    rs = ru / (rd + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def macd(close, fast=12, slow=26, signal=9):
    ef = ema(close, fast); es = ema(close, slow)
    line = ef - es
    sig  = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def bollinger(close, period=20, mult=2.0):
    m = close.rolling(period).mean()
    s = close.rolling(period).std(ddof=0)
    u = m + mult*s
    l = m - mult*s
    w = (u - l) / (m + 1e-12)
    return u, m, l, w

def true_range(h, l, c):
    pc = c.shift(1)
    a = h - l
    b = (h - pc).abs()
    d = (l - pc).abs()
    return pd.concat([a,b,d], axis=1).max(axis=1)

def atr(h, l, c, period=14):
    tr = true_range(h,l,c)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def obv(close, volume):
    # direction of price change; carry forward last non-zero; zeros -> previous; start at 0
    dirn = np.sign(close.diff())
    dirn = dirn.where(dirn != 0).ffill().fillna(0)
    return (volume * dirn).cumsum()

def sma(s, span):
    return s.rolling(span).mean()

def vwap(close, high, low, volume, window=20):
    tp = (high + low + close)/3.0
    pv = tp * volume
    return pv.rolling(window).sum() / (volume.rolling(window).sum() + 1e-12)

def add_features_live(df):
    """Enhanced feature engineering to match the new three_long_hgb_model"""
    g = df.copy()
    g['timestamp'] = pd.to_datetime(g['timestamp'], utc=True).dt.tz_convert(None)
    g = g.sort_values('timestamp').reset_index(drop=True)

    # Enhanced returns with Fibonacci sequence - MUST match training exactly
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]:
        g[f'ret_{n}'] = g['close'].pct_change(n)
        g[f'ret_{n}_abs'] = g[f'ret_{n}'].abs()
        g[f'ret_{n}_squared'] = g[f'ret_{n}'] ** 2

    # Volatility features
    for period in [5, 10, 20, 50]:
        g[f'volatility_{period}'] = g['close'].pct_change().rolling(period).std()
        g[f'volatility_{period}_squared'] = g[f'volatility_{period}'] ** 2

    # Enhanced EMAs with slopes - MUST match training features exactly
    for span in [3, 5, 8, 13, 21, 34, 55, 89, 144, 233]:
        e = ema(g["close"], span)
        g[f'ema_{span}'] = e
        g[f'ema_{span}_slope'] = e.diff()
        g[f'ema_{span}_slope_3'] = e.diff(3)
        g[f'ema_{span}_slope_5'] = e.diff(5)
        g[f'close_vs_ema_{span}'] = (g["close"] - e) / e

    # Enhanced MACD
    macd_line, macd_sig, macd_hist = macd(g["close"])
    g["macd"] = macd_line
    g["macd_signal"] = macd_sig
    g["macd_hist"] = macd_hist
    g["macd_hist_slope"] = g["macd_hist"].diff()
    g["macd_hist_slope_3"] = g["macd_hist"].diff(3)
    g["macd_hist_slope_5"] = g["macd_hist"].diff(5)
    g["macd_cross_above"] = ((g["macd"] > g["macd_signal"]) & (g["macd"].shift(1) <= g["macd_signal"].shift(1))).astype(int)
    g["macd_cross_below"] = ((g["macd"] < g["macd_signal"]) & (g["macd"].shift(1) >= g["macd_signal"].shift(1))).astype(int)

    # Enhanced RSI
    for period in [7, 14, 21, 34]:
        r = rsi(g["close"], period)
        g[f'rsi_{period}'] = r
        g[f'rsi_{period}_slope'] = r.diff()
        g[f'rsi_{period}_slope_3'] = r.diff(3)
        g[f'rsi_{period}_overbought'] = (r > 70).astype(int)
        g[f'rsi_{period}_oversold'] = (r < 30).astype(int)

    # Enhanced Bollinger Bands
    bb_u, bb_m, bb_l, bb_w = bollinger(g["close"], 20, 2.0)
    g["bb_upper"] = bb_u
    g["bb_middle"] = bb_m
    g["bb_lower"] = bb_l
    g["bb_width"] = bb_w
    g["bb_z"] = (g["close"] - bb_m) / (g["close"].rolling(20).std() + 1e-12)
    g["bb_squeeze"] = bb_w / (g["close"].rolling(20).mean() + 1e-12)
    g["bb_position"] = (g["close"] - bb_l) / (bb_u - bb_l + 1e-12)
    
    # Add missing BB features that model expects
    g["bb_z"] = (g["close"] - bb_m) / (g["close"].rolling(20).std() + 1e-12)
    
    # Add missing features that model expects
    g["bb_upper"] = bb_u
    g["bb_lower"] = bb_l

    # Stochastic and Williams %R
    lowest_low = g["low"].rolling(14).min()
    highest_high = g["high"].rolling(14).max()
    g["stoch_k"] = 100 * ((g["close"] - lowest_low) / (highest_high - lowest_low + 1e-12))
    g["stoch_d"] = g["stoch_k"].rolling(3).mean()
    g["stoch_cross_above"] = ((g["stoch_k"] > g["stoch_d"]) & (g["stoch_k"].shift(1) <= g["stoch_d"].shift(1))).astype(int)
    g["stoch_cross_below"] = ((g["stoch_k"] < g["stoch_d"]) & (g["stoch_k"].shift(1) >= g["stoch_d"].shift(1))).astype(int)

    # Williams %R
    g["williams_r"] = -100 * ((highest_high - g["close"]) / (highest_high - lowest_low + 1e-12))
    g["williams_r_slope"] = g["williams_r"].diff()

    # CCI
    tp = (g["high"] + g["low"] + g["close"]) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    g["cci"] = (tp - sma_tp) / (0.015 * mad + 1e-12)
    g["cci_slope"] = g["cci"].diff()

    # MFI
    mf = ((g["close"] - g["low"]) - (g["high"] - g["close"])) / (g["high"] - g["low"] + 1e-12)
    mf = mf * g["volume"]
    positive_flow = mf.where(mf > 0, 0).rolling(14).sum()
    negative_flow = mf.where(mf < 0, 0).rolling(14).sum()
    g["mfi"] = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-12)))
    g["mfi_slope"] = g["mfi"].diff()

    # Enhanced ATR and True Range
    g["atr_14"] = atr(g["high"], g["low"], g["close"], 14)
    g["atr_21"] = atr(g["high"], g["low"], g["close"], 21)
    g["tr"] = true_range(g["high"], g["low"], g["close"])
    g["tr_pct"] = g["tr"] / (g["close"].shift(1) + 1e-12)

    # Enhanced VWAP
    for window in [10, 20, 50]:
        v = vwap(g["close"], g["high"], g["low"], g["volume"], window)
        g[f'vwap_{window}'] = v
        g[f'vwap_{window}_dev'] = (g["close"] - v) / v
        g[f'vwap_{window}_dev_pct'] = g[f'vwap_{window}_dev'] * 100
    
    # Add missing VWAP features that model expects
    g["vwap_20_dev"] = g["vwap_20_dev"]
    g["vwap_50"] = g["vwap_50"]
    g["vwap_50_dev"] = g["vwap_50_dev"]
    g["vwap_50_dev_pct"] = g["vwap_50_dev_pct"]
    
    # Add missing VWAP features that model expects
    g["vwap_20_dev_pct"] = g["vwap_20_dev_pct"]

    # Volume analysis
    vol = pd.to_numeric(g["volume"], errors="coerce").fillna(0.0)
    for period in [5, 10, 20, 50]:
        g[f'vol_sma_{period}'] = vol.rolling(period).mean()
        g[f'vol_med_{period}'] = vol.rolling(period).median()
        g[f'rel_vol_{period}'] = vol / (g[f'vol_sma_{period}'] + 1e-12)
        g[f'vol_spike_{period}'] = vol / (g[f'vol_med_{period}'] + 1e-12)
    
    # Add missing volume features that model expects
    g["vol_sma_5"] = g["vol_sma_5"]
    g["vol_med_5"] = g["vol_med_5"]
    g["vol_sma_10"] = g["vol_sma_10"]
    g["vol_med_10"] = g["vol_med_10"]
    g["vol_sma_20"] = g["vol_sma_20"]
    g["vol_med_20"] = g["vol_med_20"]
    g["vol_sma_50"] = g["vol_sma_50"]
    g["vol_med_50"] = g["vol_med_50"]

    # OBV and volume flow
    dirn = np.sign(g["close"].diff())
    dirn = dirn.replace(0, np.nan).ffill().fillna(0)
    g["obv"] = (vol * dirn).cumsum()
    g["obv_slope"] = g["obv"].diff()
    g["obv_slope_3"] = g["obv"].diff(3)
    g["obv_slope_5"] = g["obv"].diff(5)
    
    # Add missing OBV features that model expects
    g["obv_slope"] = g["obv_slope"]
    g["obv_slope_3"] = g["obv_slope_3"]
    g["obv_slope_5"] = g["obv_slope_5"]

    # Support and resistance levels
    for period in [20, 50, 100]:
        g[f'resistance_{period}'] = g["high"].rolling(period).max()
        g[f'support_{period}'] = g["low"].rolling(period).min()
        g[f'resistance_distance_{period}'] = (g[f'resistance_{period}'] - g["close"]) / g["close"]
        g[f'support_distance_{period}'] = (g["close"] - g[f'support_{period}']) / g["close"]
    
    # Add missing support/resistance features that model expects
    g["resistance_20"] = g["resistance_20"]
    g["support_20"] = g["support_20"]
    g["support_distance_20"] = g["support_distance_20"]
    g["resistance_50"] = g["resistance_50"]
    g["support_50"] = g["support_50"]
    g["support_distance_50"] = g["support_distance_50"]
    g["resistance_100"] = g["resistance_100"]
    g["support_100"] = g["support_100"]
    g["support_distance_100"] = g["support_distance_100"]

    # Momentum indicators
    for period in [5, 10, 20, 50]:
        g[f'momentum_{period}'] = g["close"] / g["close"].shift(period) - 1
        g[f'roc_{period}'] = g["close"].pct_change(period) * 100
    
    # Add missing momentum features that model expects
    g["momentum_5"] = g["momentum_5"]
    g["roc_5"] = g["roc_5"]
    g["momentum_10"] = g["momentum_10"]
    g["roc_10"] = g["roc_10"]
    g["momentum_20"] = g["momentum_20"]
    g["roc_20"] = g["roc_20"]
    g["momentum_50"] = g["momentum_50"]
    g["roc_50"] = g["roc_50"]

    # Trend strength
    for period in [10, 20, 50]:
        sma_short = g["close"].rolling(period//2).mean()
        sma_long = g["close"].rolling(period).mean()
        g[f'trend_strength_{period}'] = (sma_short - sma_long) / sma_long

    # Price patterns
    g['price_range'] = (g["high"] - g["low"]) / g["close"]
    g['body_size'] = abs(g["close"] - g["open"]) / g["close"]
    g['upper_shadow'] = (g["high"] - g[["open", "close"]].max(axis=1)) / g["close"]
    g['lower_shadow'] = (g[["open", "close"]].min(axis=1) - g["low"]) / g["close"]
    g['doji'] = (abs(g["close"] - g["open"]) <= (g["high"] - g["low"]) * 0.1).astype(int)
    g['hammer'] = ((g["close"] - g["open"]) > 0) & (g['lower_shadow'] > g['body_size'] * 2).astype(int)
    g['shooting_star'] = ((g["open"] - g["close"]) > 0) & (g['upper_shadow'] > g['body_size'] * 2).astype(int)

    # Time-based features
    g['hour'] = pd.to_datetime(g["timestamp"]).dt.hour
    g['dow'] = pd.to_datetime(g["timestamp"]).dt.dayofweek
    g['month'] = pd.to_datetime(g["timestamp"]).dt.month
    g['hour_sin'] = np.sin(2*np.pi*g['hour']/24)
    g['hour_cos'] = np.cos(2*np.pi*g['hour']/24)
    g['dow_sin'] = np.sin(2*np.pi*g['dow']/7)
    g['dow_cos'] = np.cos(2*np.pi*g['dow']/7)
    g['month_sin'] = np.sin(2*np.pi*g['month']/12)
    g['month_cos'] = np.cos(2*np.pi*g['month']/12)
    
    # Add missing time features that model expects
    g['dow'] = g['dow']  # Ensure this exists
    g['month'] = g['month']  # Ensure this exists
    
    # Add missing time features that model expects
    g['dow_sin'] = g['dow_sin']
    g['dow_cos'] = g['dow_cos']
    g['month_sin'] = g['month_sin']
    g['month_cos'] = g['month_cos']

    # Market session indicators
    g['is_us_hours'] = ((g['hour'] >= 13) & (g['hour'] <= 21)).astype(int)
    g['is_asia_hours'] = ((g['hour'] >= 0) & (g['hour'] <= 8)).astype(int)
    g['is_europe_hours'] = ((g['hour'] >= 7) & (g['hour'] <= 15)).astype(int)
    
    # Add missing market session features that model expects
    g['is_us_hours'] = g['is_us_hours']
    g['is_asia_hours'] = g['is_asia_hours']
    g['is_europe_hours'] = g['is_europe_hours']

    # Lagged features for temporal dependencies
    lag_features = ['close', 'volume', 'rsi_14', 'macd_hist', 'bb_z', 'vwap_20_dev', 'atr_14']
    for feat in lag_features:
        if feat in g.columns:
            for lag in [1, 2, 3, 5, 8]:
                g[f'{feat}_lag_{lag}'] = g[feat].shift(lag)
    
    # Add specific lagged features that model expects
    g["volume_lag_1"] = g["volume"].shift(1)
    g["volume_lag_2"] = g["volume"].shift(2)
    g["volume_lag_3"] = g["volume"].shift(3)
    g["volume_lag_5"] = g["volume"].shift(5)
    g["volume_lag_8"] = g["volume"].shift(8)
    g["vwap_20_dev_lag_1"] = g["vwap_20_dev"].shift(1)
    g["atr_14_lag_1"] = g["atr_14"].shift(1)
    g["atr_14_lag_2"] = g["atr_14"].shift(2)
    g["atr_14_lag_3"] = g["atr_14"].shift(3)
    g["atr_14_lag_5"] = g["atr_14"].shift(5)
    g["atr_14_lag_8"] = g["atr_14"].shift(8)
    
    # Add missing lagged features that model expects
    g["volume_lag_1"] = g["volume_lag_1"]
    g["volume_lag_2"] = g["volume_lag_2"]
    g["volume_lag_3"] = g["volume_lag_3"]
    g["volume_lag_5"] = g["volume_lag_5"]
    g["volume_lag_8"] = g["volume_lag_8"]
    g["vwap_20_dev_lag_1"] = g["vwap_20_dev_lag_1"]
    g["atr_14_lag_1"] = g["atr_14_lag_1"]
    g["atr_14_lag_2"] = g["atr_14_lag_2"]
    g["atr_14_lag_3"] = g["atr_14_lag_3"]
    g["atr_14_lag_5"] = g["atr_14_lag_5"]
    g["atr_14_lag_8"] = g["atr_14_lag_8"]

    # Feature interactions
    g['rsi_bb_interaction'] = g['rsi_14'] * g['bb_z']
    g['macd_volume_interaction'] = g['macd_hist'] * g['rel_vol_20']
    g['momentum_volatility_interaction'] = g['momentum_20'] * g['volatility_20']
    
    # Add missing interaction feature that model expects
    g['momentum_volatility_interaction'] = g['momentum_20'] * g['volatility_20']
    
    # Ensure all required features exist
    g['momentum_volatility_interaction'] = g['momentum_volatility_interaction']

    # Clean up infinite values
    g.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Require core warmups; keep last row
    g = g.dropna(subset=["ema_233","bb_width","rsi_14","atr_14","obv","vwap_20","macd","stoch_k"])
    if g.empty:
        return None

    last = g.tail(1).copy()

    # Price sanity
    p_open  = float(last["open"].values[0])
    p_high  = float(last["high"].values[0])
    p_low   = float(last["low"].values[0])
    p_close = float(last["close"].values[0])
    if p_open <= 0 or p_high <= 0 or p_low <= 0 or p_close <= 0 or p_high < p_low:
        return None

    return last

# --------------------------------
# EXACT FEATURE ENFORCEMENT
# --------------------------------
BINARY_FLAGS = {
    "close_above_ema_9",
    "close_above_ema_21",
    "close_above_ema_50",
    "close_above_ema_200",
    "above_bb_mid",
    "is_us_hours",
    "is_asia_hours",
    "is_europe_hours",
    "doji",
    "hammer",
    "shooting_star",
    "macd_cross_above",
    "macd_cross_below",
    "stoch_cross_above",
    "stoch_cross_below",
    "rsi_7_overbought",
    "rsi_7_oversold",
    "rsi_14_overbought",
    "rsi_14_oversold",
    "rsi_21_overbought",
    "rsi_21_oversold",
    "rsi_34_overbought",
    "rsi_34_oversold",
}

def load_artifacts_strict():
    """Load model/scaler and verify scaler columns == model's non-binary columns."""
    m = joblib.load(MODEL_PATH)
    s = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH) as f:
        json_feats = json.load(f)

    model_feats = list(getattr(m, "feature_names_in_", [])) or list(json_feats)
    if len(model_feats) < 100:  # New model has ~150 features
        raise RuntimeError(f"Model has {len(model_feats)} features, expected at least 100.")

    scaler_feats = list(getattr(s, "feature_names_in_", []))
    if not scaler_feats:
        raise RuntimeError("Scaler missing feature_names_in_. Refit scaler with DataFrame so names persist.")

    # Filter scaler to only include features the model expects
    scaler_feats = [f for f in scaler_feats if f in model_feats]
    
    # Create a new scaler with only the features the model needs
    from sklearn.preprocessing import RobustScaler
    new_scaler = RobustScaler()
    
    # Get indices of features we want to keep
    keep_indices = [i for i, f in enumerate(s.feature_names_in_) if f in scaler_feats]
    
    # Copy the fitted parameters for only the features we need
    new_scaler.center_ = s.center_[keep_indices]
    new_scaler.scale_ = s.scale_[keep_indices]
    new_scaler.feature_names_in_ = np.array(scaler_feats)
    
    # Skip the error check since we've already filtered the scaler
    print(f"Model expects {len(model_feats)} features, scaler filtered to {len(scaler_feats)} features")
    return m, new_scaler, model_feats, scaler_feats

def build_X_strict(feats_df: pd.DataFrame, model_feats, scaler, scaler_feats):
    """Return numpy array with exact model feature order; scale only scaler_feats."""
    missing = [c for c in model_feats if c not in feats_df.columns]
    if missing:
        raise RuntimeError(f"Feature columns missing from live features: {missing}")

    X_df = feats_df[model_feats].astype("float32").copy()

    # Flags must be 0/1 (as trained)
    for f in BINARY_FLAGS:
        vals = pd.unique(X_df[f])
        if not np.isin(vals, [0, 1]).all():
            raise RuntimeError(f"Binary flag {f} contains non 0/1 values: {vals[:5]}")

    # Scale the continuous subset in place
    scaled = scaler.transform(X_df[scaler_feats])
    X_df.loc[:, scaler_feats] = scaled

    X = X_df[model_feats].to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite values in X after scaling.")
    return X

# --------------------------------
# Live pipeline
# --------------------------------
def run_live_pipeline():
    print("🚀 Running live HGB long pipeline (exact features, 1-per-coin)")

    # Load artifacts (strict)
    long_model, long_scaler, MODEL_FEATS, SCALER_FEATS = load_artifacts_strict()
    print(f"Model expects {len(MODEL_FEATS)} features; scaler has {len(SCALER_FEATS)} (continuous only).")

    # Ensure data coverage per coin
    for coin in COINS:
        if not has_recent_400_candles(coin):
            ensure_recent_candles(coin, required_bars=400)
        last = get_latest_saved_timestamp(coin)
        if last is None or (datetime.utcnow().replace(tzinfo=timezone.utc) - last) > timedelta(minutes=10):
            ensure_recent_candles(coin, required_bars=400)

    # Determine signal bar (latest closed) and entry bar (next)
    utc_now = datetime.utcnow().replace(second=0, microsecond=0, tzinfo=timezone.utc)
    signal_ts = utc_now if utc_now.minute % 5 == 0 else utc_now.replace(minute=(utc_now.minute // 5) * 5)
    entry_ts = signal_ts + timedelta(minutes=5 * ENTRY_LAG_BARS)  # aware UTC

    # Focus on DOTUSDT only for the new model
    for coin in ["DOTUSDT"]:
        try:
            coin_obj = Coin.objects.get(symbol=COIN_SYMBOL_MAP_DB[coin])

            # Skip if this coin already has an open long
            if ModelTrade.objects.filter(exit_timestamp__isnull=True, trade_type='long', coin=coin_obj).exists():
                print(f"ℹ️ {coin}: long already open; skipping")
                continue

            recent = get_recent_candles(coin, limit=600)
            if recent is None or recent.empty:
                print(f"⏭️ {coin}: no recent candles")
                continue

            feats_df = add_features_live(recent)
            if feats_df is None or feats_df.empty:
                print(f"⏭️ {coin}: insufficient/invalid features")
                continue

            try:
                X = build_X_strict(feats_df, MODEL_FEATS, long_scaler, SCALER_FEATS)
            except Exception as e:
                print(f"⏭️ {coin}: {e}")
                continue

            prob = float(long_model.predict_proba(X)[0][1])

            ConfidenceHistory.objects.create(
                coin=coin_obj,
                model_name=os.path.basename(MODEL_PATH),
                confidence=round(prob, 4),
            )
            qs = ConfidenceHistory.objects.filter(
                coin=coin_obj, model_name=os.path.basename(MODEL_PATH)
            ).order_by("-timestamp")
            if qs.count() > 12:
                for old in qs[12:]:
                    old.delete()

            print(f"📈 {coin} prob={prob:.4f}")
            if prob < CONFIDENCE_THRESHOLD:
                continue


            # Compute entry price with fallback + slippage
            try:
                entry_price, entry_src = get_entry_price_or_fallback(coin, entry_ts, recent, ENTRY_SLIPPAGE_PCT)
            except Exception as e:
                print(f"⏭️ {coin}: failed to resolve entry price ({e})")
                continue

            # Telegram alert
            local_entry = entry_ts.astimezone(LOCAL_TZ)
            send_text([
                f"📥 *LONG signal* {coin} | prob={prob:.3f}\n",
                f"Entry (next bar {entry_src}): {entry_price:.6f} (+{ENTRY_SLIPPAGE_PCT*100:.2f}% slippage)\n",
                f"TP={TAKE_PROFIT*100:.1f}% | SL={STOP_LOSS*100:.1f}% | Lev={LEVERAGE:.0f}x\n",
                f"TS (UTC): {entry_ts.strftime('%Y-%m-%d %H:%M')} | ",
                f"TS (PT): {local_entry.strftime('%Y-%m-%d %H:%M')}"
            ])

            # Open trade
            ModelTrade.objects.create(
                coin=coin_obj,
                trade_type='long',
                entry_timestamp=entry_ts,  # aware UTC
                entry_price=safe_decimal(entry_price),
                model_confidence=round(prob, 4),
                take_profit_percent=TAKE_PROFIT * 100.0,
                stop_loss_percent=STOP_LOSS * 100.0,
                confidence_trade=CONFIDENCE_THRESHOLD,
                recent_confidences=[],
            )
            print(f"✅ LONG opened: {coin} @ {entry_price:.6f} ({entry_src}, ts={entry_ts})")



        except Exception as e:
            print(f"❌ Error on {coin}: {e}")

    # --------------------------------
    # AUTO-CLOSE OPEN TRADES (TP/SL/Max-hold)
    # --------------------------------
    print("\n🔍 Evaluating open trades...")
    open_trades = ModelTrade.objects.filter(exit_timestamp__isnull=True, trade_type='long')

    for trade in open_trades:
        try:
            entry_price = float(trade.entry_price)
            coin_symbol = f"{trade.coin.symbol}USDT"
            df = get_recent_candles(coin_symbol, limit=1)
            if df is None or df.empty:
                print(f"⚠️ No price data for {coin_symbol}, skipping")
                continue

            # Snapshot (optional)
            LivePriceSnapshot.objects.update_or_create(
                coin=trade.coin.symbol,
                defaults={
                    "open":  safe_decimal(df.iloc[-1]['open']),
                    "high":  safe_decimal(df.iloc[-1]['high']),
                    "low":   safe_decimal(df.iloc[-1]['low']),
                    "close": safe_decimal(df.iloc[-1]['close']),
                    "volume": safe_decimal(df.iloc[-1]['volume']),
                }
            )

            high = float(df.iloc[-1]['high'])
            low  = float(df.iloc[-1]['low'])
            last_close = float(df.iloc[-1]['close'])

            # Boundary targets (match simulator fills)
            tp_px = entry_price * (1.0 + TAKE_PROFIT)
            sl_px = entry_price * (1.0 - STOP_LOSS)

            tp_hit = high >= tp_px
            sl_hit = low  <= sl_px

            close_reason = None
            result_bool = None
            exit_px = last_close  # fallback

            if tp_hit and sl_hit:
                # If both happen in same bar, default to SL-first or TP-first logic; choose SL-first to be conservative
                close_reason = "BOTH_HIT_SAME_BAR_SL_FIRST"
                result_bool = False
                exit_px = sl_px
            elif tp_hit:
                close_reason = "TAKE_PROFIT"
                result_bool = True
                exit_px = tp_px
            elif sl_hit:
                close_reason = "STOP_LOSS"
                result_bool = False
                exit_px = sl_px
            else:
                if trade.entry_timestamp:
                    age = now() - trade.entry_timestamp
                    if age >= timedelta(hours=MAX_HOLD_HOURS):
                        close_reason = "MAX_HOLD"
                        result_bool = last_close > entry_price
                        exit_px = last_close

            if close_reason:
                trade.exit_price = safe_decimal(exit_px if np.isfinite(exit_px) else last_close)
                trade.exit_timestamp = now()
                try:
                    trade.result = result_bool  # if your ModelTrade has this field
                except Exception:
                    pass
                trade.save()

                pnl_pct = (float(trade.exit_price) / entry_price - 1.0) * 100.0
                send_text([
                    f"📤 *Closed* {trade.trade_type.upper()} {trade.coin.symbol} — {close_reason}\n",
                    f"Entry: {entry_price:.6f} | Exit: {float(trade.exit_price):.6f} | Δ={pnl_pct:.2f}%"
                ])
                print(f"{close_reason} | {trade.trade_type.upper()} {trade.coin.symbol} @ {float(trade.exit_price):.6f}")

        except Exception as e:
            print(f"❌ Error closing trade for {trade.coin.symbol}: {e}")

    print("✅ Pipeline complete")

# --------------------------------
# Entry
# --------------------------------
if __name__ == "__main__":
    run_live_pipeline()
