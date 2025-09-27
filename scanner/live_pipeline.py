# scanner/live_pipeline.py
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
        mod = types.ModuleType(f"numpy.random._" + name.lower())
        setattr(mod, name, cls)
        sys.modules[mod.__name__] = mod
_patch_numpy_rng_compat()

# --------------------------------
# Config
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

# All long models using simple long dataset approach
MODELS = {
    "ADAUSDT": {
        "model": "ada_simple_long_rf_model.joblib",
        "scaler": "ada_simple_long_scaler.joblib", 
        "features": "ada_simple_long_features.json",
        "threshold": 0.6,
    },
    "ATOMUSDT": {
        "model": "atom_simple_long_rf_model.joblib",
        "scaler": "atom_simple_long_scaler.joblib", 
        "features": "atom_simple_long_features.json",
        "threshold": 0.65,
    },
    "AVAXUSDT": {
        "model": "avax_simple_long_rf_model.joblib",
        "scaler": "avax_simple_long_scaler.joblib", 
        "features": "avax_simple_long_features.json",
        "threshold": 0.6,
    },
    "BTCUSDT": {
        "model": "btc_simple_long_rf_model.joblib",
        "scaler": "btc_simple_long_scaler.joblib", 
        "features": "btc_simple_long_features.json",
        "threshold": 0.5,
    },
    "DOGEUSDT": {
        "model": "doge_simple_long_rf_model.joblib",
        "scaler": "doge_simple_long_scaler.joblib", 
        "features": "doge_simple_long_features.json",
        "threshold": 0.55,
    },
    "DOTUSDT": {
        "model": "dot_simple_long_rf_model.joblib",
        "scaler": "dot_simple_long_scaler.joblib", 
        "features": "dot_simple_long_features.json",
        "threshold": 0.65,
    },
    "ETHUSDT": {
        "model": "eth_simple_long_rf_model.joblib",
        "scaler": "eth_simple_long_scaler.joblib", 
        "features": "eth_simple_long_features.json",
        "threshold": 0.6,
    },
    "LINKUSDT": {
        "model": "link_simple_long_rf_model.joblib",
        "scaler": "link_simple_long_scaler.joblib", 
        "features": "link_simple_long_features.json",
        "threshold": 0.55,
    },
    "LTCUSDT": {
        "model": "ltc_simple_long_rf_model.joblib",
        "scaler": "ltc_simple_long_scaler.joblib", 
        "features": "ltc_simple_long_features.json",
        "threshold": 0.55,
    },
    "SHIBUSDT": {
        "model": "shib_simple_long_rf_model.joblib",
        "scaler": "shib_simple_long_scaler.joblib", 
        "features": "shib_simple_long_features.json",
        "threshold": 0.65,
    },
    "SOLUSDT": {
        "model": "sol_simple_long_rf_model.joblib",
        "scaler": "sol_simple_long_scaler.joblib", 
        "features": "sol_simple_long_features.json",
        "threshold": 0.6,
    },
    "TRXUSDT": {
        "model": "trx_simple_long_rf_model.joblib",
        "scaler": "trx_simple_long_scaler.joblib", 
        "features": "trx_simple_long_features.json",
        "threshold": 0.75,
    },
    "UNIUSDT": {
        "model": "uni_simple_long_rf_model.joblib",
        "scaler": "uni_simple_long_scaler.joblib", 
        "features": "uni_simple_long_features.json",
        "threshold": 0.6,
    },
    "XLMUSDT": {
        "model": "xlm_simple_long_rf_model.joblib",
        "scaler": "xlm_simple_long_scaler.joblib", 
        "features": "xlm_simple_long_features.json",
        "threshold": 0.65,
    },
    "XRPUSDT": {
        "model": "xrp_simple_long_rf_model.joblib",
        "scaler": "xrp_simple_long_scaler.joblib", 
        "features": "xrp_simple_long_features.json",
        "threshold": 0.55,
    },
}

# All models now use new simplified approach from dot_dataset.py script
# No more old model constants needed

# --------------------------------
# DUAL MODEL CONFIGURATION
# --------------------------------
# All short models using simple short dataset approach
SHORT_MODELS = {
    "ADAUSDT": {
        "model": "ada_simple_short_rf_model.joblib",
        "scaler": "ada_simple_short_scaler.joblib", 
        "features": "ada_simple_short_features.json",
        "threshold": 0.55,
    },
    "ATOMUSDT": {
        "model": "atom_simple_short_rf_model.joblib",
        "scaler": "atom_simple_short_scaler.joblib", 
        "features": "atom_simple_short_features.json",
        "threshold": 0.5,
    },
    "AVAXUSDT": {
        "model": "avax_simple_short_rf_model.joblib",
        "scaler": "avax_simple_short_scaler.joblib", 
        "features": "avax_simple_short_features.json",
        "threshold": 0.5,
    },
    "BTCUSDT": {
        "model": "btc_simple_short_rf_model.joblib",
        "scaler": "btc_simple_short_scaler.joblib", 
        "features": "btc_simple_short_features.json",
        "threshold": 0.5,
    },
    "DOGEUSDT": {
        "model": "doge_simple_short_rf_model.joblib",
        "scaler": "doge_simple_short_scaler.joblib", 
        "features": "doge_simple_short_features.json",
        "threshold": 0.5,
    },
    "DOTUSDT": {
        "model": "dot_simple_short_rf_model.joblib",
        "scaler": "dot_simple_short_scaler.joblib", 
        "features": "dot_simple_short_features.json",
        "threshold": 0.5,
    },
    "ETHUSDT": {
        "model": "eth_simple_short_rf_model.joblib",
        "scaler": "eth_simple_short_scaler.joblib", 
        "features": "eth_simple_short_features.json",
        "threshold": 0.5,
    },
    "LINKUSDT": {
        "model": "link_simple_short_rf_model.joblib",
        "scaler": "link_simple_short_scaler.joblib", 
        "features": "link_simple_short_features.json",
        "threshold": 0.5,
    },
    "LTCUSDT": {
        "model": "ltc_simple_short_rf_model.joblib",
        "scaler": "ltc_simple_short_scaler.joblib", 
        "features": "ltc_simple_short_features.json",
        "threshold": 0.5,
    },
    "SHIBUSDT": {
        "model": "shib_simple_short_rf_model.joblib",
        "scaler": "shib_simple_short_scaler.joblib", 
        "features": "shib_simple_short_features.json",
        "threshold": 0.5,
    },
    "SOLUSDT": {
        "model": "sol_simple_short_rf_model.joblib",
        "scaler": "sol_simple_short_scaler.joblib", 
        "features": "sol_simple_short_features.json",
        "threshold": 0.5,
    },
    "TRXUSDT": {
        "model": "trx_simple_short_rf_model.joblib",
        "scaler": "trx_simple_short_scaler.joblib", 
        "features": "trx_simple_short_features.json",
        "threshold": 0.5,
    },
    "UNIUSDT": {
        "model": "uni_simple_short_rf_model.joblib",
        "scaler": "uni_simple_short_scaler.joblib", 
        "features": "uni_simple_short_features.json",
        "threshold": 0.5,
    },
    "XLMUSDT": {
        "model": "xlm_simple_short_rf_model.joblib",
        "scaler": "xlm_simple_short_scaler.joblib", 
        "features": "xlm_simple_short_features.json",
        "threshold": 0.6,
    },
    "XRPUSDT": {
        "model": "xrp_simple_short_rf_model.joblib",
        "scaler": "xrp_simple_short_scaler.joblib", 
        "features": "xrp_simple_short_features.json",
        "threshold": 0.6,
    },
}

# Dual model thresholds (optimized from our testing)
DUAL_LONG_THRESHOLD = 0.8   # Higher threshold for long model
DUAL_SHORT_THRESHOLD = 0.6  # Lower threshold for short model


COINAPI_KEY = os.environ.get("COINAPI_KEY", "01293e2a-dcf1-4e81-8310-c6aa9d0cb743")
BASE_URL = "https://rest.coinapi.io/v1/ohlcv"

# Trade config
TAKE_PROFIT     = 0.02  # +2%
STOP_LOSS       = 0.01  # -1%
LEVERAGE        = 15.0   # 15x leverage
MAX_HOLD_HOURS  = None   # No max hold constraint
ENTRY_LAG_BARS  = 1
ENTRY_SLIPPAGE_PCT = float(os.environ.get("ENTRY_SLIPPAGE_PCT", "0.001"))
LOCAL_TZ = ZoneInfo("America/Los_Angeles")

# --------------------------------
# Telegram
# --------------------------------
def send_text(lines):
    if not lines:
        return
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "7672687080:AAFWvkwzp-LQE92XdO9vcVa5yWJDUxO17yE")
    chat_ids = [os.environ.get("TELEGRAM_CHAT_ID", "1077594551")]
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    text = "\n".join(lines)
    for chat_id in chat_ids:
        try:
            r = requests.post(
                url,
                data={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
                timeout=10
            )
            if r.status_code != 200:
                print(f"Telegram send failed: {r.text}")
        except Exception as e:
            print(f"Telegram error: {e}")

# --------------------------------
# DB helpers
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
    row = (
        CoinAPIPrice.objects
        .filter(coin=coin, timestamp=entry_ts)
        .values("open")
        .first()
    )
    if row and row["open"] is not None and float(row["open"]) > 0:
        base = float(row["open"]); src = "next_bar_open"
    else:
        if recent_df is None or recent_df.empty:
            raise RuntimeError("No recent candles available for fallback close.")
        base = float(recent_df.iloc[-1]["close"]); src = "fallback_last_close"
    price = base * (1.0 + slippage_pct)
    return price, src

# --------------------------------
# Feature engineering (MUST match training)
# --------------------------------
def ema(s, span): return s.ewm(span=span, adjust=False).mean()
def sma(s, span): return s.rolling(span).mean()

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
    line = ef - es; sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def bollinger(close, period=20, mult=2.0):
    m = close.rolling(period).mean()
    s = close.rolling(period).std(ddof=0)
    u = m + mult*s; l = m - mult*s
    w = (u - l) / (m + 1e-12)
    return u, m, l, w, s

def true_range(h, l, c):
    pc = c.shift(1)
    a = (h - l); b = (h - pc).abs(); d = (l - pc).abs()
    return pd.concat([a,b,d], axis=1).max(axis=1)

def atr(h, l, c, period=14):
    tr = true_range(h,l,c)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def vwap(close, high, low, volume, window=20):
    tp = (high + low + close)/3.0
    pv = tp * volume
    return pv.rolling(window).sum() / (volume.rolling(window).sum() + 1e-12)

def williams_r(high, low, close, period=14):
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    return -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-12))

def cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma_tp) / (0.015 * mad + 1e-12)

def money_flow_index(high, low, close, volume, period=14):
    mf = ((close - low) - (high - close)) / (high - low + 1e-12)
    mf = mf * volume
    positive_flow = mf.where(mf > 0, 0).rolling(period).sum()
    negative_flow = mf.where(mf < 0, 0).rolling(period).sum()
    return 100 - (100 / (1 + positive_flow / (negative_flow + 1e-12)))

# --------------------------------
# Complex Feature engineering (NO LONGER USED - replaced by simplified approach)
# --------------------------------
def add_features_live(df):
    g = df.copy()
    g['timestamp'] = pd.to_datetime(g['timestamp'], utc=True)
    g = g.sort_values('timestamp').reset_index(drop=True)

    F = {}
    # Price action
    F['price_range'] = (g['high'] - g['low']) / g['close']
    F['body_size'] = (g['close'] - g['open']).abs() / g['close']
    F['upper_shadow'] = (g['high'] - g[['open', 'close']].max(axis=1)) / g['close']
    F['lower_shadow'] = (g[['open', 'close']].min(axis=1) - g['low']) / g['close']

    # Returns
    for n in [1,2,3,5,8,13,21,34,55,89]:
        r = g['close'].pct_change(n)
        F[f'ret_{n}'] = r
        F[f'ret_{n}_abs'] = r.abs()
        F[f'ret_{n}_squared'] = r**2

    # Volatility
    for period in [5,10,20,50]:
        v = g['close'].pct_change().rolling(period).std()
        F[f'volatility_{period}'] = v
        F[f'volatility_{period}_squared'] = v**2

    # EMAs
    for span in [3,5,8,13,21,34,55,89,144,233]:
        e = ema(g['close'], span)
        F[f'ema_{span}'] = e
        F[f'ema_{span}_slope'] = e.diff()
        F[f'ema_{span}_slope_3'] = e.diff(3)
        F[f'ema_{span}_slope_5'] = e.diff(5)
        F[f'close_vs_ema_{span}'] = (g['close'] - e) / (e + 1e-12)

    # MACD
    macd_line, macd_sig, macd_hist = macd(g['close'])
    F['macd'] = macd_line
    F['macd_signal'] = macd_sig
    F['macd_hist'] = macd_hist
    F['macd_hist_slope'] = macd_hist.diff()
    F['macd_hist_slope_3'] = macd_hist.diff(3)
    F['macd_hist_slope_5'] = macd_hist.diff(5)

    # RSI
    for period in [7,14,21,34]:
        r = rsi(g['close'], period)
        F[f'rsi_{period}'] = r
        F[f'rsi_{period}_slope'] = r.diff()
        F[f'rsi_{period}_slope_3'] = r.diff(3)
        F[f'rsi_{period}_overbought'] = (r > 70).astype(int)
        F[f'rsi_{period}_oversold'] = (r < 30).astype(int)

    # Bollinger
    bb_u, bb_m, bb_l, bb_w, bb_std = bollinger(g['close'], 20, 2.0)
    F['bb_upper'] = bb_u; F['bb_middle'] = bb_m; F['bb_lower'] = bb_l
    F['bb_width'] = bb_w
    F['bb_z'] = (g['close'] - bb_m) / (bb_std + 1e-12)
    F['bb_squeeze'] = bb_w / (g['close'].rolling(20).mean() + 1e-12)
    F['bb_position'] = (g['close'] - bb_l) / (bb_u - bb_l + 1e-12)

    # Stochastic
    lowest_low = g['low'].rolling(14).min()
    highest_high = g['high'].rolling(14).max()
    stoch_k = 100 * ((g['close'] - lowest_low) / (highest_high - lowest_low + 1e-12))
    stoch_d = stoch_k.rolling(3).mean()
    F['stoch_k'] = stoch_k; F['stoch_d'] = stoch_d

    # Williams %R + slope
    wr = williams_r(g['high'], g['low'], g['close'])
    F['williams_r'] = wr; F['williams_r_slope'] = wr.diff()

    # CCI + slope
    cci_val = cci(g['high'], g['low'], g['close'])
    F['cci'] = cci_val; F['cci_slope'] = cci_val.diff()

    # MFI + slope
    mfi_val = money_flow_index(g['high'], g['low'], g['close'], g['volume'])
    F['mfi'] = mfi_val; F['mfi_slope'] = mfi_val.diff()

    # ATR/TR
    atr14 = atr(g['high'], g['low'], g['close'], 14)
    atr21 = atr(g['high'], g['low'], g['close'], 21)
    tr = true_range(g['high'], g['low'], g['close'])
    F['atr_14'] = atr14; F['atr_21'] = atr21
    F['tr'] = tr; F['tr_pct'] = tr / (g['close'].shift(1) + 1e-12)

    # VWAP 10/20/50
    for window in [10,20,50]:
        v = vwap(g['close'], g['high'], g['low'], g['volume'], window)
        F[f'vwap_{window}'] = v
        dev = (g['close'] - v) / (v + 1e-12)
        F[f'vwap_{window}_dev'] = dev
        F[f'vwap_{window}_dev_pct'] = dev * 100.0

    # Volume
    vol = pd.to_numeric(g['volume'], errors='coerce').fillna(0.0)
    for period in [5,10,20,50]:
        sma_vol = vol.rolling(period).mean()
        med_vol = vol.rolling(period).median()
        F[f'vol_sma_{period}'] = sma_vol
        F[f'vol_med_{period}'] = med_vol
        F[f'rel_vol_{period}'] = vol / (sma_vol + 1e-12)
        F[f'vol_spike_{period}'] = vol / (med_vol + 1e-12)

    # OBV + slopes
    dirn = np.sign(g['close'].diff())
    dirn = dirn.replace(0, np.nan).ffill().fillna(0)
    obv_val = (vol * dirn).cumsum()
    F['obv'] = obv_val
    F['obv_slope'] = obv_val.diff()
    F['obv_slope_3'] = obv_val.diff(3)
    F['obv_slope_5'] = obv_val.diff(5)

    # Support/Resistance (+ distances)
    for period in [20,50,100]:
        res = g['high'].rolling(period).max()
        sup = g['low'].rolling(period).min()
        F[f'resistance_{period}'] = res
        F[f'support_{period}'] = sup
        F[f'resistance_distance_{period}'] = (res - g['close']) / (g['close'] + 1e-12)
        F[f'support_distance_{period}'] = (g['close'] - sup) / (g['close'] + 1e-12)

    # Momentum & ROC
    for period in [5,10,20,50]:
        F[f'momentum_{period}'] = g['close'] / g['close'].shift(period) - 1
        F[f'roc_{period}'] = g['close'].pct_change(period) * 100.0

    # Trend strength
    for period in [10,20,50]:
        sma_short = sma(g['close'], period//2)
        sma_long = sma(g['close'], period)
        F[f'trend_strength_{period}'] = (sma_short - sma_long) / (sma_long + 1e-12)

    # Candles
    F['doji'] = ((g['close'] - g['open']).abs() <= (g['high'] - g['low']) * 0.1).astype(int)
    F['hammer'] = (((g['close'] - g['open']) > 0) & (( (g[['open','close']].min(axis=1) - g['low']) / g['close']) > ((g['close'] - g['open']).abs() / g['close']) * 2)).astype(int)
    F['shooting_star'] = (((g['open'] - g['close']) > 0) & (((g['high'] - g[['open','close']].max(axis=1)) / g['close']) > ((g['close'] - g['open']).abs() / g['close']) * 2)).astype(int)

    # Time features (UTC)
    hour = g['timestamp'].dt.hour
    dow = g['timestamp'].dt.dayofweek
    month = g['timestamp'].dt.month
    F['hour'] = hour; F['dow'] = dow; F['month'] = month
    F['hour_sin'] = np.sin(2*np.pi*hour/24);   F['hour_cos'] = np.cos(2*np.pi*hour/24)
    F['dow_sin']  = np.sin(2*np.pi*dow/7);     F['dow_cos']  = np.cos(2*np.pi*dow/7)
    F['month_sin']= np.sin(2*np.pi*month/12);  F['month_cos']= np.cos(2*np.pi*month/12)

    # Sessions
    F['is_us_hours'] = ((hour >= 13) & (hour <= 21)).astype(int)
    F['is_asia_hours'] = ((hour >= 0) & (hour <= 8)).astype(int)
    F['is_europe_hours'] = ((hour >= 7) & (hour <= 15)).astype(int)

    feat_df = pd.DataFrame(F, index=g.index)

    # Crosses
    feat_df['macd_cross_above'] = ((feat_df['macd'] > feat_df['macd_signal']) &
                                   (feat_df['macd'].shift(1) <= feat_df['macd_signal'].shift(1))).astype(int)
    feat_df['macd_cross_below'] = ((feat_df['macd'] < feat_df['macd_signal']) &
                                   (feat_df['macd'].shift(1) >= feat_df['macd_signal'].shift(1))).astype(int)
    feat_df['stoch_cross_above'] = ((feat_df['stoch_k'] > feat_df['stoch_d']) &
                                    (feat_df['stoch_k'].shift(1) <= feat_df['stoch_d'].shift(1))).astype(int)
    feat_df['stoch_cross_below'] = ((feat_df['stoch_k'] < feat_df['stoch_d']) &
                                    (feat_df['stoch_k'].shift(1) >= feat_df['stoch_d'].shift(1))).astype(int)

    # Lags
    lag_sources = {
        'close': g['close'],
        'volume': g['volume'],
        'rsi_14': feat_df['rsi_14'],
        'macd_hist': feat_df['macd_hist'],
        'bb_z': feat_df['bb_z'],
        'vwap_20_dev': feat_df['vwap_20_dev'],
        'atr_14': feat_df['atr_14'],
    }
    for name, s in lag_sources.items():
        for lag in [1,2,3,5,8]:
            feat_df[f'{name}_lag_{lag}'] = s.shift(lag)

    # Interactions
    feat_df['rsi_bb_interaction'] = feat_df['rsi_14'] * feat_df['bb_z']
    feat_df['macd_volume_interaction'] = feat_df['macd_hist'] * feat_df['rel_vol_20']
    feat_df['momentum_volatility_interaction'] = feat_df['momentum_20'] * feat_df['volatility_20']

    g = pd.concat([g, feat_df], axis=1)
    g.replace([np.inf, -np.inf], np.nan, inplace=True)

    core_cols = ["ema_233","bb_width","rsi_14","atr_14","obv","vwap_20","macd","stoch_k"]
    g = g.dropna(subset=core_cols)
    if g.empty:
        return None
    return g.tail(1).copy()

# --------------------------------
# Simplified Feature engineering for DOTUSDT (20 features only)
# --------------------------------
def add_features_simplified_dot(df):
    """
    Simplified feature engineering for DOTUSDT - matches the new 20-feature model
    This is completely separate from the existing complex feature engineering
    """
    g = df.copy()
    g['timestamp'] = pd.to_datetime(g['timestamp'], utc=True)
    g = g.sort_values('timestamp').reset_index(drop=True)

    F = {}
    
    # Core Price Action (3 features)
    F['price_range'] = (g['high'] - g['low']) / g['close']
    F['body_size'] = (g['close'] - g['open']).abs() / g['close']
    F['close_position'] = (g['close'] - g['low']) / (g['high'] - g['low'] + 1e-12)
    
    # Core Returns (3 features)
    F['ret_1'] = g['close'].pct_change(1)
    F['ret_5'] = g['close'].pct_change(5)
    F['ret_20'] = g['close'].pct_change(20)
    
    # Core Volatility (2 features)
    F['volatility_20'] = g['close'].pct_change().rolling(20).std()
    F['volatility_50'] = g['close'].pct_change().rolling(50).std()
    
    # Core EMAs (3 features)
    F['ema_20'] = g['close'].ewm(span=20).mean()
    F['ema_50'] = g['close'].ewm(span=50).mean()
    F['ema_200'] = g['close'].ewm(span=200).mean()
    F['close_vs_ema_20'] = (g['close'] - F['ema_20']) / F['ema_20']
    F['close_vs_ema_50'] = (g['close'] - F['ema_50']) / F['ema_50']
    F['ema_trend'] = (F['ema_20'] - F['ema_50']) / F['ema_50']
    
    # Core MACD (2 features)
    F['macd'] = g['close'].ewm(span=12).mean() - g['close'].ewm(span=26).mean()
    F['macd_signal'] = F['macd'].ewm(span=9).mean()
    F['macd_hist'] = F['macd'] - F['macd_signal']
    
    # Core RSI (1 feature)
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-12)
        return 100 - (100 / (1 + rs))
    
    F['rsi_14'] = rsi(g['close'], 14)
    
    # Core Bollinger Bands (2 features)
    bb_20 = g['close'].rolling(20).mean()
    bb_std = g['close'].rolling(20).std()
    F['bb_upper'] = bb_20 + (bb_std * 2)
    F['bb_lower'] = bb_20 - (bb_std * 2)
    F['bb_position'] = (g['close'] - bb_20) / (bb_std + 1e-12)
    F['bb_width'] = (F['bb_upper'] - F['bb_lower']) / bb_20
    
    # Core Volume (3 features)
    vol = pd.to_numeric(g['volume'], errors='coerce').fillna(0.0)
    F['vol_sma_20'] = vol.rolling(20).mean()
    F['rel_vol'] = vol / F['vol_sma_20']
    F['vol_spike'] = vol / vol.rolling(50).median()
    
    # Core ATR (1 feature)
    F['atr_14'] = ((g['high'] - g['low']).rolling(14).mean() + 
                    (g['high'] - g['close'].shift(1)).abs().rolling(14).mean() + 
                    (g['low'] - g['close'].shift(1)).abs().rolling(14).mean()) / 3
    
    # Core Support/Resistance (2 features)
    F['resistance_20'] = g['high'].rolling(20).max()
    F['support_20'] = g['low'].rolling(20).min()
    F['resistance_distance'] = (F['resistance_20'] - g['close']) / g['close']
    F['support_distance'] = (g['close'] - F['support_20']) / g['close']
    
    # Core Momentum (2 features)
    F['momentum_20'] = g['close'] / g['close'].shift(20) - 1
    F['momentum_50'] = g['close'] / g['close'].shift(50) - 1
    
    # Core Time Features (2 features)
    hour = g['timestamp'].dt.hour
    F['hour_sin'] = np.sin(2*np.pi*hour/24)
    F['is_us_hours'] = ((hour >= 13) & (hour <= 21)).astype(int)
    
    # Core Crosses (2 features)
    F['ema_cross'] = ((F['ema_20'] > F['ema_50']) & (F['ema_20'].shift(1) <= F['ema_50'].shift(1))).astype(int)
    F['macd_cross'] = ((F['macd'] > F['macd_signal']) & (F['macd'].shift(1) <= F['macd_signal'].shift(1))).astype(int)
    
    feat_df = pd.DataFrame(F, index=g.index)
    g = pd.concat([g, feat_df], axis=1)
    g.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Only keep essential columns for prediction (matches training)
    core_cols = ["ema_20", "ema_50", "rsi_14", "bb_position", "vol_sma_20"]
    g = g.dropna(subset=core_cols)
    if g.empty:
        return None
    
    # Ensure proper column names for the model
    result = g.tail(1).copy()
    result.columns = result.columns.astype(str)  # Ensure all column names are strings
    return result

# --------------------------------
# Artifact loading (bundle preferred; otherwise strict set-match)
# --------------------------------
def _infer_config_path(model_path: str, features_path: str) -> str:
    base = os.path.basename(features_path)
    if base.endswith("_feature_list.json"):
        return features_path.replace("_feature_list.json", "_trade_config.json")
    mbase = os.path.basename(model_path)
    if mbase.endswith("_long_hgb_model.joblib"):
        prefix = mbase[:-len("_long_hgb_model.joblib")]
        return os.path.join(os.path.dirname(model_path), f"{prefix}_trade_config.json")
    return os.path.join(os.path.dirname(model_path), "trade_config.json")

def _load_threshold_from_config(config_path: str, fallback: float) -> float:
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        thr = cfg.get("threshold", None)
        if thr is None:
            return float(fallback)
        return float(thr)
    except Exception:
        return float(fallback)

def load_artifacts_bundle(path: str, fallback_thr: float):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not {"model","scaler","features"}.issubset(bundle.keys()):
        raise RuntimeError(f"{os.path.basename(path)} is not a valid bundle (missing keys).")
    m = bundle["model"]; s = bundle["scaler"]; feats = list(bundle["features"])
    thr = float(bundle.get("threshold", fallback_thr))

    model_feats  = list(getattr(m, "feature_names_in_", []))
    scaler_feats = list(getattr(s, "feature_names_in_", []))
    if not model_feats or not scaler_feats:
        raise RuntimeError(f"{os.path.basename(path)} bundle has model/scaler missing feature_names_in_. Re-export.")

    if set(model_feats) != set(scaler_feats):
        only_m = list(set(model_feats) - set(scaler_feats))[:10]
        only_s = list(set(scaler_feats) - set(model_feats))[:10]
        raise RuntimeError(
            f"Bundle model↔scaler feature SET mismatch in {os.path.basename(path)}."
            f"\n  model-only:  {only_m}\n  scaler-only: {only_s}"
        )

    # Use scaler order for transform (safe), but require it to be identical to model order in training runs.
    if model_feats != scaler_feats:
        print("⚠️  Bundle warning: model and scaler feature ORDER differ. "
              "Training should export them with identical order. Using scaler order at runtime.")

    # JSON advisory only
    if feats and set(feats) != set(model_feats):
        print("⚠️  Bundle warning: stored features list differs by set. Using scaler/model set at runtime.")

    print(f"[ARTIFACTS] {os.path.basename(path)} | n_features={len(scaler_feats)} | thr={thr:.3f}")
    return m, s, list(scaler_feats), thr  # return order = scaler order

def load_artifacts_separate(model_path, scaler_path, features_path, fallback_thr: float):
    m = joblib.load(model_path)
    s = joblib.load(scaler_path)
    try:
        with open(features_path) as f:
            json_feats = list(json.load(f))
    except Exception:
        json_feats = []

    model_feats  = list(getattr(m, "feature_names_in_", []))
    scaler_feats = list(getattr(s, "feature_names_in_", []))
    if not model_feats:
        raise RuntimeError(f"{os.path.basename(model_path)} missing feature_names_in_. Re-export by fitting on a DataFrame.")
    if not scaler_feats:
        raise RuntimeError(f"{os.path.basename(scaler_path)} missing feature_names_in_. Re-export scaler fit on a DataFrame.")

    if set(model_feats) != set(scaler_feats):
        only_model  = list(set(model_feats) - set(scaler_feats))[:10]
        only_scaler = list(set(scaler_feats) - set(model_feats))[:10]
        raise RuntimeError(
            "Model↔Scaler feature SET mismatch."
            f"\n  model file:  {os.path.basename(model_path)}"
            f"\n  scaler file: {os.path.basename(scaler_path)}"
            f"\n  in model only (first 10):  {only_model}"
            f"\n  in scaler only (first 10): {only_scaler}"
            "\n-> Fix: re-run training export for this coin so model & scaler are produced together."
        )

    if json_feats and set(json_feats) != set(model_feats):
        miss = list(set(model_feats) - set(json_feats))[:10]
        extra = list(set(json_feats) - set(model_feats))[:10]
        print(
            "⚠️  JSON feature list differs from model/scaler set."
            f"\n  features json: {os.path.basename(features_path)}"
            f"\n  missing in json (first 10): {miss}"
            f"\n  extra in json (first 10):   {extra}"
            "\n  (Using scaler/model features at runtime.)"
        )

    cfg_path = _infer_config_path(model_path, features_path)
    thr = _load_threshold_from_config(cfg_path, fallback_thr)

    print(f"[ARTIFACTS] {os.path.basename(model_path)} | n_features={len(scaler_feats)} | thr={thr:.3f} | model/scaler sets match")
    return m, s, list(scaler_feats), thr  # order = scaler order

# --------------------------------
# STRICT FEATURE BUILD & SCALE (using scaler order)
# --------------------------------
def build_X_exact(feats_df: pd.DataFrame, scaler_feature_order: list, scaler):
    missing = [c for c in scaler_feature_order if c not in feats_df.columns]
    if missing:
        more = f" ... (+{len(missing)-50} more)" if len(missing) > 50 else ""
        raise RuntimeError(f"Feature columns missing from live features: {missing[:50]}{more}")

    X_df = feats_df[scaler_feature_order].astype("float32")

    if X_df.isna().any().any():
        bad = X_df.columns[X_df.isna().any()].tolist()
        more = f" ... (+{len(bad)-50} more)" if len(bad) > 50 else ""
        raise RuntimeError(f"NaNs present in features: {bad[:50]}{more}")

    if hasattr(scaler, "feature_names_in_"):
        names = list(getattr(scaler, "feature_names_in_", []))
        if names and names != scaler_feature_order:
            raise RuntimeError("Scaler feature order mismatch vs provided order (internal bug).")

    X_scaled = scaler.transform(X_df)
    if not np.isfinite(X_scaled).all():
        raise RuntimeError("Non-finite values after scaling.")
    return X_scaled

def drift_report(feats_row: pd.DataFrame, scaler, top=8, prefix=""):
    try:
        cols = list(getattr(scaler, "feature_names_in_", []))
        if not cols or not hasattr(scaler, "center_") or not hasattr(scaler, "scale_"):
            return
        x = feats_row[cols].astype(np.float32).values.reshape(1, -1)
        z = (x - scaler.center_) / (scaler.scale_ + 1e-12)
        pairs = sorted(
            [(cols[i], float(abs(z[0, i])), float(z[0, i])) for i in range(len(cols))],
            key=lambda t: t[1],
            reverse=True
        )[:top]
        print(f"{prefix} drift top-|z|:")
        for name, abz, valz in pairs:
            print(f"   {name:32s}  |z|={abz:7.3f}  z={valz:+7.3f}")
    except Exception as e:
        print(f"{prefix} drift report failed: {e}")

# --------------------------------
# LONG MODEL FEATURE ENGINEERING (Simple Long Dataset)
# --------------------------------
def add_features_long_model(df):
    """
    Feature engineering for long models - matches the simple long dataset approach
    """
    g = df.copy()
    g['timestamp'] = pd.to_datetime(g['timestamp'], utc=True)
    g = g.sort_values('timestamp').reset_index(drop=True)

    F = {}
    
    # 1. TREND FEATURES (4 features)
    F['ema_20'] = g['close'].ewm(span=20).mean()
    F['ema_50'] = g['close'].ewm(span=50).mean()
    F['ema_200'] = g['close'].ewm(span=200).mean()
    F['trend_strength'] = (F['ema_20'] - F['ema_200']) / F['ema_200']
    
    # 2. MOMENTUM FEATURES (3 features)
    F['ret_5'] = g['close'].pct_change(5)
    F['ret_20'] = g['close'].pct_change(20)
    F['momentum'] = g['close'] / g['close'].shift(20) - 1
    
    # 3. VOLATILITY FEATURES (2 features)
    F['atr_14'] = ((g['high'] - g['low']).rolling(14).mean() + 
                    (g['high'] - g['close'].shift(1)).abs().rolling(14).mean() + 
                    (g['low'] - g['close'].shift(1)).abs().rolling(14).mean()) / 3
    F['volatility'] = g['close'].pct_change().rolling(20).std()
    
    # 4. RSI FEATURES (2 features)
    F['rsi_14'] = rsi(g['close'], 14)
    F['rsi_oversold'] = (F['rsi_14'] < 30).astype(int)  # Bullish signal
    
    # 5. BOLLINGER BANDS (2 features)
    bb_20 = g['close'].rolling(20).mean()
    bb_std = g['close'].rolling(20).std()
    F['bb_position'] = (g['close'] - bb_20) / (bb_std + 1e-12)
    F['bb_squeeze'] = (bb_std < bb_std.rolling(50).quantile(0.25)).astype(int)
    
    # 6. VOLUME FEATURES (2 features)
    vol = pd.to_numeric(g['volume'], errors='coerce').fillna(0.0)
    F['vol_sma_20'] = vol.rolling(20).mean()
    F['rel_vol'] = vol / F['vol_sma_20']
    
    # 7. LONG-SPECIFIC SIGNALS (3 features)
    F['price_above_ema_20'] = (g['close'] > F['ema_20']).astype(int)
    F['ema_trend_up'] = (F['ema_20'] > F['ema_50']).astype(int)
    F['bullish_momentum'] = ((F['ret_5'] > 0) & (F['ret_20'] > 0)).astype(int)
    
    feat_df = pd.DataFrame(F, index=g.index)
    g = pd.concat([g, feat_df], axis=1)
    g.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Core columns for prediction
    core_cols = ["ema_20", "ema_50", "rsi_14", "bb_position", "vol_sma_20"]
    g = g.dropna(subset=core_cols)
    if g.empty:
        return None
    
    # Ensure proper column names for the model
    result = g.tail(1).copy()
    result.columns = result.columns.astype(str)
    return result

# --------------------------------
# SHORT MODEL FEATURE ENGINEERING
# --------------------------------
def add_features_short_model(df):
    """
    Feature engineering for short models - matches the simple short dataset approach
    """
    g = df.copy()
    g['timestamp'] = pd.to_datetime(g['timestamp'], utc=True)
    g = g.sort_values('timestamp').reset_index(drop=True)

    F = {}
    
    # 1. TREND FEATURES (4 features)
    F['ema_20'] = g['close'].ewm(span=20).mean()
    F['ema_50'] = g['close'].ewm(span=50).mean()
    F['ema_200'] = g['close'].ewm(span=200).mean()
    F['trend_strength'] = (F['ema_20'] - F['ema_200']) / F['ema_200']
    
    # 2. MOMENTUM FEATURES (3 features)
    F['ret_5'] = g['close'].pct_change(5)
    F['ret_20'] = g['close'].pct_change(20)
    F['momentum'] = g['close'] / g['close'].shift(20) - 1
    
    # 3. VOLATILITY FEATURES (2 features)
    F['atr_14'] = ((g['high'] - g['low']).rolling(14).mean() + 
                    (g['high'] - g['close'].shift(1)).abs().rolling(14).mean() + 
                    (g['low'] - g['close'].shift(1)).abs().rolling(14).mean()) / 3
    F['volatility'] = g['close'].pct_change().rolling(20).std()
    
    # 4. RSI FEATURES (2 features)
    F['rsi_14'] = rsi(g['close'], 14)
    F['rsi_overbought'] = (F['rsi_14'] > 70).astype(int)  # Bearish signal
    
    # 5. BOLLINGER BANDS (2 features)
    bb_20 = g['close'].rolling(20).mean()
    bb_std = g['close'].rolling(20).std()
    F['bb_position'] = (g['close'] - bb_20) / (bb_std + 1e-12)
    F['bb_squeeze'] = (bb_std < bb_std.rolling(50).quantile(0.25)).astype(int)
    
    # 6. VOLUME FEATURES (2 features)
    vol = pd.to_numeric(g['volume'], errors='coerce').fillna(0.0)
    F['vol_sma_20'] = vol.rolling(20).mean()
    F['rel_vol'] = vol / F['vol_sma_20']
    
    # 7. SHORT-SPECIFIC SIGNALS (3 features)
    F['price_below_ema_20'] = (g['close'] < F['ema_20']).astype(int)
    F['ema_trend_down'] = (F['ema_20'] < F['ema_50']).astype(int)
    F['bearish_momentum'] = ((F['ret_5'] < 0) & (F['ret_20'] < 0)).astype(int)
    
    feat_df = pd.DataFrame(F, index=g.index)
    g = pd.concat([g, feat_df], axis=1)
    g.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Core columns for prediction
    core_cols = ["ema_20", "ema_50", "rsi_14", "bb_position", "vol_sma_20"]
    g = g.dropna(subset=core_cols)
    if g.empty:
        return None
    
    # Ensure proper column names for the model
    result = g.tail(1).copy()
    result.columns = result.columns.astype(str)
    return result

# --------------------------------
# DUAL MODEL FUNCTIONS
# --------------------------------
def load_dual_models():
    """Load both long and short models for dual model approach"""
    loaded_models = {}
    
    # Load long models (existing)
    for coin, cfg in MODELS.items():
        try:
            if "bundle" in cfg and os.path.exists(cfg["bundle"]):
                m, s, feat_order, thr = load_artifacts_bundle(cfg["bundle"], cfg.get("threshold", 0.6))
            else:
                m, s, feat_order, thr = load_artifacts_separate(
                    cfg["model"], cfg["scaler"], cfg["features"], cfg.get("threshold", 0.6)
                )
            loaded_models[coin] = {
                "long_model": m, 
                "long_scaler": s, 
                "long_order": feat_order, 
                "long_thr": float(thr)
            }
        except Exception as e:
            print(f"❌ Error loading long model for {coin}: {e}")
            continue
    
    # Load short models (new)
    for coin, cfg in SHORT_MODELS.items():
        try:
            if "bundle" in cfg and os.path.exists(cfg["bundle"]):
                m, s, feat_order, thr = load_artifacts_bundle(cfg["bundle"], cfg.get("threshold", 0.6))
            else:
                m, s, feat_order, thr = load_artifacts_separate(
                    cfg["model"], cfg["scaler"], cfg["features"], cfg.get("threshold", 0.6)
                )
            
            if coin not in loaded_models:
                loaded_models[coin] = {}
            
            loaded_models[coin].update({
                "short_model": m,
                "short_scaler": s, 
                "short_order": feat_order,
                "short_thr": float(thr)
            })
        except Exception as e:
            print(f"❌ Error loading short model for {coin}: {e}")
            continue
    
    return loaded_models

def get_dual_model_prediction(coin, recent_df, loaded_models):
    """Get predictions from both long and short models"""
    if coin not in loaded_models:
        return None, None, None, None
    
    coin_models = loaded_models[coin]
    
    # Long model prediction (use long model features)
    long_prob = None
    long_confidence = None
    if "long_model" in coin_models:
        try:
            feats_df_long = add_features_long_model(recent_df)
            if feats_df_long is not None and not feats_df_long.empty:
                X_long = build_X_exact(feats_df_long, coin_models["long_order"], coin_models["long_scaler"])
                model = coin_models["long_model"]
                if hasattr(model, "predict_proba"):
                    long_prob = float(model.predict_proba(X_long)[0][1])
                else:
                    raw = float(getattr(model, "decision_function", lambda x: model.predict(x))(X_long))
                    long_prob = raw if 0.0 <= raw <= 1.0 else 1.0 / (1.0 + np.exp(-raw))
                long_confidence = long_prob
        except Exception as e:
            print(f"❌ Long model prediction error for {coin}: {e}")
    
    # Short model prediction (use short model features)
    short_prob = None
    short_confidence = None
    if "short_model" in coin_models:
        try:
            feats_df_short = add_features_short_model(recent_df)
            if feats_df_short is not None and not feats_df_short.empty:
                X_short = build_X_exact(feats_df_short, coin_models["short_order"], coin_models["short_scaler"])
                model = coin_models["short_model"]
                if hasattr(model, "predict_proba"):
                    short_prob = float(model.predict_proba(X_short)[0][1])
                else:
                    raw = float(getattr(model, "decision_function", lambda x: model.predict(x))(X_short))
                    short_prob = raw if 0.0 <= raw <= 1.0 else 1.0 / (1.0 + np.exp(-raw))
                short_confidence = short_prob
        except Exception as e:
            print(f"❌ Short model prediction error for {coin}: {e}")
    
    return long_prob, long_confidence, short_prob, short_confidence

def determine_dual_model_signal(long_prob, short_prob, long_thr, short_thr):
    """Determine trading signal based on dual model predictions"""
    if long_prob is None and short_prob is None:
        return None, None, None
    
    # Long signal: long model confident and short model not confident
    if long_prob is not None and long_prob >= long_thr:
        if short_prob is None or short_prob < short_thr:
            return 'long', long_prob, 'long_model'
    
    # Short signal: short model confident and long model not confident  
    if short_prob is not None and short_prob >= short_thr:
        if long_prob is None or long_prob < long_thr:
            return 'short', short_prob, 'short_model'
    
    # No signal if both are confident or both are not confident
    return None, None, None

# --------------------------------
# Live pipeline
# --------------------------------
def run_dual_model_pipeline():
    """New dual model pipeline with both long and short trading"""
    print("🚀 Running DUAL MODEL pipeline (long + short models)")
    
    # Load dual models
    loaded_models = load_dual_models()
    if not loaded_models:
        print("❌ No dual models loaded. Check artifacts.")
        return
    
    print(f"✅ Loaded {len(loaded_models)} coins with dual models")
    
    # Ensure data coverage for all coins with dual models
    print("📊 Ensuring recent data for all coins...")
    for coin in loaded_models.keys():
        if not has_recent_400_candles(coin):
            print(f"📥 Fetching recent data for {coin}")
            ensure_recent_candles(coin, required_bars=400)
        last = get_latest_saved_timestamp(coin)
        if last is None or (datetime.utcnow().replace(tzinfo=timezone.utc) - last) > timedelta(minutes=10):
            print(f"📥 Updating stale data for {coin}")
            ensure_recent_candles(coin, required_bars=400)
    
    # Determine signal bar (latest closed) and entry bar (next)
    utc_now = datetime.utcnow().replace(second=0, microsecond=0, tzinfo=timezone.utc)
    signal_ts = utc_now if utc_now.minute % 5 == 0 else utc_now.replace(minute=(utc_now.minute // 5) * 5)
    entry_ts = signal_ts + timedelta(minutes=5 * ENTRY_LAG_BARS)
    
    # Iterate coins with dual models
    for coin, coin_models in loaded_models.items():
        try:
            coin_obj = Coin.objects.get(symbol=COIN_SYMBOL_MAP_DB[coin])
            
            # Skip if this coin already has an open trade (long or short)
            if ModelTrade.objects.filter(exit_timestamp__isnull=True, coin=coin_obj).exists():
                print(f"ℹ️ {coin}: trade already open; skipping")
                continue
            
            recent = get_recent_candles(coin, limit=600)
            if recent is None or recent.empty:
                print(f"⏭️ {coin}: no recent candles")
                continue
            
            # Get dual model predictions (each model uses its own feature engineering)
            print(f"🔧 {coin}: Getting dual model predictions")
            long_prob, long_conf, short_prob, short_conf = get_dual_model_prediction(coin, recent, loaded_models)
            
            # Get individual coin thresholds
            long_thr = coin_models.get("long_thr", 0.6)
            short_thr = coin_models.get("short_thr", 0.5)
            
            # Always print confidence scores for monitoring
            long_str = f"{long_prob:.4f}" if long_prob is not None else "None"
            short_str = f"{short_prob:.4f}" if short_prob is not None else "None"
            print(f"📊 {coin}: Long={long_str} (thr={long_thr:.2f}) | Short={short_str} (thr={short_thr:.2f})")
            
            if long_prob is None and short_prob is None:
                print(f"⏭️ {coin}: no model predictions available")
                continue
            
            # Determine signal using individual coin thresholds
            signal_type, confidence, model_type = determine_dual_model_signal(
                long_prob, short_prob, long_thr, short_thr
            )
            
            if signal_type is None:
                print(f"⏭️ {coin}: no signal (both models below threshold)")
                continue
            
            print(f"📈 {coin} {signal_type.upper()} signal: {model_type} conf={confidence:.4f}")
            
            # Entry price (next bar)
            try:
                entry_price, entry_src = get_entry_price_or_fallback(coin, entry_ts, recent, ENTRY_SLIPPAGE_PCT)
            except Exception as e:
                print(f"⏭️ {coin}: failed to resolve entry price ({e})")
                continue
            
            # Telegram alert
            local_entry = entry_ts.astimezone(LOCAL_TZ)
            send_text([
                f"📥 {signal_type.upper()} {coin}  {model_type} conf={confidence:.3f}",
                f"Entry ({entry_src}): {entry_price:.6f}  (+{ENTRY_SLIPPAGE_PCT*100:.2f}% slippage)",
                f"TP={TAKE_PROFIT*100:.1f}%  SL={STOP_LOSS*100:.1f}%  Lev={LEVERAGE:.0f}x",
                f"UTC: {entry_ts.strftime('%Y-%m-%d %H:%M')} | PT: {local_entry.strftime('%Y-%m-%d %H:%M')}"
            ])
            
            # Open trade
            ModelTrade.objects.create(
                coin=coin_obj,
                trade_type=signal_type,  # 'long' or 'short'
                entry_timestamp=entry_ts,
                entry_price=safe_decimal(entry_price),
                model_confidence=round(confidence, 4),
                take_profit_percent=TAKE_PROFIT * 100.0,
                stop_loss_percent=STOP_LOSS * 100.0,
                confidence_trade=long_thr if signal_type == 'long' else short_thr,
                recent_confidences=[],
                model_name=model_type,
            )
            print(f"✅ {signal_type.upper()} opened: {coin} @ {entry_price:.6f} ({entry_src}, ts={entry_ts})")
            
        except Exception as e:
            print(f"❌ Error on {coin}: {e}")
    
    # Auto-close open trades (both long and short)
    print("\n🔍 Evaluating open trades...")
    open_trades = ModelTrade.objects.filter(exit_timestamp__isnull=True)
    
    for trade in open_trades:
        try:
            entry_price = float(trade.entry_price)
            coin_symbol = f"{trade.coin.symbol}USDT"
            df = get_recent_candles(coin_symbol, limit=1)
            if df is None or df.empty:
                print(f"⚠️ No price data for {coin_symbol}, skipping")
                continue
            
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
            
            # Calculate TP/SL based on trade type
            if trade.trade_type == 'long':
                tp_px = entry_price * (1.0 + TAKE_PROFIT)
                sl_px = entry_price * (1.0 - STOP_LOSS)
                tp_hit = high >= tp_px
                sl_hit = low  <= sl_px
            else:  # short
                tp_px = entry_price * (1.0 - TAKE_PROFIT)
                sl_px = entry_price * (1.0 + STOP_LOSS)
                tp_hit = low  <= tp_px
                sl_hit = high >= sl_px
            
            close_reason = None
            result_bool = None
            exit_px = last_close
            
            if tp_hit and sl_hit:
                close_reason = "BOTH_HIT_SAME_BAR_SL_FIRST"; result_bool = False; exit_px = sl_px
            elif tp_hit:
                close_reason = "TAKE_PROFIT"; result_bool = True; exit_px = tp_px
            elif sl_hit:
                close_reason = "STOP_LOSS"; result_bool = False; exit_px = sl_px
            else:
                # No max hold constraint - trades stay open until TP or SL hit
                pass
            
            if close_reason:
                trade.exit_price = safe_decimal(exit_px if np.isfinite(exit_px) else last_close)
                trade.exit_timestamp = now()
                try: trade.result = result_bool
                except Exception: pass
                trade.save()
                
                pnl_pct = (float(trade.exit_price) / entry_price - 1.0) * 100.0
                if trade.trade_type == 'short':
                    pnl_pct = -pnl_pct  # Invert for short trades
                
                send_text([
                    f"📤 Closed {trade.trade_type.upper()} {trade.coin.symbol} — {close_reason}",
                    f"Entry: {entry_price:.6f} | Exit: {float(trade.exit_price):.6f} | Δ={pnl_pct:.2f}%"
                ])
                print(f"{close_reason} | {trade.trade_type.upper()} {trade.coin.symbol} @ {float(trade.exit_price):.6f}")
                
        except Exception as e:
            print(f"❌ Error closing trade for {trade.coin.symbol}: {e}")
    
    print("✅ Dual model pipeline complete")

# ORIGINAL SINGLE MODEL PIPELINE (COMMENTED OUT - USE DUAL MODEL INSTEAD)
def run_live_pipeline_original():
    print("🚀 Running live HGB long pipeline (multi-model per coin, strict feature parity)")

    # Load per-coin artifacts
    loaded = {}
    for coin, cfg in MODELS.items():
        try:
            if "bundle" in cfg and os.path.exists(cfg["bundle"]):
                m, s, feat_order, thr = load_artifacts_bundle(cfg["bundle"], cfg.get("threshold", 0.6))
            else:
                m, s, feat_order, thr = load_artifacts_separate(
                    cfg["model"], cfg["scaler"], cfg["features"], cfg.get("threshold", 0.6)
                )
            loaded[coin] = {"model": m, "scaler": s, "order": feat_order, "thr": float(thr)}
        except Exception as e:
            print(f"❌ Live pipeline error loading {coin}: {e}")
            # Do not crash the whole loop; just skip this coin
            continue

    if not loaded:
        print("❌ No coins loaded. Check artifacts.")
        return

    # Ensure data coverage
    for coin in loaded.keys():
        if not has_recent_400_candles(coin):
            ensure_recent_candles(coin, required_bars=400)
        last = get_latest_saved_timestamp(coin)
        if last is None or (datetime.utcnow().replace(tzinfo=timezone.utc) - last) > timedelta(minutes=10):
            ensure_recent_candles(coin, required_bars=400)

    # Determine signal bar (latest closed) and entry bar (next)
    utc_now = datetime.utcnow().replace(second=0, microsecond=0, tzinfo=timezone.utc)
    signal_ts = utc_now if utc_now.minute % 5 == 0 else utc_now.replace(minute=(utc_now.minute // 5) * 5)
    entry_ts = signal_ts + timedelta(minutes=5 * ENTRY_LAG_BARS)

    # Iterate coins
    for coin, arte in loaded.items():
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

            # All coins now use simplified feature engineering (20 features)
            print(f"🔧 {coin}: Using simplified feature engineering (20 features)")
            feats_df = add_features_simplified_dot(recent)
                
            if feats_df is None or feats_df.empty:
                print(f"⏭️ {coin}: insufficient/invalid features after warmup")
                continue

            drift_report(feats_df, arte["scaler"], top=8, prefix=f"[{coin}]")

            try:
                X = build_X_exact(feats_df, arte["order"], arte["scaler"])
            except Exception as e:
                print(f"⏭️ {coin}: {e}")
                continue

            model = arte["model"]
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(X)[0][1])
            else:
                raw = float(getattr(model, "decision_function", lambda x: model.predict(x))(X))
                prob = raw if 0.0 <= raw <= 1.0 else 1.0 / (1.0 + np.exp(-raw))

            thr = float(arte["thr"])


            CURRENT_MODEL = "no model found"

            if coin == "ADAUSDT":
                CURRENT_MODEL = "ada_rf_model.joblib"
            elif coin == "ATOMUSDT":
                CURRENT_MODEL = "atom_rf_model.joblib"
            elif coin == "AVAXUSDT":
                CURRENT_MODEL = "avax_lr_model.joblib"
            elif coin == "BTCUSDT":
                CURRENT_MODEL = "btc_lr_model.joblib"
            elif coin == "DOGEUSDT":
                CURRENT_MODEL = "doge_two_long_hgb_model.joblib"
            elif coin == "DOTUSDT":
                CURRENT_MODEL = "dot_rf_model.joblib"
            elif coin == "ETHUSDT":
                CURRENT_MODEL = "eth_lr_model.joblib"
            elif coin == "LINKUSDT":
                CURRENT_MODEL = "link_lr_model.joblib"
            elif coin == "LTCUSDT":
                CURRENT_MODEL = "ltc_rf_model.joblib"
            elif coin == "SHIBUSDT":
                CURRENT_MODEL = "shib_rf_model.joblib"
            elif coin == "SOLUSDT":
                CURRENT_MODEL = "sol_rf_model.joblib"
            elif coin == "TRXUSDT":
                CURRENT_MODEL = "trx_two_long_hgb_model.joblib"
            elif coin == "UNIUSDT":
                CURRENT_MODEL = "uni_rf_model.joblib"
            elif coin == "XLMUSDT":
                CURRENT_MODEL = "xlm_rf_model.joblib"
            elif coin == "XRPUSDT":
                CURRENT_MODEL = "xrp_rf_model.joblib"


            ConfidenceHistory.objects.create(
                coin=coin_obj,
                model_name=CURRENT_MODEL,
                confidence=round(prob, 4),
            )
            qs = ConfidenceHistory.objects.filter(
                coin=coin_obj
            ).order_by("-timestamp")
            if qs.count() > 12:
                for old in qs[12:]:
                    old.delete()

            print(f"📈 {coin} prob={prob:.4f} (thr=0.6)")
            if prob < 0.6:
                continue

            # Entry price (next bar)
            try:
                entry_price, entry_src = get_entry_price_or_fallback(coin, entry_ts, recent, ENTRY_SLIPPAGE_PCT)
            except Exception as e:
                print(f"⏭️ {coin}: failed to resolve entry price ({e})")
                continue

            # Telegram alert
            local_entry = entry_ts.astimezone(LOCAL_TZ)
            send_text([
                f"📥 LONG {coin}  prob={prob:.3f}  thr={thr:.2f}",
                f"Entry ({entry_src}): {entry_price:.6f}  (+{ENTRY_SLIPPAGE_PCT*100:.2f}% slippage)",
                f"TP={TAKE_PROFIT*100:.1f}%  SL={STOP_LOSS*100:.1f}%  Lev={LEVERAGE:.0f}x",
                f"UTC: {entry_ts.strftime('%Y-%m-%d %H:%M')} | PT: {local_entry.strftime('%Y-%m-%d %H:%M')}"
            ])

            # Open trade
            ModelTrade.objects.create(
                coin=coin_obj,
                trade_type='long',
                entry_timestamp=entry_ts,
                entry_price=safe_decimal(entry_price),
                model_confidence=round(prob, 4),
                take_profit_percent=TAKE_PROFIT * 100.0,
                stop_loss_percent=STOP_LOSS * 100.0,
                confidence_trade=thr,
                recent_confidences=[],
                model_name=CURRENT_MODEL,
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

            tp_px = entry_price * (1.0 + TAKE_PROFIT)
            sl_px = entry_price * (1.0 - STOP_LOSS)

            tp_hit = high >= tp_px
            sl_hit = low  <= sl_px

            close_reason = None
            result_bool = None
            exit_px = last_close

            if tp_hit and sl_hit:
                close_reason = "BOTH_HIT_SAME_BAR_SL_FIRST"; result_bool = False; exit_px = sl_px
            elif tp_hit:
                close_reason = "TAKE_PROFIT"; result_bool = True; exit_px = tp_px
            elif sl_hit:
                close_reason = "STOP_LOSS"; result_bool = False; exit_px = sl_px
            else:
                # No max hold constraint - trades stay open until TP or SL hit
                pass

            if close_reason:
                trade.exit_price = safe_decimal(exit_px if np.isfinite(exit_px) else last_close)
                trade.exit_timestamp = now()
                try: trade.result = result_bool
                except Exception: pass
                trade.save()

                pnl_pct = (float(trade.exit_price) / entry_price - 1.0) * 100.0
                send_text([
                    f"📤 Closed {trade.trade_type.upper()} {trade.coin.symbol} — {close_reason}",
                    f"Entry: {entry_price:.6f} | Exit: {float(trade.exit_price):.6f} | Δ={pnl_pct:.2f}%"
                ])
                print(f"{close_reason} | {trade.trade_type.upper()} {trade.coin.symbol} @ {float(trade.exit_price):.6f}")

        except Exception as e:
            print(f"❌ Error closing trade for {trade.coin.symbol}: {e}")

    print("✅ Pipeline complete")

# --------------------------------
# CONVENIENCE FUNCTIONS
# --------------------------------
def run_live_pipeline():
    """Main entry point - uses dual model pipeline by default"""
    run_dual_model_pipeline()

def run_single_model_pipeline():
    """Run original single model pipeline (long only)"""
    run_live_pipeline_original()

# --------------------------------
# Django Management Command
# --------------------------------
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Run dual model pipeline (long + short models)'
    
    def handle(self, *args, **options):
        """Run the dual model pipeline"""
        run_dual_model_pipeline()

# --------------------------------
# Entry (if run as script)
# --------------------------------
if __name__ == "__main__":
    # Use dual model pipeline by default
    run_dual_model_pipeline()
    
    # Uncomment below to use original single model pipeline
    # run_live_pipeline_original()
