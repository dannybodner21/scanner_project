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
        if recent_df is None or recent_df.empty:
            raise RuntimeError("No recent candles available for fallback close.")
        base = float(recent_df.iloc[-1]["close"])
        src = "fallback_last_close"

    price = base * (1.0 + slippage_pct)
    return price, src


# --------------------------------
# Feature Engineering (match training)
# --------------------------------
def ema(s, span): 
    return s.ewm(span=span, adjust=False).mean()

def sma(s, span):
    return s.rolling(span).mean()

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
    dirn = np.sign(close.diff())
    dirn = dirn.where(dirn != 0).ffill().fillna(0)
    return (volume * dirn).cumsum()

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
    pos = mf.where(mf > 0, 0).rolling(period).sum()
    neg = mf.where(mf < 0, 0).rolling(period).sum()
    return 100 - (100 / (1 + pos / (neg + 1e-12)))

def add_features_live(df):
    """Generate features matching training names; return last row (DataFrame with 1 row)."""
    g = df.copy()
    g['timestamp'] = pd.to_datetime(g['timestamp'], utc=True)  # keep UTC like training
    g = g.sort_values('timestamp').reset_index(drop=True)

    features = {}

    # ---- Price action
    features['price_range'] = (g['high'] - g['low']) / g['close']
    features['body_size'] = (g['close'] - g['open']).abs() / g['close']
    features['upper_shadow'] = (g['high'] - g[['open','close']].max(axis=1)) / g['close']
    features['lower_shadow'] = (g[['open','close']].min(axis=1) - g['low']) / g['close']

    # ---- Returns (Fibo)
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]:
        r = g['close'].pct_change(n)
        features[f'ret_{n}'] = r
        features[f'ret_{n}_abs'] = r.abs()
        features[f'ret_{n}_squared'] = r ** 2

    # ---- Volatility
    for period in [5, 10, 20, 50]:
        v = g['close'].pct_change().rolling(period).std()
        features[f'volatility_{period}'] = v
        features[f'volatility_{period}_squared'] = v ** 2

    # ---- EMAs + slopes + price vs EMA
    for span in [3, 5, 8, 13, 21, 34, 55, 89, 144, 233]:
        e = ema(g["close"], span)
        features[f'ema_{span}_slope'] = e.diff()
        features[f'ema_{span}_slope_3'] = e.diff(3)
        features[f'ema_{span}_slope_5'] = e.diff(5)
    for span in [5, 8, 13, 21, 34, 55, 89, 144, 233]:
        e = ema(g["close"], span)
        features[f'close_vs_ema_{span}'] = (g["close"] - e) / e
    for span in [34, 55, 89, 144, 233]:
        features[f'ema_{span}'] = ema(g["close"], span)

    # ---- MACD (+ slopes, crosses)
    macd_line, macd_sig, macd_hist = macd(g["close"])
    features["macd"] = macd_line
    features["macd_signal"] = macd_sig
    features["macd_hist"] = macd_hist
    features["macd_hist_slope"]   = macd_hist.diff()
    features["macd_hist_slope_3"] = macd_hist.diff(3)
    features["macd_hist_slope_5"] = macd_hist.diff(5)
    features["macd_cross_above"] = ((macd_line > macd_sig) & (macd_line.shift(1) <= macd_sig.shift(1))).astype(int)
    features["macd_cross_below"] = ((macd_line < macd_sig) & (macd_line.shift(1) >= macd_sig.shift(1))).astype(int)

    # ---- RSI (7,14,21,34) + slopes + OB/OS flags
    for period in [7, 14, 21, 34]:
        r = rsi(g["close"], period)
        features[f"rsi_{period}"] = r
        features[f"rsi_{period}_slope"] = r.diff()
        features[f"rsi_{period}_slope_3"] = r.diff(3)
        features[f"rsi_{period}_overbought"] = (r > 70).astype(int)
        features[f"rsi_{period}_oversold"]  = (r < 30).astype(int)

    # ---- Bollinger (+ middle, z, squeeze, position)
    bb_u, bb_m, bb_l, bb_w = bollinger(g["close"], 20, 2.0)
    features["bb_upper"]  = bb_u
    features["bb_middle"] = bb_m
    features["bb_lower"]  = bb_l
    features["bb_width"]  = bb_w
    bb_std = g["close"].rolling(20).std(ddof=0)
    features["bb_z"] = (g["close"] - bb_m) / (bb_std + 1e-12)
    features["bb_squeeze"] = bb_w / (g["close"].rolling(20).mean() + 1e-12)
    features["bb_position"] = (g["close"] - bb_l) / (bb_u - bb_l + 1e-12)

    # ---- Stochastic (%K/%D + crosses)
    lowest_low = g["low"].rolling(14).min()
    highest_high = g["high"].rolling(14).max()
    stoch_k = 100 * ((g["close"] - lowest_low) / (highest_high - lowest_low + 1e-12))
    stoch_d = stoch_k.rolling(3).mean()
    features["stoch_k"] = stoch_k
    features["stoch_d"] = stoch_d
    features["stoch_cross_above"] = ((stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))).astype(int)
    features["stoch_cross_below"] = ((stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1))).astype(int)

    # ---- Williams %R, CCI, MFI (+ slopes where relevant)
    wr = williams_r(g["high"], g["low"], g["close"])
    features["williams_r"] = wr
    features["williams_r_slope"] = wr.diff()
    cci_v = cci(g["high"], g["low"], g["close"])
    features["cci"] = cci_v
    features["cci_slope"] = cci_v.diff()
    mfi_v = money_flow_index(g["high"], g["low"], g["close"], g["volume"])
    features["mfi"] = mfi_v
    features["mfi_slope"] = mfi_v.diff()

    # ---- ATR + TR + pct
    features["atr_14"] = atr(g["high"], g["low"], g["close"], 14)
    features["atr_21"] = atr(g["high"], g["low"], g["close"], 21)
    tr = true_range(g["high"], g["low"], g["close"])
    features["tr"] = tr
    features["tr_pct"] = tr / (g["close"].shift(1) + 1e-12)

    # ---- VWAP (base + dev + pct) for 20 & 50
    for window in [20, 50]:
        v = vwap(g["close"], g["high"], g["low"], g["volume"], window)
        features[f'vwap_{window}'] = v
        features[f'vwap_{window}_dev'] = (g["close"] - v) / v
        features[f'vwap_{window}_dev_pct'] = features[f'vwap_{window}_dev'] * 100

    # ---- Volume analytics + rel vol (so interactions match)
    for period in [5, 10, 20, 50]:
        sma_v = g['volume'].rolling(period).mean()
        features[f'vol_sma_{period}'] = sma_v
        features[f'vol_med_{period}'] = g['volume'].rolling(period).median()
        features[f'rel_vol_{period}'] = g['volume'] / (sma_v + 1e-12)

    # ---- OBV (base + slopes)
    obv_val = obv(g['close'], g['volume'])
    features['obv'] = obv_val
    features['obv_slope'] = obv_val.diff()
    features['obv_slope_3'] = obv_val.diff(3)
    features['obv_slope_5'] = obv_val.diff(5)

    # ---- Support / Resistance (+ distances relative to close)
    for period in [20, 50, 100]:
        res = g['high'].rolling(period).max()
        sup = g['low'].rolling(period).min()
        features[f'resistance_{period}'] = res
        features[f'support_{period}'] = sup
        features[f'resistance_distance_{period}'] = (res - g['close']) / g['close']
        features[f'support_distance_{period}'] = (g['close'] - sup) / g['close']

    # ---- Momentum & ROC
    for period in [5, 10, 20, 50]:
        features[f'momentum_{period}'] = g['close'] / g['close'].shift(period) - 1
        features[f'roc_{period}'] = g['close'].pct_change(period)

    # ---- Time features (UTC like training)
    hour = g['timestamp'].dt.hour
    dow = g['timestamp'].dt.dayofweek
    month = g['timestamp'].dt.month
    features['hour'] = hour
    features['dow'] = dow
    features['month'] = month
    features['hour_sin'] = np.sin(2*np.pi*hour/24)
    features['hour_cos'] = np.cos(2*np.pi*hour/24)
    features['dow_sin'] = np.sin(2*np.pi*dow/7)
    features['dow_cos'] = np.cos(2*np.pi*dow/7)
    features['month_sin'] = np.sin(2*np.pi*month/12)
    features['month_cos'] = np.cos(2*np.pi*month/12)

    # ---- Market sessions (UTC ranges to match training)
    features['is_us_hours'] = ((hour >= 13) & (hour <= 21)).astype(int)
    features['is_asia_hours'] = ((hour >= 0) & (hour <= 8)).astype(int)
    features['is_europe_hours'] = ((hour >= 7) & (hour <= 15)).astype(int)

    # ---- Lags (match training lag set)
    lag_features = ['close', 'volume', 'rsi_14', 'macd_hist', 'bb_z', 'vwap_20_dev', 'atr_14']
    for feat in lag_features:
        # ensure exists before lagging
        src = features.get(feat, g.get(feat))
        if src is None and feat in g.columns:
            src = g[feat]
        if src is not None:
            for lag in [1, 2, 3, 5, 8]:
                features[f'{feat}_lag_{lag}'] = src.shift(lag)

    # ---- Interactions (match training)
    features['rsi_bb_interaction'] = features['rsi_14'] * features['bb_z']
    features['macd_volume_interaction'] = features['macd_hist'] * features.get('rel_vol_20', pd.Series(index=g.index, dtype='float64'))
    features['momentum_volatility_interaction'] = features['momentum_20'] * features['volatility_20']

    # ---- Clean & materialize
    for k, v in list(features.items()):
        if hasattr(v, "replace"):
            features[k] = v.replace([np.inf, -np.inf], np.nan)

    features_df = pd.DataFrame(features, index=g.index)
    g = pd.concat([g, features_df], axis=1)

    # ---- Warmup gate (now all exist)
    required = ["ema_233","bb_width","rsi_14","atr_14","obv","vwap_20","macd","stoch_k"]
    g = g.dropna(subset=[c for c in required if c in g.columns])
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
    if len(model_feats) < 100:
        raise RuntimeError(f"Model has {len(model_feats)} features, expected at least 100.")

    scaler_feats = list(getattr(s, "feature_names_in_", []))
    if not scaler_feats:
        raise RuntimeError("Scaler missing feature_names_in_. Refit scaler with DataFrame so names persist.")

    # Filter scaler to only include features the model expects
    scaler_feats = [f for f in scaler_feats if f in model_feats]

    # Build a filtered RobustScaler with copied params
    from sklearn.preprocessing import RobustScaler
    new_scaler = RobustScaler()
    keep_indices = [i for i, f in enumerate(s.feature_names_in_) if f in scaler_feats]
    new_scaler.center_ = s.center_[keep_indices]
    new_scaler.scale_ = s.scale_[keep_indices]
    new_scaler.feature_names_in_ = np.array(scaler_feats)

    print(f"Model expects {len(model_feats)} features, scaler filtered to {len(scaler_feats)} features")
    return m, new_scaler, model_feats, scaler_feats

def build_X_strict(feats_df: pd.DataFrame, model_feats, scaler, scaler_feats):
    """Return numpy array with exact model feature order; scale only scaler_feats."""
    missing = [c for c in model_feats if c not in feats_df.columns]
    if missing:
        raise RuntimeError(f"Feature columns missing from live features: {missing}")

    X_df = feats_df[model_feats].astype("float32").copy()

    # Validate binary flags when present
    for f in BINARY_FLAGS:
        if f in X_df.columns:
            vals = pd.unique(X_df[f])
            if not np.isin(vals, [0, 1]).all():
                raise RuntimeError(f"Binary flag {f} contains non 0/1 values: {vals[:5]}")

    # Scale the continuous subset
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
    entry_ts = signal_ts + timedelta(minutes=5 * ENTRY_LAG_BARS)

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

            # Quick diagnostics if anything goes off the rails later
            need = set(MODEL_FEATS)
            have = set(feats_df.columns)
            missing = sorted(need - have)
            nan_last = sorted([c for c in need if c in feats_df.columns and pd.isna(feats_df[c].iloc[-1])])
            if missing or nan_last:
                print("MISSING (first 50):", missing[:50])
                print("NaN in last row (first 50):", nan_last[:50])

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
                entry_timestamp=entry_ts,
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
            dfp = get_recent_candles(coin_symbol, limit=1)
            if dfp is None or dfp.empty:
                print(f"⚠️ No price data for {coin_symbol}, skipping")
                continue

            # Snapshot (optional)
            LivePriceSnapshot.objects.update_or_create(
                coin=trade.coin.symbol,
                defaults={
                    "open":  safe_decimal(dfp.iloc[-1]['open']),
                    "high":  safe_decimal(dfp.iloc[-1]['high']),
                    "low":   safe_decimal(dfp.iloc[-1]['low']),
                    "close": safe_decimal(dfp.iloc[-1]['close']),
                    "volume": safe_decimal(dfp.iloc[-1]['volume']),
                }
            )

            high = float(dfp.iloc[-1]['high'])
            low  = float(dfp.iloc[-1]['low'])
            last_close = float(dfp.iloc[-1]['close'])

            tp_px = entry_price * (1.0 + TAKE_PROFIT)
            sl_px = entry_price * (1.0 - STOP_LOSS)

            tp_hit = high >= tp_px
            sl_hit = low  <= sl_px

            close_reason = None
            result_bool = None
            exit_px = last_close  # fallback

            if tp_hit and sl_hit:
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
                    trade.result = result_bool
                except Exception:
                    pass
                trade.save()

                pnl_pct = (float(trade.exit_price) / entry_price - 1.0) * 100.0
                send_text([
                    f"📤 *Closed* {trade.trade_type.upper()} {trade.coin.symbol} — {close_reason}\n",
                    f"Entry: {entry_price:.6f} | Exit: {float(trade.exit_price):.6f} | Δ={pnl_pct:.2f}%"
                ])
                print(f"{close_reason} | {trade.trade_type.UPPER()} {trade.coin.symbol} @ {float(trade.exit_price):.6f}")

        except Exception as e:
            print(f"❌ Error closing trade for {trade.coin.symbol}: {e}")

    print("✅ Pipeline complete")

# --------------------------------
# Entry
# --------------------------------
if __name__ == "__main__":
    run_live_pipeline()
