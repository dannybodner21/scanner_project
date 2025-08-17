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

# Only trade coins we have dedicated models for
MODELS = {
    "DOTUSDT": {
        "model": "three_long_hgb_model.joblib",
        "scaler": "three_feature_scaler.joblib",
        "features": "three_feature_list.json",
        "threshold": 0.75,
    },
    "LTCUSDT": {
        "model": "ltc_long_hgb_model.joblib",
        "scaler": "ltc_feature_scaler.joblib",
        "features": "ltc_feature_list.json",
        "threshold": 0.85,
    },
    "UNIUSDT": {
        "model": "uni_long_hgb_model.joblib",
        "scaler": "uni_feature_scaler.joblib",
        "features": "uni_feature_list.json",
        "threshold": 0.75,
    },
}

COINAPI_KEY = os.environ.get("COINAPI_KEY", "01293e2a-dcf1-4e81-8310-c6aa9d0cb743")
BASE_URL = "https://rest.coinapi.io/v1/ohlcv"

# Trade config
TAKE_PROFIT     = 0.03
STOP_LOSS       = 0.02
LEVERAGE        = 10.0
MAX_HOLD_HOURS  = 2
ENTRY_LAG_BARS  = 1
ENTRY_SLIPPAGE_PCT = float(os.environ.get("ENTRY_SLIPPAGE_PCT", "0.001"))
LOCAL_TZ = ZoneInfo("America/Los_Angeles")

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
def generate_chart_image(*args, **kwargs): return None
def generate_chart_image_30m(*args, **kwargs): return None

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
# Feature Engineering (exact parity with training two_dataset.py)
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
    ef = ema(close, fast)
    es = ema(close, slow)
    line = ef - es
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def bollinger(close, period=20, mult=2.0):
    m = close.rolling(period).mean()
    s = close.rolling(period).std(ddof=0)
    u = m + mult*s
    l = m - mult*s
    w = (u - l) / (m + 1e-12)
    return u, m, l, w, s

def true_range(h, l, c):
    pc = c.shift(1)
    a = (h - l)
    b = (h - pc).abs()
    d = (l - pc).abs()
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
    wr = -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-12))
    return wr

def cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci_val = (tp - sma_tp) / (0.015 * mad + 1e-12)
    return cci_val

def money_flow_index(high, low, close, volume, period=14):
    mf = ((close - low) - (high - close)) / (high - low + 1e-12)
    mf = mf * volume
    positive_flow = mf.where(mf > 0, 0).rolling(period).sum()
    negative_flow = mf.where(mf < 0, 0).rolling(period).sum()
    mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-12)))
    return mfi

def add_features_live(df):
    """
    Generate the exact same features as training (two_dataset.py).
    - ROC features scaled by 100
    - Session flags use UTC windows (13–21, 0–8, 7–15)
    - Includes vwap_10, ema_13/21, trend_strength_10, raw obv
    Returns the last row with all features (after core warmup) or None.
    """
    g = df.copy()
    g['timestamp'] = pd.to_datetime(g['timestamp'], utc=True)  # keep UTC-aware
    g = g.sort_values('timestamp').reset_index(drop=True)

    # Price action
    g['price_range'] = (g['high'] - g['low']) / g['close']
    g['body_size'] = (g['close'] - g['open']).abs() / g['close']
    g['upper_shadow'] = (g['high'] - g[['open', 'close']].max(axis=1)) / g['close']
    g['lower_shadow'] = (g[['open', 'close']].min(axis=1) - g['low']) / g['close']

    # Returns (Fibonacci)
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]:
        g[f'ret_{n}'] = g['close'].pct_change(n)
        g[f'ret_{n}_abs'] = g[f'ret_{n}'].abs()
        g[f'ret_{n}_squared'] = g[f'ret_{n}'] ** 2

    # Volatility
    for period in [5, 10, 20, 50]:
        g[f'volatility_{period}'] = g['close'].pct_change().rolling(period).std()
        g[f'volatility_{period}_squared'] = g[f'volatility_{period}'] ** 2

    # EMAs (values, slopes, close_vs)
    for span in [3, 5, 8, 13, 21, 34, 55, 89, 144, 233]:
        e = ema(g['close'], span)
        g[f'ema_{span}'] = e
        g[f'ema_{span}_slope'] = e.diff()
        g[f'ema_{span}_slope_3'] = e.diff(3)
        g[f'ema_{span}_slope_5'] = e.diff(5)
        g[f'close_vs_ema_{span}'] = (g['close'] - e) / (e + 1e-12)

    # MACD (+ slopes + crosses)
    macd_line, macd_sig, macd_hist = macd(g['close'])
    g['macd'] = macd_line
    g['macd_signal'] = macd_sig
    g['macd_hist'] = macd_hist
    g['macd_hist_slope'] = g['macd_hist'].diff()
    g['macd_hist_slope_3'] = g['macd_hist'].diff(3)
    g['macd_hist_slope_5'] = g['macd_hist'].diff(5)
    g['macd_cross_above'] = ((g['macd'] > g['macd_signal']) & (g['macd'].shift(1) <= g['macd_signal'].shift(1))).astype(int)
    g['macd_cross_below'] = ((g['macd'] < g['macd_signal']) & (g['macd'].shift(1) >= g['macd_signal'].shift(1))).astype(int)

    # RSI + flags
    for period in [7, 14, 21, 34]:
        r = rsi(g['close'], period)
        g[f'rsi_{period}'] = r
        g[f'rsi_{period}_slope'] = r.diff()
        g[f'rsi_{period}_slope_3'] = r.diff(3)
        g[f'rsi_{period}_overbought'] = (r > 70).astype(int)
        g[f'rsi_{period}_oversold'] = (r < 30).astype(int)

    # Bollinger (5 outputs in training)
    bb_u, bb_m, bb_l, bb_w, bb_std = bollinger(g['close'], 20, 2.0)
    g['bb_upper'] = bb_u
    g['bb_middle'] = bb_m
    g['bb_lower'] = bb_l
    g['bb_width'] = bb_w
    g['bb_z'] = (g['close'] - bb_m) / (bb_std + 1e-12)
    g['bb_squeeze'] = bb_w / (g['close'].rolling(20).mean() + 1e-12)
    g['bb_position'] = (g['close'] - bb_l) / (bb_u - bb_l + 1e-12)

    # Stochastic + crosses
    lowest_low = g['low'].rolling(14).min()
    highest_high = g['high'].rolling(14).max()
    stoch_k = 100 * ((g['close'] - lowest_low) / (highest_high - lowest_low + 1e-12))
    stoch_d = stoch_k.rolling(3).mean()
    g['stoch_k'] = stoch_k
    g['stoch_d'] = stoch_d
    g['stoch_cross_above'] = ((stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))).astype(int)
    g['stoch_cross_below'] = ((stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1))).astype(int)

    # Williams %R + slope
    g['williams_r'] = williams_r(g['high'], g['low'], g['close'])
    g['williams_r_slope'] = g['williams_r'].diff()

    # CCI + slope
    g['cci'] = cci(g['high'], g['low'], g['close'])
    g['cci_slope'] = g['cci'].diff()

    # MFI + slope
    g['mfi'] = money_flow_index(g['high'], g['low'], g['close'], g['volume'])
    g['mfi_slope'] = g['mfi'].diff()

    # ATR/TR
    g['atr_14'] = atr(g['high'], g['low'], g['close'], 14)
    g['atr_21'] = atr(g['high'], g['low'], g['close'], 21)
    g['tr'] = true_range(g['high'], g['low'], g['close'])
    g['tr_pct'] = g['tr'] / (g['close'].shift(1) + 1e-12)

    # VWAP 10/20/50 (+ dev/%)
    for window in [10, 20, 50]:
        v = vwap(g['close'], g['high'], g['low'], g['volume'], window)
        g[f'vwap_{window}'] = v
        g[f'vwap_{window}_dev'] = (g['close'] - v) / (v + 1e-12)
        g[f'vwap_{window}_dev_pct'] = g[f'vwap_{window}_dev'] * 100.0

    # Volume analysis (+ rel/spike)
    vol = pd.to_numeric(g['volume'], errors='coerce').fillna(0.0)
    for period in [5, 10, 20, 50]:
        g[f'vol_sma_{period}'] = vol.rolling(period).mean()
        g[f'vol_med_{period}'] = vol.rolling(period).median()
        g[f'rel_vol_{period}'] = vol / (g[f'vol_sma_{period}'] + 1e-12)
        g[f'vol_spike_{period}'] = vol / (g[f'vol_med_{period}'] + 1e-12)

    # OBV + slopes (keep raw obv)
    dirn = np.sign(g['close'].diff())
    dirn = dirn.replace(0, np.nan).ffill().fillna(0)
    g['obv'] = (vol * dirn).cumsum()
    g['obv_slope'] = g['obv'].diff()
    g['obv_slope_3'] = g['obv'].diff(3)
    g['obv_slope_5'] = g['obv'].diff(5)

    # Support/Resistance (+ distances) — denominator = close (as in training)
    for period in [20, 50, 100]:
        g[f'resistance_{period}'] = g['high'].rolling(period).max()
        g[f'support_{period}'] = g['low'].rolling(period).min()
        g[f'resistance_distance_{period}'] = (g[f'resistance_{period}'] - g['close']) / (g['close'] + 1e-12)
        g[f'support_distance_{period}'] = (g['close'] - g[f'support_{period}']) / (g['close'] + 1e-12)

    # Momentum & ROC (scale match)
    for period in [5, 10, 20, 50]:
        g[f'momentum_{period}'] = g['close'] / g['close'].shift(period) - 1
        g[f'roc_{period}'] = g['close'].pct_change(period) * 100.0

    # Trend strength (includes 10)
    for period in [10, 20, 50]:
        sma_short = sma(g['close'], period//2)
        sma_long = sma(g['close'], period)
        g[f'trend_strength_{period}'] = (sma_short - sma_long) / (sma_long + 1e-12)

    # Candles — match training operator precedence exactly
    g['doji'] = ( (g['close'] - g['open']).abs() <= (g['high'] - g['low']) * 0.1 ).astype(int)
    g['hammer'] = ((g['close'] - g['open']) > 0) & (g['lower_shadow'] > g['body_size'] * 2).astype(int)
    g['shooting_star'] = ((g['open'] - g['close']) > 0) & (g['upper_shadow'] > g['body_size'] * 2).astype(int)

    # Time features (UTC)
    g['hour'] = g['timestamp'].dt.hour
    g['dow'] = g['timestamp'].dt.dayofweek
    g['month'] = g['timestamp'].dt.month
    g['hour_sin'] = np.sin(2*np.pi*g['hour']/24)
    g['hour_cos'] = np.cos(2*np.pi*g['hour']/24)
    g['dow_sin'] = np.sin(2*np.pi*g['dow']/7)
    g['dow_cos'] = np.cos(2*np.pi*g['dow']/7)
    g['month_sin'] = np.sin(2*np.pi*g['month']/12)
    g['month_cos'] = np.cos(2*np.pi*g['month']/12)

    # Session flags — UTC windows (training)
    g['is_us_hours'] = ((g['hour'] >= 13) & (g['hour'] <= 21)).astype(int)
    g['is_asia_hours'] = ((g['hour'] >= 0) & (g['hour'] <= 8)).astype(int)
    g['is_europe_hours'] = ((g['hour'] >= 7) & (g['hour'] <= 15)).astype(int)

    # Lags
    lag_features = ['close', 'volume', 'rsi_14', 'macd_hist', 'bb_z', 'vwap_20_dev', 'atr_14']
    for feat in lag_features:
        if feat in g.columns:
            for lag in [1, 2, 3, 5, 8]:
                g[f'{feat}_lag_{lag}'] = g[feat].shift(lag)

    # Interactions
    g['rsi_bb_interaction'] = g['rsi_14'] * g['bb_z']
    g['macd_volume_interaction'] = g['macd_hist'] * g['rel_vol_20']
    g['momentum_volatility_interaction'] = g['momentum_20'] * g['volatility_20']

    g.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Core warmup requirement identical to training
    core_cols = ["ema_233","bb_width","rsi_14","atr_14","obv","vwap_20","macd","stoch_k"]
    g = g.dropna(subset=core_cols)
    if g.empty:
        return None

    return g.tail(1).copy()

# --------------------------------
# EXACT FEATURE ENFORCEMENT / ARTIFACT LOADING
# --------------------------------
BINARY_FLAGS = {
    "close_above_ema_9",           # keep if present
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

def load_artifacts_strict(model_path, scaler_path, features_path):
    m = joblib.load(model_path)
    s = joblib.load(scaler_path)
    with open(features_path) as f:
        json_feats = json.load(f)

    model_feats = list(getattr(m, "feature_names_in_", []))
    scaler_feats = list(getattr(s, "feature_names_in_", []))
    if not model_feats or not scaler_feats:
        raise RuntimeError("Model/scaler missing feature_names_in_. Fit with a pandas DataFrame so names persist.")

    if model_feats != json_feats:
        raise RuntimeError(f"Model feature order != JSON feature list.\nmodel[:8]={model_feats[:8]}\njson[:8]={json_feats[:8]}")
    if scaler_feats != json_feats:
        raise RuntimeError(f"Scaler feature order != JSON feature list.\nscaler[:8]={scaler_feats[:8]}\njson[:8]={json_feats[:8]}")

    print(f"[ARTIFACTS] model={os.path.basename(model_path)} expects={len(model_feats)} | scaler-cols={len(scaler_feats)}")
    return m, s, json_feats

def build_X_exact(feats_df: pd.DataFrame, feature_order: list, scaler):
    missing = [c for c in feature_order if c not in feats_df.columns]
    if missing:
        more = f" ... (+{len(missing)-50} more)" if len(missing) > 50 else ""
        raise RuntimeError(f"Feature columns missing from live features: {missing[:50]}{more}")

    X_df = feats_df[feature_order].astype("float32")
    if X_df.isna().any().any():
        bad = X_df.columns[X_df.isna().any()].tolist()
        more = f" ... (+{len(bad)-50} more)" if len(bad) > 50 else ""
        raise RuntimeError(f"NaNs present in features: {bad[:50]}{more}")

    if list(getattr(scaler, "feature_names_in_", [])) != feature_order:
        raise RuntimeError("Scaler feature order mismatch.")

    X_scaled = scaler.transform(X_df)  # ndarray
    if not np.isfinite(X_scaled).all():
        raise RuntimeError("Non-finite values after scaling.")
    return X_scaled

def drift_report(feats_row: pd.DataFrame, scaler, top=8, prefix=""):
    """Optional: top-|z| robust z-scores vs scaler center/scale to catch drift fast."""
    try:
        cols = list(getattr(scaler, "feature_names_in_", []))
        if not cols:
            return
        x = feats_row[cols].astype(np.float32).values.reshape(1, -1)
        z = (x - scaler.center_) / (scaler.scale_ + 1e-12)  # RobustScaler params
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
# Live pipeline
# --------------------------------
def run_live_pipeline():
    print("🚀 Running live HGB long pipeline (multi-model per coin, strict parity)")

    # Load per-coin artifacts
    loaded = {}
    for coin, cfg in MODELS.items():
        m, s, feats = load_artifacts_strict(cfg["model"], cfg["scaler"], cfg["features"])
        loaded[coin] = {"model": m, "scaler": s, "features": feats, "thr": cfg["threshold"]}

    # Ensure data coverage
    for coin in MODELS.keys():
        if not has_recent_400_candles(coin):
            ensure_recent_candles(coin, required_bars=400)
        last = get_latest_saved_timestamp(coin)
        if last is None or (datetime.utcnow().replace(tzinfo=timezone.utc) - last) > timedelta(minutes=10):
            ensure_recent_candles(coin, required_bars=400)

    # Determine signal bar (latest closed) and entry bar (next)
    utc_now = datetime.utcnow().replace(second=0, microsecond=0, tzinfo=timezone.utc)
    signal_ts = utc_now if utc_now.minute % 5 == 0 else utc_now.replace(minute=(utc_now.minute // 5) * 5)
    entry_ts = signal_ts + timedelta(minutes=5 * ENTRY_LAG_BARS)  # aware UTC

    # Iterate only dedicated coins
    for coin in MODELS.keys():
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
                print(f"⏭️ {coin}: insufficient/invalid features after warmup")
                continue

            # Optional drift print
            drift_report(feats_df, loaded[coin]["scaler"], top=8, prefix=f"[{coin}]")

            try:
                X = build_X_exact(feats_df, loaded[coin]["features"], loaded[coin]["scaler"])
            except Exception as e:
                print(f"⏭️ {coin}: {e}")
                continue

            prob = float(loaded[coin]["model"].predict_proba(X)[0][1])
            thr = float(loaded[coin]["thr"])

            ConfidenceHistory.objects.create(
                coin=coin_obj,
                model_name=os.path.basename(MODELS[coin]["model"]),
                confidence=round(prob, 4),
            )
            qs = ConfidenceHistory.objects.filter(
                coin=coin_obj, model_name=os.path.basename(MODELS[coin]["model"])
            ).order_by("-timestamp")
            if qs.count() > 12:
                for old in qs[12:]:
                    old.delete()

            print(f"📈 {coin} prob={prob:.4f} (thr={thr:.3f})")
            if prob < thr:
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
                confidence_trade=thr,
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
                print(f"{close_reason} | {trade.trade_type.upper()} {trade.coin.symbol} @ {float(trade.exit_price):.6f}")

        except Exception as e:
            print(f"❌ Error closing trade for {trade.coin.symbol}: {e}")

    print("✅ Pipeline complete")

# --------------------------------
# Entry (if run as script)
# --------------------------------
if __name__ == "__main__":
    run_live_pipeline()
