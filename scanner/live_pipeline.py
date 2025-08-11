import os
import json
import requests
import numpy as np
import pandas as pd
import joblib
from decimal import Decimal, InvalidOperation
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from joblib import dump

from django.utils.timezone import now, make_aware
from scanner.models import (
    Coin, ModelTrade, ConfidenceHistory, LivePriceSnapshot, CoinAPIPrice
)

from skops.io import load as skops_load, get_untrusted_types

def _trusted_list(path):
    try:
        # skops >= 0.10
        return get_untrusted_types(path)
    except TypeError:
        # skops < 0.10 signature (no args)
        return get_untrusted_types()

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

# Model artifacts
# MODEL_PATH    = "two_long_hgb_model.joblib"
MODEL_PATH = "two_long_model.skops"
trusted_types = get_untrusted_types(file=MODEL_PATH)

SCALER_PATH   = "two_feature_scaler.joblib"
FEATURES_PATH = "two_feature_list.json"
CONFIDENCE_THRESHOLD = 0.85

# Trade config
TAKE_PROFIT     = 0.03
STOP_LOSS       = 0.02
LEVERAGE        = 10.0
MAX_HOLD_HOURS  = 3
ENTRY_LAG_BARS  = 1

LOCAL_TZ = ZoneInfo("America/Los_Angeles")  # for nice timestamps in Telegram alerts

# --------------------------------
# Telegram
# --------------------------------
def send_text(messages):
    if not messages:
        return
    # Your existing constants (swap to env if you prefer)
    bot_token = '7672687080:AAFWvkwzp-LQE92XdO9vcVa5yWJDUxO17yE'
    chat_ids = ['1077594551']
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
    """Ensure we have at least `required_bars` most recent 5-min candles (contiguous-ish)."""
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
    dirn = np.sign(close.diff().fillna(0))
    return (volume * dirn.replace(0, method="ffill").fillna(0)).cumsum()

def vwap(close, high, low, volume, window=20):
    tp = (high + low + close)/3.0
    pv = tp * volume
    return pv.rolling(window).sum() / (volume.rolling(window).sum() + 1e-12)

def add_features_live(df):
    g = df.copy()
    g['timestamp'] = pd.to_datetime(g['timestamp'], utc=True).dt.tz_convert(None)
    g = g.sort_values('timestamp').reset_index(drop=True)

    for n in [1,3,6,12,24,48]:
        g[f"ret_{n}"] = g["close"].pct_change(n)

    for span in [9,21,50,200]:
        g[f"ema_{span}"] = ema(g["close"], span)

    macd_line, macd_sig, macd_hist = macd(g["close"])
    g["macd"] = macd_line; g["macd_signal"] = macd_sig; g["macd_hist"] = macd_hist

    g["rsi_14"] = rsi(g["close"], 14)

    bb_u, bb_m, bb_l, bb_w = bollinger(g["close"], 20, 2.0)
    g["bb_upper"] = bb_u; g["bb_middle"] = bb_m; g["bb_lower"] = bb_l; g["bb_width"] = bb_w

    g["atr_14"] = atr(g["high"], g["low"], g["close"], 14)
    g["obv"] = obv(g["close"], g["volume"])
    g["vwap_20"] = vwap(g["close"], g["high"], g["low"], g["volume"], 20)

    g["close_above_ema_9"]   = (g["close"] > g["ema_9"]).astype(int)
    g["close_above_ema_21"]  = (g["close"] > g["ema_21"]).astype(int)
    g["close_above_ema_50"]  = (g["close"] > g["ema_50"]).astype(int)
    g["close_above_ema_200"] = (g["close"] > g["ema_200"]).astype(int)
    g["above_bb_mid"]        = (g["close"] > g["bb_middle"]).astype(int)

    g["hour"] = pd.to_datetime(g["timestamp"]).dt.hour
    g["dow"]  = pd.to_datetime(g["timestamp"]).dt.dayofweek
    g["hour_sin"] = np.sin(2*np.pi*g["hour"]/24); g["hour_cos"] = np.cos(2*np.pi*g["hour"]/24)
    g["dow_sin"]  = np.sin(2*np.pi*g["dow"]/7);   g["dow_cos"]  = np.cos(2*np.pi*g["dow"]/7)

    g.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Require core warmups; keep last row
    g = g.dropna(subset=["ema_200","bb_width","rsi_14","atr_14","obv","vwap_20"])
    if g.empty:
        return None

    last = g.tail(1).copy()

    # Price sanity (avoid zero/negatives and malformed bars)
    p_open  = float(last["open"].values[0])
    p_high  = float(last["high"].values[0])
    p_low   = float(last["low"].values[0])
    p_close = float(last["close"].values[0])
    if p_open <= 0 or p_high <= 0 or p_low <= 0 or p_close <= 0 or p_high < p_low:
        return None

    return last

# --------------------------------
# Live pipeline
# --------------------------------
def run_live_pipeline():
    print("🚀 Running live HGB long pipeline (Telegram alerts + auto-closer, 1-per-coin)")

    # Load artifacts
    # long_model = joblib.load(MODEL_PATH)
    long_model = skops_load(file=MODEL_PATH, trusted=trusted_types)
    long_scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH, "r") as f:
        long_features = json.load(f)

    # Ensure each coin has at least 400 contiguous recent candles; refresh if stale
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
    entry_ts_naive = entry_ts.replace(tzinfo=None)

    # Evaluate each coin independently; allow multiple concurrent trades (1 per coin)
    for coin in COINS:
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

            # Must include all model features; values must be finite
            missing = [c for c in long_features if c not in feats_df.columns]
            if missing:
                print(f"❌ {coin}: missing model features {missing[:5]}{'...' if len(missing)>5 else ''}")
                continue

            X = feats_df[long_features].astype("float32").values
            if not np.isfinite(X).all():
                print(f"⚠️ {coin}: non-finite feature values, skipping")
                continue

            X_scaled = long_scaler.transform(X)
            prob = float(long_model.predict_proba(X_scaled)[0][1])

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

            # Entry bar open price (enter next bar)
            base_row = (
                CoinAPIPrice.objects
                .filter(coin=coin, timestamp=entry_ts)
                .values("open")
                .first()
            )
            if not base_row:
                print(f"⏭️ {coin}: missing entry bar {entry_ts} for open price")
                continue

            entry_price = float(base_row["open"])

            # Telegram alert
            local_entry = entry_ts.astimezone(LOCAL_TZ)
            send_text([
                f"📥 *LONG signal* {coin} | prob={prob:.3f}\n",
                f"Entry (next bar open): {entry_price:.6f}\n",
                f"TP={TAKE_PROFIT*100:.1f}% | SL={STOP_LOSS*100:.1f}% | Lev={LEVERAGE:.0f}x\n",
                f"TS (UTC): {entry_ts.strftime('%Y-%m-%d %H:%M')} | ",
                f"TS (PT): {local_entry.strftime('%Y-%m-%d %H:%M')}"
            ])

            # Open trade for this coin
            ModelTrade.objects.create(
                coin=coin_obj,
                trade_type='long',
                entry_timestamp=make_aware(entry_ts_naive),
                entry_price=safe_decimal(entry_price),
                model_confidence=round(prob, 4),
                take_profit_percent=TAKE_PROFIT * 100.0,
                stop_loss_percent=STOP_LOSS * 100.0,
                confidence_trade=CONFIDENCE_THRESHOLD,
                recent_confidences=[],
            )
            print(f"✅ LONG opened: {coin} @ {entry_price:.6f} (ts={entry_ts})")

            # Do NOT break; we allow multiple coins to enter on the same run

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

            tp_hit = high >= entry_price * (1.0 + TAKE_PROFIT)
            sl_hit = low  <= entry_price * (1.0 - STOP_LOSS)

            close_reason = None
            result_bool = None

            if tp_hit:
                close_reason = "TAKE_PROFIT"
                result_bool = True
            elif sl_hit:
                close_reason = "STOP_LOSS"
                result_bool = False
            else:
                if trade.entry_timestamp:
                    age = now() - trade.entry_timestamp
                    if age >= timedelta(hours=MAX_HOLD_HOURS):
                        close_reason = "MAX_HOLD"
                        result_bool = last_close > entry_price

            if close_reason:
                trade.exit_price = safe_decimal(last_close)
                trade.exit_timestamp = now()
                try:
                    trade.result = result_bool  # if your ModelTrade has this field
                except Exception:
                    pass
                trade.save()

                pnl_pct = (last_close / entry_price - 1.0) * 100.0
                send_text([
                    f"📤 *Closed* {trade.trade_type.upper()} {trade.coin.symbol} — {close_reason}\n",
                    f"Entry: {entry_price:.6f} | Exit: {last_close:.6f} | Δ={pnl_pct:.2f}%"
                ])
                print(f"{close_reason} | {trade.trade_type.upper()} {trade.coin.symbol} @ {last_close:.6f}")

        except Exception as e:
            print(f"❌ Error closing trade for {trade.coin.symbol}: {e}")

    print("✅ Pipeline complete")

# --------------------------------
# Entry
# --------------------------------
if __name__ == "__main__":
    run_live_pipeline()
