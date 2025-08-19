# scanner/management/commands/train_coin.py
from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice

import os, json, math, warnings
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    f1_score, confusion_matrix, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 300)





# max hold is 2 hours

# UNIUSDT is good to go. threshold 0.6. 58% accurate. 1k -> 17M in 5 months. sl=1, tp=2
# XRPUSDT is good to go. threshold 0.6. 56% accurate. 1k -> 20k. sl=1, tp=2
# LINKUSDT is goog to go. threshold 0.6. 56% accurate. 1k -> 76k. sl=1, tp=2
# LTCUSDT is good to go. threshold 0.6. 56% accurate. 1k -> 35k in 5 months. sl=1, tp=2
# SOLUSDT is good to go. threshold 0.6. 60% accurate. 1k -> 231k. sl=1, tp=2
# DOGEUSDT is good to go. threshold 0.6. 51% accurate. 1k -> 1.3M. sl=1, tp=2
# AVAXUSDT is good to go. threshold 0.6. 54% accurate. 1k -> 442k. sl=1, tp=2
# BTCUSDT good but only a few trades. threshold 0.6. 80% accurate. 1k -> 2k. sl=1, tp=2

# ADAUSDT is average. threshold 0.6. 45% accurate. 1k -> 9k in 5 months. sl=1, tp=2
# ETHUSDT is not good
# TRXUSDT doesnt take enough trades
# DOTUSDT not good enough
# SHIBUSDT is ok. threshold 0.6. 47% accurate. 1k -> 20k. sl=1, tp=2
# XLMUSDT not good enough.
# ATOMUSDT is bad.




# -------------------------
# Feature engineering (IDENTICAL to live add_features_live, but returns ALL rows)
# -------------------------
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
    return pd.concat([a, b, d], axis=1).max(axis=1)

def atr(h, l, c, period=14):
    tr = true_range(h, l, c)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def vwap(close, high, low, volume, window=20):
    tp = (high + low + close) / 3.0
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

def compute_features(df):
    g = df.copy()
    g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True)
    g = g.sort_values("timestamp").reset_index(drop=True)

    F = {}
    # Price action
    F['price_range'] = (g['high'] - g['low']) / g['close']
    F['body_size'] = (g['close'] - g['open']).abs() / g['close']
    F['upper_shadow'] = (g['high'] - g[['open', 'close']].max(axis=1)) / g['close']
    F['lower_shadow'] = (g[['open', 'close']].min(axis=1) - g['low']) / g['close']

    # Returns (Fibonacci)
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]:
        r = g['close'].pct_change(n)
        F[f'ret_{n}'] = r
        F[f'ret_{n}_abs'] = r.abs()
        F[f'ret_{n}_squared'] = r ** 2

    # Volatility
    for period in [5, 10, 20, 50]:
        vol = g['close'].pct_change().rolling(period).std()
        F[f'volatility_{period}'] = vol
        F[f'volatility_{period}_squared'] = vol ** 2

    # EMAs
    for span in [3, 5, 8, 13, 21, 34, 55, 89, 144, 233]:
        e = ema(g['close'], span)
        F[f'ema_{span}'] = e
        F[f'ema_{span}_slope'] = e.diff()
        F[f'ema_{span}_slope_3'] = e.diff(3)
        F[f'ema_{span}_slope_5'] = e.diff(5)
        F[f'close_vs_ema_{span}'] = (g['close'] - e) / (e + 1e-12)

    # MACD (+ slopes)
    macd_line, macd_sig, macd_hist = macd(g['close'])
    F['macd'] = macd_line
    F['macd_signal'] = macd_sig
    F['macd_hist'] = macd_hist
    F['macd_hist_slope'] = macd_hist.diff()
    F['macd_hist_slope_3'] = macd_hist.diff(3)
    F['macd_hist_slope_5'] = macd_hist.diff(5)

    # RSI + flags
    for period in [7, 14, 21, 34]:
        r = rsi(g['close'], period)
        F[f'rsi_{period}'] = r
        F[f'rsi_{period}_slope'] = r.diff()
        F[f'rsi_{period}_slope_3'] = r.diff(3)
        F[f'rsi_{period}_overbought'] = (r > 70).astype(int)
        F[f'rsi_{period}_oversold'] = (r < 30).astype(int)

    # Bollinger
    bb_u, bb_m, bb_l, bb_w, bb_std = bollinger(g['close'], 20, 2.0)
    F['bb_upper'] = bb_u
    F['bb_middle'] = bb_m
    F['bb_lower'] = bb_l
    F['bb_width'] = bb_w
    F['bb_z'] = (g['close'] - bb_m) / (bb_std + 1e-12)
    F['bb_squeeze'] = bb_w / (g['close'].rolling(20).mean() + 1e-12)
    F['bb_position'] = (g['close'] - bb_l) / (bb_u - bb_l + 1e-12)

    # Stochastic
    lowest_low = g['low'].rolling(14).min()
    highest_high = g['high'].rolling(14).max()
    stoch_k = 100 * ((g['close'] - lowest_low) / (highest_high - lowest_low + 1e-12))
    stoch_d = stoch_k.rolling(3).mean()
    F['stoch_k'] = stoch_k
    F['stoch_d'] = stoch_d

    # Williams %R + slope
    wr = williams_r(g['high'], g['low'], g['close'])
    F['williams_r'] = wr
    F['williams_r_slope'] = wr.diff()

    # CCI + slope
    cci_val = cci(g['high'], g['low'], g['close'])
    F['cci'] = cci_val
    F['cci_slope'] = cci_val.diff()

    # MFI + slope
    mfi_val = money_flow_index(g['high'], g['low'], g['close'], g['volume'])
    F['mfi'] = mfi_val
    F['mfi_slope'] = mfi_val.diff()

    # ATR/TR
    atr14 = atr(g['high'], g['low'], g['close'], 14)
    atr21 = atr(g['high'], g['low'], g['close'], 21)
    tr = true_range(g['high'], g['low'], g['close'])
    F['atr_14'] = atr14
    F['atr_21'] = atr21
    F['tr'] = tr
    F['tr_pct'] = tr / (g['close'].shift(1) + 1e-12)

    # VWAP 10/20/50 (+ dev/%)
    for window in [10, 20, 50]:
        v = vwap(g['close'], g['high'], g['low'], g['volume'], window)
        F[f'vwap_{window}'] = v
        dev = (g['close'] - v) / (v + 1e-12)
        F[f'vwap_{window}_dev'] = dev
        F[f'vwap_{window}_dev_pct'] = dev * 100.0

    # Volume analysis
    vol = pd.to_numeric(g['volume'], errors='coerce').fillna(0.0)
    for period in [5, 10, 20, 50]:
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
    for period in [20, 50, 100]:
        res = g['high'].rolling(period).max()
        sup = g['low'].rolling(period).min()
        F[f'resistance_{period}'] = res
        F[f'support_{period}'] = sup
        F[f'resistance_distance_{period}'] = (res - g['close']) / (g['close'] + 1e-12)
        F[f'support_distance_{period}'] = (g['close'] - sup) / (g['close'] + 1e-12)

    # Momentum & ROC
    for period in [5, 10, 20, 50]:
        F[f'momentum_{period}'] = g['close'] / g['close'].shift(period) - 1
        F[f'roc_{period}'] = g['close'].pct_change(period) * 100.0

    # Trend strength
    for period in [10, 20, 50]:
        sma_short = sma(g['close'], period // 2)
        sma_long = sma(g['close'], period)
        F[f'trend_strength_{period}'] = (sma_short - sma_long) / (sma_long + 1e-12)

    # Candles
    F['doji'] = ((g['close'] - g['open']).abs() <= (g['high'] - g['low']) * 0.1).astype(int)
    F['hammer'] = (((g['close'] - g['open']) > 0) & (F['lower_shadow'] > F['body_size'] * 2)).astype(int)
    F['shooting_star'] = (((g['open'] - g['close']) > 0) & (F['upper_shadow'] > F['body_size'] * 2)).astype(int)

    # Time features (UTC)
    hour = g['timestamp'].dt.hour
    dow = g['timestamp'].dt.dayofweek
    month = g['timestamp'].dt.month
    F['hour'] = hour
    F['dow'] = dow
    F['month'] = month
    F['hour_sin'] = np.sin(2*np.pi*hour/24)
    F['hour_cos'] = np.cos(2*np.pi*hour/24)
    F['dow_sin'] = np.sin(2*np.pi*dow/7)
    F['dow_cos'] = np.cos(2*np.pi*dow/7)
    F['month_sin'] = np.sin(2*np.pi*month/12)
    F['month_cos'] = np.cos(2*np.pi*month/12)

    # Session flags
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
        for lag in [1, 2, 3, 5, 8]:
            feat_df[f'{name}_lag_{lag}'] = s.shift(lag)

    # Interactions
    feat_df['rsi_bb_interaction'] = feat_df['rsi_14'] * feat_df['bb_z']
    feat_df['macd_volume_interaction'] = feat_df['macd_hist'] * feat_df['rel_vol_20']
    feat_df['momentum_volatility_interaction'] = feat_df['momentum_20'] * feat_df['volatility_20']

    g = pd.concat([g, feat_df], axis=1)
    g.replace([np.inf, -np.inf], np.nan, inplace=True)

    core_cols = ["ema_233", "bb_width", "rsi_14", "atr_14", "obv", "vwap_20", "macd", "stoch_k"]
    g = g.dropna(subset=core_cols)
    return g.reset_index(drop=True)

# -------------------------
# Labeling (smart, with explicit trailing NaNs)
# -------------------------
def create_smart_labels(df, min_move_pct=0.005, max_horizon=24):
    close = df['close'].values
    high  = df['high'].values
    low   = df['low'].values
    n = len(close)
    labels = np.full(n, np.nan, dtype=np.float64)

    for i in range(n - max_horizon):
        current = close[i]
        min_move = current * min_move_pct
        for j in range(1, max_horizon + 1):
            fh = high[i + j]; fl = low[i + j]
            if (fh >= current + min_move) or (fl <= current - min_move):
                up = (fh - current) / current
                dn = (current - fl) / current
                if up > dn and up >= min_move_pct:
                    labels[i] = 1.0
                elif dn > up and dn >= min_move_pct:
                    labels[i] = 0.0
                break
    labels[-max_horizon:] = np.nan
    return pd.Series(labels.astype("float32"), index=df.index)

# -------------------------
# Models (no SMOTE/RUS; use class weights)
# -------------------------
def build_models():
    return {
        "hgb": HistGradientBoostingClassifier(
            max_depth=6, max_iter=300, learning_rate=0.05,
            l2_regularization=0.1, min_samples_leaf=30,
            n_iter_no_change=15, random_state=42
        ),
        "rf": RandomForestClassifier(
            n_estimators=400, max_depth=10,
            min_samples_split=50, min_samples_leaf=25,
            n_jobs=-1, random_state=42, class_weight="balanced"
        ),
        "lr": LogisticRegression(
            C=0.2, max_iter=2000, solver="liblinear",
            class_weight="balanced", random_state=42
        ),
    }

def compute_sample_weights(y):
    classes = np.unique(y)
    w = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    mapping = {c: w[i] for i, c in enumerate(classes)}
    return np.array([mapping[v] for v in y], dtype=np.float32)

def pick_threshold(y_true, y_proba, metric="f1"):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best = 0.5, -1.0
    for t in thresholds:
        yp = (y_proba >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, yp, average='binary', zero_division=0)
        else:
            score = average_precision_score(y_true, y_proba)
        if score > best:
            best, best_t = score, t
    return float(best_t), float(best)

# -------------------------
# Training routine (strict, no leakage)
# -------------------------
class Command(BaseCommand):
    help = "Leak-free training for a single coin; outputs model, scaler, feature list, config used by live pipeline."

    def add_arguments(self, parser):
        parser.add_argument("--coin", type=str, default="DOTUSDT", help="e.g., DOTUSDT / LINKUSDT / UNIUSDT")
        parser.add_argument("--export_dir", type=str, default=".", help="Where to write outputs")
        parser.add_argument("--train_start", type=str, default="2023-01-01 00:00:00+00:00")
        parser.add_argument("--train_end",   type=str, default="2025-03-31 23:55:00+00:00")
        parser.add_argument("--test_start",  type=str, default="2025-04-01 00:00:00+00:00")
        parser.add_argument("--test_end",    type=str, default="2025-08-18 23:55:00+00:00")
        parser.add_argument("--min_move_pct", type=float, default=0.025)
        parser.add_argument("--horizon", type=int, default=12)
        parser.add_argument("--k_features", type=int, default=150)
        parser.add_argument("--val_frac", type=float, default=0.2, help="fraction of TRAIN used as validation for threshold/model pick")

    def handle(self, *args, **opts):
        COIN = opts["coin"].upper()
        out_dir = opts["export_dir"]
        os.makedirs(out_dir, exist_ok=True)
        self.stdout.write(f"▶ Loading {COIN} OHLCV...")

        qs = (CoinAPIPrice.objects
              .filter(coin=COIN, timestamp__gte=opts["train_start"], timestamp__lte=opts["test_end"])
              .values("coin","timestamp","open","high","low","close","volume")
              .order_by("timestamp"))
        df = pd.DataFrame.from_records(list(qs))
        if df.empty:
            self.stderr.write("No data returned.")
            return

        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.dropna(subset=["open","high","low","close","volume"]).reset_index(drop=True)

        self.stdout.write("▶ Engineering features...")
        feat = compute_features(df)

        self.stdout.write("▶ Creating labels (drop NaNs)...")
        labels = create_smart_labels(feat, opts["min_move_pct"], opts["horizon"])
        feat["label"] = labels
        feat = feat.dropna(subset=["label"]).reset_index(drop=True)
        feat["label"] = feat["label"].astype(int)

        # Train/Test split by fixed timestamps (no leakage)
        TRAIN_START = pd.Timestamp(opts["train_start"])
        TRAIN_END   = pd.Timestamp(opts["train_end"])
        TEST_START  = pd.Timestamp(opts["test_start"])
        TEST_END    = pd.Timestamp(opts["test_end"])

        train_mask = (feat["timestamp"] >= TRAIN_START) & (feat["timestamp"] <= TRAIN_END)
        test_mask  = (feat["timestamp"] >= TEST_START) & (feat["timestamp"] <= TEST_END)

        train_df = feat.loc[train_mask].copy()
        test_df  = feat.loc[test_mask].copy()

        if train_df.empty or test_df.empty:
            self.stderr.write("Train or test window empty.")
            return

        # Feature columns: drop obvious non-features
        exclude = {'coin','timestamp','open','high','low','close','volume','label'}
        cols = [c for c in train_df.columns if c not in exclude]

        # Keep columns with at least 90% non-NaN in TRAIN ONLY
        cols = [c for c in cols if train_df[c].notna().mean() >= 0.9]
        if not cols:
            self.stderr.write("No usable features after NaN filter.")
            return

        # ===== Inner split: TRAIN -> inner_train / inner_val (for model/threshold pick) =====
        n_train = len(train_df)
        val_len = max(200, int(n_train * float(opts["val_frac"])))
        if n_train <= val_len:
            self.stderr.write("Train too small for requested val_frac.")
            return
        inner_train_df = train_df.iloc[:-val_len].copy()
        inner_val_df   = train_df.iloc[-val_len:].copy()

        X_inner = inner_train_df[cols].astype("float32").fillna(0.0)
        y_inner = inner_train_df["label"].astype(int).values

        X_val   = inner_val_df[cols].astype("float32").fillna(0.0)
        y_val   = inner_val_df["label"].astype(int).values

        # Feature selection on INNER TRAIN ONLY
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        k = min(int(opts["k_features"]), X_inner.shape[1])
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X_inner, y_inner)
        selected_features = X_inner.columns[selector.get_support()].tolist()
        if not selected_features:
            self.stderr.write("No features selected.")
            return

        # Scale on INNER TRAIN ONLY; apply to VAL
        scaler = RobustScaler()
        X_inner_sel = pd.DataFrame(
            scaler.fit_transform(X_inner[selected_features]),
            columns=selected_features, index=X_inner.index
        )
        X_val_sel = pd.DataFrame(
            scaler.transform(X_val[selected_features]),
            columns=selected_features, index=X_val.index
        )
        # Parity assertions (these are important for live)
        assert hasattr(scaler, "feature_names_in_") and list(scaler.feature_names_in_) == selected_features

        # Train candidates on INNER TRAIN; pick by VAL F1
        models = build_models()
        best_name, best_model, best_f1_val, best_thr = None, None, -1.0, 0.5
        sw_inner = compute_sample_weights(y_inner)

        for name, mdl in models.items():
            if isinstance(mdl, HistGradientBoostingClassifier):
                mdl.fit(X_inner_sel, y_inner, sample_weight=sw_inner)
            else:
                mdl.fit(X_inner_sel, y_inner)

            if hasattr(mdl, "predict_proba"):
                proba_val = mdl.predict_proba(X_val_sel)[:, 1]
            else:
                raw = mdl.decision_function(X_val_sel)
                proba_val = 1.0 / (1.0 + np.exp(-raw))

            thr, f1_at_thr = pick_threshold(y_val, proba_val, metric="f1")
            if f1_at_thr > best_f1_val:
                best_f1_val = f1_at_thr
                best_name = name
                best_model = mdl
                best_thr = thr

        if best_model is None:
            self.stderr.write("Model selection failed.")
            return

        self.stdout.write(f"▶ Best model on VAL: {best_name} | F1={best_f1_val:.4f} | thr={best_thr:.3f}")

        # ===== Refit on FULL TRAIN with SAME selected_features; scaler refit on FULL TRAIN ONLY =====
        X_train_full = train_df[selected_features].astype("float32").fillna(0.0)
        y_train_full = train_df["label"].astype(int).values

        scaler_full = RobustScaler()
        X_train_full_scaled = pd.DataFrame(
            scaler_full.fit_transform(X_train_full),
            columns=selected_features, index=X_train_full.index
        )
        assert hasattr(scaler_full, "feature_names_in_") and list(scaler_full.feature_names_in_) == selected_features

        # Reinitialize chosen model type and fit on FULL TRAIN
        mdl_final = type(best_model)(**best_model.get_params())
        sw_full = compute_sample_weights(y_train_full)
        if isinstance(mdl_final, HistGradientBoostingClassifier):
            mdl_final.fit(X_train_full_scaled, y_train_full, sample_weight=sw_full)
        else:
            mdl_final.fit(X_train_full_scaled, y_train_full)
        assert hasattr(mdl_final, "feature_names_in_"), "Model missing feature_names_in_; must fit on a DataFrame"

        # ===== Final evaluation on UNSEEN TEST (transform only; DO NOT change threshold) =====
        X_test = test_df[selected_features].astype("float32").fillna(0.0)
        X_test_scaled = pd.DataFrame(
            scaler_full.transform(X_test),
            columns=selected_features, index=X_test.index
        )
        y_test = test_df["label"].astype(int).values

        if hasattr(mdl_final, "predict_proba"):
            proba_test = mdl_final.predict_proba(X_test_scaled)[:, 1]
        else:
            raw = mdl_final.decision_function(X_test_scaled)
            proba_test = 1.0 / (1.0 + np.exp(-raw))

        y_pred_test = (proba_test >= best_thr).astype(int)

        acc  = accuracy_score(y_test, y_pred_test)
        prec = precision_score(y_test, y_pred_test, zero_division=0)
        rec  = recall_score(y_test, y_pred_test, zero_division=0)
        f1   = f1_score(y_test, y_pred_test, average="binary")
        try:
            auc = roc_auc_score(y_test, proba_test)
        except Exception:
            auc = float('nan')
        cm = confusion_matrix(y_test, y_pred_test).tolist()

        self.stdout.write("▶ TEST metrics (unseen):")
        self.stdout.write(f"  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
        self.stdout.write(f"  Confusion={cm}")

        # ===== Save artifacts (prefix per coin) =====
        prefix = COIN.split("USDT")[0].lower()  # e.g., dot/link/uni
        model_path   = os.path.join(out_dir, f"{prefix}_long_hgb_model.joblib") if best_name == "hgb" else os.path.join(out_dir, f"{prefix}_{best_name}_model.joblib")
        scaler_path  = os.path.join(out_dir, f"{prefix}_feature_scaler.joblib")
        feats_path   = os.path.join(out_dir, f"{prefix}_feature_list.json")
        config_path  = os.path.join(out_dir, f"{prefix}_trade_config.json")
        train_csv    = os.path.join(out_dir, f"{prefix}_train_dataset.csv")
        test_csv     = os.path.join(out_dir, f"{prefix}_test_dataset.csv")
        preds_csv    = os.path.join(out_dir, f"{prefix}_predictions.csv")

        dump(mdl_final, model_path)
        dump(scaler_full, scaler_path)
        with open(feats_path, "w") as f:
            json.dump(selected_features, f, indent=2)

        cfg = {
            "coin": COIN,
            "train_window": {"start": str(TRAIN_START), "end": str(TRAIN_END)},
            "test_window": {"start": str(TEST_START),  "end": str(TEST_END)},
            "min_move_pct": float(opts["min_move_pct"]),
            "horizon_bars": int(opts["horizon"]),
            "threshold": round(float(best_thr), 3),
            "test_metrics": {
                "acc": round(float(acc), 4),
                "prec": round(float(prec), 4),
                "rec": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "auc": round(float(auc), 4) if not math.isnan(auc) else None,
                "confusion": cm
            },
            "n_features": len(selected_features),
            "model_type": type(mdl_final).__name__
        }
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        # Add coin to datasets before export
        train_out = train_df.copy()
        test_out  = test_df.copy()
        train_out["coin"] = COIN
        test_out["coin"]  = COIN

        export_cols = ['coin','timestamp','open','high','low','close','volume','label'] + selected_features
        train_out[export_cols].to_csv(train_csv, index=False)
        test_out[export_cols].to_csv(test_csv, index=False)

        # Save predictions WITH coin
        pd.DataFrame({
            "coin": COIN,
            "timestamp": test_df["timestamp"].values,
            "label": y_test,
            "pred_prob": proba_test,
            "pred_at_thr": y_pred_test
        }).to_csv(preds_csv, index=False)

        self.stdout.write(self.style.SUCCESS(
            f"Done.\nModel   = {model_path}\nScaler  = {scaler_path}\nFeatures= {feats_path}\nConfig  = {config_path}\n"
            f"TrainCSV= {train_csv}\nTestCSV = {test_csv}\nPredCSV = {preds_csv}"
        ))
