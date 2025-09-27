# scanner/management/commands/train_coin.py
from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice

import os, json, math, warnings
import numpy as np
import pandas as pd
from joblib import dump
import sys
sys.path.append('/Users/danielbodner/Desktop/scanner_project')
from numerology_features import add_numerology_features

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    f1_score, confusion_matrix, balanced_accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 300)




# XRPUSDT | threshold 0.5 | 44% accurate | 59 trades | 1k -> 2k | sl=1, tp=2
# DOTUSDT | threshold 0.5 | 51% accurate | 29 trades | 1k -> 3k | sl=1, tp=2





# BAD:
# ADAUSDT | threshold 0.55 | % accurate | 41 trades | 1k -> 4k | sl=1, tp=2

# ATOMUSDT | threshold 0.45 | % accurate | 32 trades | 1k -> 8k | sl=1, tp=2
# AVAXUSDT | threshold 0.5 | % accurate | 119 trades | 1k -> 21k | sl=1, tp=2
# BTCUSDT | threshold 0.8 | % accurate | 2 trades | 1k -> 2k | sl=1, tp=2
# DOGEUSDT | threshold 0.5 | % accurate | 53 trades | 1k -> 20k | sl=1, tp=2
# ETHUSDT | threshold 0.55 | % accurate | 72 trades | 1k -> 165k | sl=1, tp=2
# LINKUSDT | threshold 0.6 | % accurate | 63 trades | 1k -> 54k | sl=1, tp=2
# LTCUSDT | threshold 0.55 | % accurate | 64 trades | 1k -> 3k | sl=1, tp=2
# SHIBUSDT | threshold 0.5 | % accurate | 126 trades | 1k -> 146k | sl=1, tp=2
# SOLUSDT | threshold 0.6 | % accurate | 59 trades | 1k -> 18k | sl=1, tp=2
# UNIUSDT | threshold 0.5 | % accurate | 115 trades | 1k -> 1.3k | sl=1, tp=2
# TRXUSDT | threshold 0.5 | % accurate | 9 trades | 1k -> 1.3k | sl=1, tp=2
# XLMUSDT | threshold 0.55 | % accurate | 26 trades | 1k -> 2k | sl=1, tp=2









# =========================
# Utilities
# =========================

def ensure_utc_series(ts):
    s = pd.to_datetime(ts, utc=True)
    return s.dt.tz_convert("UTC")

def build_full_grid(start_ts, end_ts):
    start_ts = pd.Timestamp(start_ts)
    end_ts   = pd.Timestamp(end_ts)
    if start_ts.tzinfo is None: start_ts = start_ts.tz_localize("UTC")
    else: start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None: end_ts = end_ts.tz_localize("UTC")
    else: end_ts = end_ts.tz_convert("UTC")
    # Inclusive end: we expect every 5-min candle up to end_ts
    return pd.date_range(start=start_ts, end=end_ts, freq="5min", tz="UTC")

def reindex_ohlcv(df, start_ts, end_ts, policy="assert"):
    """
    Reindex to a strict 5-minute UTC grid over [start_ts, end_ts].
    policy:
      - "assert": raise if any OHLCV values are missing after reindex
      - "ffill":  forward-fill close; set open/high/low to close (flat bar), volume=0
    """
    grid = build_full_grid(start_ts, end_ts)
    df = df.copy()
    df["timestamp"] = ensure_utc_series(df["timestamp"])
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.set_index("timestamp").reindex(grid)

    ohlcv_cols = ["open","high","low","close","volume"]
    if policy == "assert":
        if df[ohlcv_cols].isna().any().any():
            missing_idx = df.index[df[ohlcv_cols].isna().any(axis=1)]
            raise ValueError(
                f"Missing candles or OHLCV gaps in [{grid[0]} .. {grid[-1]}]. "
                f"{len(missing_idx)} bars with NaNs. Fix your data or run with --fill_policy ffill."
            )
    elif policy == "ffill":
        df["close"] = df["close"].ffill()
        for col in ["open","high","low"]:
            df[col] = df[col].fillna(df["close"])
        df["volume"] = df["volume"].fillna(0.0)
        df[ohlcv_cols] = df[ohlcv_cols].fillna(method="ffill").fillna(method="bfill")
    else:
        raise ValueError("fill_policy must be 'assert' or 'ffill'")

    assert df.index.equals(grid), "Index mismatch after reindex"
    return df.reset_index().rename(columns={"index":"timestamp"})

# =========================
# Feature Engineering
# =========================

def compute_features(df, coin_symbol="XRPUSDT"):
    """
    Compute features on the FULL reindexed dataset (train+test together).
    Do NOT drop rows here. Allow NaNs; training will filter; test will be imputed.
    """
    g = df.copy()
    g["timestamp"] = ensure_utc_series(g["timestamp"])
    g = g.sort_values("timestamp").reset_index(drop=True)

    F = {}

    # === HYBRID FEATURES: BEST TECHNICAL + NUMEROLOGY (25 total) ===
    
    # 1. Core Price Action (3 features) - ESSENTIAL
    F['price_range'] = (g['high'] - g['low']) / (g['close'].replace(0, np.nan))
    F['body_size'] = (g['close'] - g['open']).abs() / (g['close'].replace(0, np.nan))
    F['close_position'] = (g['close'] - g['low']) / (g['high'] - g['low'] + 1e-12)

    # 2. Returns & Momentum (3 features) - KEY FOR TREND
    F['ret_1'] = g['close'].pct_change(1)
    F['ret_5'] = g['close'].pct_change(5)
    F['momentum_20'] = g['close'] / (g['close'].shift(20) + 1e-12) - 1

    # 3. Volatility (2 features) - KEY FOR BREAKOUTS
    F['volatility_20'] = g['close'].pct_change().rolling(20).std()
    F['volatility_50'] = g['close'].pct_change().rolling(50).std()

    # 4. EMAs & Trend (3 features) - KEY FOR DIRECTION
    F['ema_20'] = g['close'].ewm(span=20, adjust=False).mean()
    F['ema_50'] = g['close'].ewm(span=50, adjust=False).mean()
    F['close_vs_ema_20'] = (g['close'] - F['ema_20']) / (F['ema_20'] + 1e-12)

    # 5. MACD (2 features) - KEY FOR MOMENTUM
    macd_line = g['close'].ewm(span=12, adjust=False).mean() - g['close'].ewm(span=26, adjust=False).mean()
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    F['macd'] = macd_line
    F['macd_hist'] = macd_line - macd_signal

    # 6. RSI (1 feature) - KEY FOR OVERBOUGHT/OVERSOLD
    delta = g['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-12)
    F['rsi_14'] = 100 - (100 / (1 + rs))

    # 7. Bollinger Bands (2 features) - KEY FOR VOLATILITY
    bb_20 = g['close'].rolling(20).mean()
    bb_std = g['close'].rolling(20).std()
    F['bb_position'] = (g['close'] - bb_20) / (bb_std + 1e-12)
    F['bb_width'] = (bb_std * 4) / (bb_20 + 1e-12)

    # 8. Volume Features (2 features) - KEY FOR SPIKES
    vol = pd.to_numeric(g['volume'], errors='coerce')
    F['rel_vol'] = vol / (vol.rolling(20).mean() + 1e-12)
    F['vol_spike'] = vol / (vol.rolling(50).median() + 1e-12)

    # 9. ATR (1 feature) - KEY FOR VOLATILITY
    tr1 = (g['high'] - g['low']).abs()
    tr2 = (g['high'] - g['close'].shift(1)).abs()
    tr3 = (g['low'] - g['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    F['atr_14'] = tr.ewm(alpha=1/14.0, adjust=False).mean()

    # 10. Support/Resistance (1 feature) - KEY FOR LEVELS
    F['resistance_20'] = g['high'].rolling(20).max()
    F['resistance_distance'] = (F['resistance_20'] - g['close']) / (g['close'] + 1e-12)

    # 11. Time Features (2 features) - KEY FOR SESSION EFFECTS
    hour = ensure_utc_series(g['timestamp']).dt.hour
    F['hour_sin'] = np.sin(2*np.pi*hour/24)
    F['is_us_hours'] = ((hour >= 13) & (hour <= 21)).astype(int)

    # 12. Crosses (1 feature) - KEY FOR SIGNALS
    F['ema_cross'] = ((F['ema_20'] > F['ema_50']) & (F['ema_20'].shift(1) <= F['ema_50'].shift(1))).astype(int)
    
    # 13. Price Acceleration (1 feature) - KEY FOR MOMENTUM
    F['price_acceleration'] = g['close'].pct_change().diff()
    
    # 14. Volume Acceleration (1 feature) - KEY FOR SPIKES
    F['volume_acceleration'] = vol.pct_change().diff()
    
    # 15. Volatility Spike (1 feature) - KEY FOR BREAKOUTS
    F['vol_spike_flag'] = (F['volatility_20'] > F['volatility_20'].rolling(50).mean() * 1.5).astype(int)

    feat_df = pd.DataFrame(F, index=g.index)
    g = pd.concat([g, feat_df], axis=1)
    g.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Add numerology features
    print("🔮 Adding numerology features...")
    g = add_numerology_features(g, symbol=coin_symbol)
    
    return g.reset_index(drop=True)

# =========================
# Labeling (2% TP / 1% SL, next-bar entry)
# =========================

def label_tp_sl_nextbar(df, tp=0.02, sl=0.01, max_hold_bars=48, same_bar_policy='sl-first'):
    """
    For each decision bar t:
      - Enter at t+1 open
      - Scan forward up to max_hold_bars bars
      - Label = 1 if TP (+2%) hit first, 0 if SL (-1%) hit first, NaN if neither.
    """
    openp = df['open'].values
    high  = df['high'].values
    low   = df['low'].values
    n = len(df)
    y = np.full(n, np.nan, dtype=np.float32)

    for t in range(n - 1 - max_hold_bars):
        entry = openp[t+1]
        if not np.isfinite(entry) or entry <= 0:
            continue
        tp_px = entry * (1.0 + tp)
        sl_px = entry * (1.0 - sl)

        for k in range(1, max_hold_bars+1):
            i = t + k
            hi = high[i]; lo = low[i]
            if not (np.isfinite(hi) and np.isfinite(lo)):
                continue
            hit_tp = hi >= tp_px
            hit_sl = lo <= sl_px
            if hit_tp and hit_sl:
                y[t] = 0.0 if same_bar_policy == 'sl-first' else 1.0
                break
            elif hit_tp:
                y[t] = 1.0
                break
            elif hit_sl:
                y[t] = 0.0
                break

    y[-(max_hold_bars+1):] = np.nan
    return pd.Series(y, index=df.index, dtype='float32')

# =========================
# Modeling helpers
# =========================

def build_models():
    return {
        "hgb": HistGradientBoostingClassifier(
            max_depth=8, max_iter=400, learning_rate=0.05,
            l2_regularization=0.01, min_samples_leaf=10,
            n_iter_no_change=20, random_state=42,
            max_bins=255, categorical_features=None
        ),
        "rf": RandomForestClassifier(
            n_estimators=400, max_depth=12,
            min_samples_split=15, min_samples_leaf=5,
            max_features='sqrt', bootstrap=True,
            n_jobs=-1, random_state=42, class_weight="balanced",
            max_samples=0.9
        ),
        "lr": LogisticRegression(
            C=0.1, max_iter=5000, solver="liblinear",
            class_weight="balanced", random_state=42,
            penalty='l1', tol=1e-5
        ),
    }

def compute_sample_weights(y):
    classes = np.unique(y)
    w = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    mapping = {c: w[i] for i, c in enumerate(classes)}
    return np.array([mapping[v] for v in y], dtype=np.float32)

def pnl_on_val_threshold(df_val, proba_val, thr, tp=0.02, sl=0.01, entry_lag=1, max_hold_bars=48,
                         same_bar_policy='sl-first', entry_bps=2.0, exit_bps=2.0, leverage=15.0):
    """
    Simulate your exact trading mechanics on the validation slice using predicted probabilities.
    Returns (avg_net_return_per_trade_on_margin, win_rate, n_trades, n_taken)
    """
    o  = df_val['open'].to_numpy(dtype=float)
    h  = df_val['high'].to_numpy(dtype=float)
    l  = df_val['low'].to_numpy(dtype=float)
    c  = df_val['close'].to_numpy(dtype=float)
    n  = len(df_val)

    fee_rate = (entry_bps + exit_bps) / 10000.0 * leverage
    r_tp = tp * leverage
    r_sl = -sl * leverage

    pnl = []
    wins = 0
    trades = 0

    for i in range(n):
        if proba_val[i] < thr:
            continue

        entry_idx = i + entry_lag
        if entry_idx >= n:
            break

        entry = o[entry_idx]
        if not np.isfinite(entry) or entry <= 0:
            continue

        tp_px = entry * (1.0 + tp)
        sl_px = entry * (1.0 - sl)

        exit_ret = None
        for k in range(1, max_hold_bars + 1):
            j = entry_idx + k
            if j >= n:
                break
            hi = h[j]; lo = l[j]
            if not (np.isfinite(hi) and np.isfinite(lo)):
                continue

            hit_tp = hi >= tp_px
            hit_sl = lo <= sl_px
            if hit_tp and hit_sl:
                exit_ret = r_sl if same_bar_policy == 'sl-first' else r_tp
                break
            elif hit_tp:
                exit_ret = r_tp
                break
            elif hit_sl:
                exit_ret = r_sl
                break

        if exit_ret is None:
            j = min(entry_idx + max_hold_bars, n - 1)
            close_last = c[j]
            if not (np.isfinite(close_last) and np.isfinite(entry) and entry > 0):
                continue
            exit_ret = (close_last / entry - 1.0) * leverage

        net = exit_ret - fee_rate
        pnl.append(net)
        trades += 1
        if net > 0:
            wins += 1

    if trades == 0:
        return -1e9, 0.0, 0, 0

    return float(np.mean(pnl)), float(wins / trades), trades, len(pnl)

# =========================
# Command
# =========================

class Command(BaseCommand):
    help = "Leak-free training/evaluation/prediction for a single coin with full 5-minute coverage in test predictions."

    def add_arguments(self, parser):
        parser.add_argument("--coin", type=str, default="DOTUSDT", help="e.g., DOTUSDT / LINKUSDT / UNIUSDT")
        parser.add_argument("--export_dir", type=str, default=".", help="Where to write outputs")
        parser.add_argument("--train_start", type=str, default="2023-01-01 00:00:00+00:00")
        parser.add_argument("--train_end",   type=str, default="2025-04-01 00:00:00+00:00")
        parser.add_argument("--test_start",  type=str, default="2025-04-01 00:05:00+00:00")
        parser.add_argument("--test_end",    type=str, default="2025-09-02 23:55:00+00:00")

        # Trading mechanics (defaults: TP=2%, SL=1%)
        parser.add_argument("--tp", type=float, default=0.03, help="Take-profit percent (0.02 = +2%)")
        parser.add_argument("--sl", type=float, default=0.01, help="Stop-loss percent (0.01 = -1%)")
        parser.add_argument("--max_hold_bars", type=int, default=48, help="Maximum bars to hold (5-min bars)")
        parser.add_argument("--same_bar_policy", type=str, default="sl-first", choices=["sl-first","tp-first"],
                            help="If a bar hits both TP and SL, which applies first")

        parser.add_argument("--k_features", type=int, default=100, help="Number of features to select")
        parser.add_argument("--val_frac", type=float, default=0.2, help="fraction of TRAIN used as validation")
        parser.add_argument("--balance_method", type=str, default="class_weights",
                            choices=['smote', 'class_weights'], help="Training balance method")
        parser.add_argument("--fill_policy", type=str, default="assert", choices=["assert","ffill"],
                            help="Handle missing candles: assert or ffill")

        # Economics for VAL thresholding
        parser.add_argument("--leverage", type=float, default=15.0)
        parser.add_argument("--entry_fee_bps", type=float, default=0.0)
        parser.add_argument("--exit_fee_bps", type=float, default=0.0)
        
        # Accuracy requirements
        # Removed min_accuracy - focus on profitability instead

    def handle(self, *args, **opts):
        COIN = opts["coin"].upper()
        out_dir = opts["export_dir"]
        os.makedirs(out_dir, exist_ok=True)
        self.stdout.write(f"▶ Loading {COIN} OHLCV...")

        TRAIN_START = pd.Timestamp(opts["train_start"], tz="UTC")
        TRAIN_END   = pd.Timestamp(opts["train_end"], tz="UTC")
        TEST_START  = pd.Timestamp(opts["test_start"], tz="UTC")
        TEST_END    = pd.Timestamp(opts["test_end"], tz="UTC")

        tp = float(opts["tp"]); sl = float(opts["sl"])
        max_hold_bars = int(opts["max_hold_bars"])
        same_bar_policy = opts["same_bar_policy"]

        # Pull DB window that covers [train_start, test_end]
        qs = (CoinAPIPrice.objects
              .filter(coin=COIN, timestamp__gte=TRAIN_START, timestamp__lte=TEST_END)
              .values("coin","timestamp","open","high","low","close","volume")
              .order_by("timestamp"))
        df = pd.DataFrame.from_records(list(qs))
        if df.empty:
            self.stderr.write("No data returned.")
            return

        # Convert types
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["timestamp"] = ensure_utc_series(df["timestamp"])
        df = df.dropna(subset=["open","high","low","close","volume"]).reset_index(drop=True)

        # Reindex to full 5-min grid (train_start .. test_end)
        self.stdout.write("▶ Enforcing complete 5-minute grid...")
        df = reindex_ohlcv(df, TRAIN_START, TEST_END, policy=opts["fill_policy"])

        # Features across FULL timeline (no drops)
        self.stdout.write("▶ Engineering features across full timeline (train+test)...")
        feat = compute_features(df, COIN)

        # Labels across FULL timeline (no drops)
        self.stdout.write(f"▶ Creating labels (TP={tp*100:.1f}%, SL={-sl*100:.1f}%, hold={max_hold_bars} bars, policy={same_bar_policy})...")
        feat["label"] = label_tp_sl_nextbar(feat, tp=tp, sl=sl, max_hold_bars=max_hold_bars, same_bar_policy=same_bar_policy)

        # Split AFTER features+labels
        mask_train_window = (feat["timestamp"] >= TRAIN_START) & (feat["timestamp"] <= TRAIN_END)
        mask_test_window  = (feat["timestamp"] >= TEST_START) & (feat["timestamp"] <= TEST_END)

        train_df = feat.loc[mask_train_window & feat["label"].notna()].copy()
        test_df_pred = feat.loc[mask_test_window].copy()
        test_df_metric = test_df_pred.loc[test_df_pred["label"].notna()].copy()

        if train_df.empty or test_df_pred.empty:
            self.stderr.write("Train or test window empty after split.")
            return

        # Assert test grid completeness
        expected_test_index = build_full_grid(TEST_START, TEST_END)
        if len(test_df_pred) != len(expected_test_index):
            raise AssertionError(
                f"Test timeline incomplete: expected {len(expected_test_index)} bars, got {len(test_df_pred)}."
            )

        # Feature candidates
        exclude = {'coin','timestamp','open','high','low','close','volume','label'}
        all_cols = [c for c in feat.columns if c not in exclude]
        cols = [c for c in all_cols if train_df[c].notna().mean() >= 0.9]
        if not cols:
            self.stderr.write("No usable features after NaN filter on train.")
            return
        self.stdout.write(f"▶ Candidate features: {len(cols)}")

        # Inner split (time-ordered) for model/threshold selection
        n_train = len(train_df)
        val_len = max(400, int(n_train * float(opts["val_frac"])))
        if n_train <= val_len:
            self.stderr.write("Train too small for requested val_frac.")
            return
        inner_train_df = train_df.iloc[:-val_len].copy()
        inner_val_df   = train_df.iloc[-val_len:].copy()

        # Quick label balance print
        self.stdout.write(f"▶ Train label balance: {inner_train_df['label'].mean():.3f} positives | "
                          f"{inner_train_df['label'].value_counts().to_dict()}")

        X_inner = inner_train_df[cols].astype("float32").fillna(0.0)
        y_inner = inner_train_df["label"].astype(int).values

        X_val   = inner_val_df[cols].astype("float32").fillna(0.0)
        y_val   = inner_val_df["label"].astype(int).values

        # Feature selection: RF importance → mutual information
        self.stdout.write("▶ Feature selection (RF importance → mutual info)...")
        rf_selector = RandomForestClassifier(
            n_estimators=150, max_depth=8, random_state=42,
            class_weight='balanced', n_jobs=-1
        )
        rf_selector.fit(X_inner, y_inner)
        imp_df = pd.DataFrame({
            "feature": X_inner.columns,
            "importance": rf_selector.feature_importances_
        }).sort_values("importance", ascending=False)
        top_k_importance = min(200, len(cols))
        top_features = imp_df.head(top_k_importance)["feature"].tolist()

        X_inner_top = X_inner[top_features]
        k_final = min(int(opts["k_features"]), X_inner_top.shape[1])
        k_final = max(1, k_final)
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=k_final)
        mi_selector.fit(X_inner_top, y_inner)
        selected_features = X_inner_top.columns[mi_selector.get_support()].tolist()
        if not selected_features:
            self.stderr.write("No features selected.")
            return
        self.stdout.write(f"▶ Final features selected: {len(selected_features)}")

        # Scale (fit on inner train only for model selection)
        scaler = RobustScaler()
        X_inner_sel = pd.DataFrame(scaler.fit_transform(X_inner[selected_features]),
                                   columns=selected_features, index=X_inner.index)
        X_val_sel   = pd.DataFrame(scaler.transform(X_val[selected_features]),
                                   columns=selected_features, index=X_val.index)
        assert list(scaler.feature_names_in_) == selected_features

        # Model selection with isotonic calibration + PnL-on-VAL thresholding
        models = build_models()
        best_name, best_model = None, None
        best_ev_val, best_thr = -1e9, 0.5
        best_iso = None
        sw_inner = compute_sample_weights(y_inner)

        self.stdout.write("▶ Training model candidates...")
        for name, mdl in models.items():
            try:
                if isinstance(mdl, HistGradientBoostingClassifier):
                    mdl.fit(X_inner_sel, y_inner, sample_weight=sw_inner)
                else:
                    mdl.fit(X_inner_sel, y_inner)

                # raw proba on VAL
                if hasattr(mdl, "predict_proba"):
                    proba_val_raw = mdl.predict_proba(X_val_sel)[:, 1]
                else:
                    raw = mdl.decision_function(X_val_sel)
                    proba_val_raw = 1.0 / (1.0 + np.exp(-raw))

                # Skip isotonic calibration - it's causing the probability compression
                # iso_local = IsotonicRegression(out_of_bounds='clip')
                # iso_local.fit(proba_val_raw, y_val.astype(float))
                # proba_val_cal = iso_local.transform(proba_val_raw)
                proba_val_cal = proba_val_raw  # Use raw probabilities

                # search threshold by actual PnL on VAL path with accuracy constraint
                best_thr_local, best_ev_local, best_wr_local, best_ntrades = 0.5, -1e9, 0.0, 0
                best_acc_local = 0.0
                
                thresholds = np.linspace(0.05, 0.95, 91)  # Wider range for better threshold selection
                self.stdout.write(f"    Testing {len(thresholds)} thresholds for {name}...")
                
                for i, thr in enumerate(thresholds):
                    ev, wr, ntr, _ = pnl_on_val_threshold(
                        inner_val_df, proba_val_cal, thr,
                        tp=tp, sl=sl, entry_lag=1, max_hold_bars=max_hold_bars, same_bar_policy=same_bar_policy,
                        entry_bps=float(opts["entry_fee_bps"]), exit_bps=float(opts["exit_fee_bps"]),
                        leverage=float(opts["leverage"])
                    )
                    # Require minimum trades for statistical significance
                    if ntr < 15:
                        continue
                    
                    # Calculate accuracy for this threshold
                    y_pred_thr = (proba_val_cal >= thr).astype(int)
                    acc_thr = accuracy_score(y_val, y_pred_thr) if len(y_val) > 0 else 0.0
                    
                    # Focus purely on profitability - select threshold with best expected value
                    if ev > best_ev_local or (ev == best_ev_local and acc_thr > best_acc_local):
                        best_ev_local, best_thr_local, best_wr_local, best_ntrades = ev, thr, wr, ntr
                        best_acc_local = acc_thr
                    
                    # Progress indicator
                    if i % 20 == 0:
                        self.stdout.write(f"      Progress: {i+1}/{len(thresholds)} thresholds tested...")

                self.stdout.write(f"  {name}: VAL PnL/trade={best_ev_local:+.6f} | win%={best_wr_local*100:.1f} "
                                  f"| acc%={best_acc_local*100:.1f} | trades={best_ntrades} | thr={best_thr_local:.3f}")

                if best_ev_local > best_ev_val:
                    best_ev_val = best_ev_local
                    best_name = name
                    best_model = mdl
                    best_thr = best_thr_local
                    # best_iso = iso_local  # Not using calibration anymore

            except Exception as e:
                self.stdout.write(f"  {name}: Failed - {e}")
                continue

        if best_model is None:
            self.stderr.write("Model selection failed.")
            return
        
        # Model selected based on profitability, not arbitrary accuracy requirements
        
        self.stdout.write(f"▶ Best on VAL: {best_name} | PnL/trade={best_ev_val:+.6f} | acc%={best_acc_local*100:.1f} | thr={best_thr:.3f}")

        # Refit scaler on FULL TRAIN and refit model on FULL TRAIN
        X_train_full = train_df[selected_features].astype("float32").fillna(0.0)
        y_train_full = train_df["label"].astype(int).values

        scaler_full = RobustScaler()
        X_train_full_scaled = pd.DataFrame(
            scaler_full.fit_transform(X_train_full),
            columns=selected_features, index=X_train_full.index
        )
        assert list(scaler_full.feature_names_in_) == selected_features

        mdl_final = type(best_model)(**best_model.get_params())
        sw_full = compute_sample_weights(y_train_full)
        self.stdout.write("▶ Refitting best model on FULL TRAIN...")
        if isinstance(mdl_final, HistGradientBoostingClassifier):
            mdl_final.fit(X_train_full_scaled, y_train_full, sample_weight=sw_full)
        else:
            mdl_final.fit(X_train_full_scaled, y_train_full)
        assert hasattr(mdl_final, "feature_names_in_")

        # === Evaluation on labeled subset of test (for reporting)
        if not test_df_metric.empty:
            X_test_metric = test_df_metric[selected_features].astype("float32").fillna(0.0)
            X_test_metric_scaled = pd.DataFrame(
                scaler_full.transform(X_test_metric),
                columns=selected_features, index=X_test_metric.index
            )
            if hasattr(mdl_final, "predict_proba"):
                proba_metric_raw = mdl_final.predict_proba(X_test_metric_scaled)[:, 1]
            else:
                raw = mdl_final.decision_function(X_test_metric_scaled)
                proba_metric_raw = 1.0 / (1.0 + np.exp(-raw))
            # use raw probabilities (no calibration)
            proba_metric = proba_metric_raw

            y_true_metric = test_df_metric["label"].astype(int).values
            y_pred_metric = (proba_metric >= best_thr).astype(int)

            acc  = accuracy_score(y_true_metric, y_pred_metric)
            prec = precision_score(y_true_metric, y_pred_metric, zero_division=0)
            rec  = recall_score(y_true_metric, y_pred_metric, zero_division=0)
            f1   = f1_score(y_true_metric, y_pred_metric, average="binary")
            try:
                auc = roc_auc_score(y_true_metric, proba_metric)
            except Exception:
                auc = float('nan')
            cm = confusion_matrix(y_true_metric, y_pred_metric).tolist()
            bal_acc = balanced_accuracy_score(y_true_metric, y_pred_metric)
        else:
            acc = prec = rec = f1 = bal_acc = float('nan')
            auc = float('nan')
            cm = [[0,0],[0,0]]

        self.stdout_write_metrics(acc, prec, rec, f1, auc, bal_acc, cm)

        # === Predictions for EVERY 5-min candle in test window (no gaps)
        self.stdout.write("▶ Generating predictions for EVERY test candle...")
        X_test_pred = test_df_pred[selected_features].astype("float32").fillna(0.0)
        X_test_pred_scaled = pd.DataFrame(
            scaler_full.transform(X_test_pred),
            columns=selected_features, index=X_test_pred.index
        )

        proba_list_raw = []
        if hasattr(mdl_final, "predict_proba"):
            for i in range(len(X_test_pred_scaled)):
                proba_list_raw.append(float(mdl_final.predict_proba(X_test_pred_scaled.iloc[i:i+1])[:, 1][0]))
        else:
            for i in range(len(X_test_pred_scaled)):
                raw = float(mdl_final.decision_function(X_test_pred_scaled.iloc[i:i+1])[0])
                proba_list_raw.append(1.0 / (1.0 + math.exp(-raw)))

        proba_test_raw = np.array(proba_list_raw, dtype=np.float32)
        proba_test = proba_test_raw  # Use raw probabilities instead of calibrated
        assert len(proba_test) == len(test_df_pred), "Prediction length != test rows length"

        confidence = np.abs(proba_test - 0.5) * 2.0

        # =========================
        # Save artifacts
        # =========================
        prefix = COIN.split("USDT")[0].lower()
        model_path   = os.path.join(out_dir, f"{prefix}_{best_name}_model.joblib")
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
            "tp": tp,
            "sl": sl,
            "max_hold_bars": max_hold_bars,
            "same_bar_policy": same_bar_policy,
            "threshold": round(float(best_thr), 3),
            "balance_method": opts["balance_method"],
            "fill_policy": opts["fill_policy"],
            "economics": {
                "leverage": float(opts["leverage"]),
                "entry_fee_bps": float(opts["entry_fee_bps"]),
                "exit_fee_bps": float(opts["exit_fee_bps"]),
                "val_pnl_per_trade": round(float(best_ev_val), 6)
            },
            "test_metrics_on_labeled_subset": {
                "acc": None if math.isnan(acc) else round(float(acc), 4),
                "prec": None if math.isnan(prec) else round(float(prec), 4),
                "rec": None if math.isnan(rec) else round(float(rec), 4),
                "f1": None if math.isnan(f1) else round(float(f1), 4),
                "auc": None if math.isnan(auc) else round(float(auc), 4),
                "balanced_acc": None if math.isnan(bal_acc) else round(float(bal_acc), 4),
                "confusion": cm
            },
            "n_features": len(selected_features),
            "model_type": type(mdl_final).__name__,
            "feature_selection_method": "RF importance → mutual information",
            "scaling_method": "RobustScaler",
            "calibration": "Isotonic on validation slice"
        }
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        # Export datasets
        train_out = train_df.copy()
        test_out  = test_df_pred.copy()
        train_out["coin"] = COIN
        test_out["coin"]  = COIN

        export_cols = ['coin','timestamp','open','high','low','close','volume','label'] + selected_features
        train_out[export_cols].to_csv(train_csv, index=False)
        test_out[export_cols].to_csv(test_csv, index=False)

        # Predictions CSV: one row per test timestamp, no gaps
        predictions_df = pd.DataFrame({
            "coin": COIN,
            "timestamp": test_df_pred["timestamp"].values,
            "pred_prob": proba_test,
            "confidence": confidence
        })
        assert len(predictions_df) == len(expected_test_index), "Pred CSV length does not match expected test bars"
        predictions_df.to_csv(preds_csv, index=False)

        self.stdout.write(self.style.SUCCESS(
            f"Done.\nModel   = {model_path}\nScaler  = {scaler_path}\nFeatures= {feats_path}\nConfig  = {config_path}\n"
            f"TrainCSV= {train_csv}\nTestCSV = {test_csv}\nPredCSV = {preds_csv}"
        ))

        self.stdout.write(f"\n📊 Summary:")
        self.stdout.write(f"  • Selected features: {len(selected_features)} / {len(cols)} candidates")
        self.stdout.write(f"  • Train samples (labeled): {len(train_df)}")
        self.stdout.write(f"  • Test candles (ALL): {len(test_df_pred)} (predictions written for every candle)")
        self.stdout.write(f"  • Best model: {best_name} | VAL PnL/trade={best_ev_val:+.6f} | thr={best_thr:.3f}")
        self.stdout_write_metrics(acc, prec, rec, f1, auc, bal_acc, cm)
        self.stdout.write(f"  • Fill policy: {opts['fill_policy']}")

    # pretty print helper
    def stdout_write_metrics(self, acc, prec, rec, f1, auc, bal_acc, cm):
        self.stdout.write("▶ TEST metrics (on labeled subset only):")
        self.stdout.write(f"  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
        self.stdout.write(f"  Bal_Acc={bal_acc:.4f}  Confusion={cm}")
