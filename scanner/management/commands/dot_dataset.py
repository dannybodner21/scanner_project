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

# max hold is 4 hours

# DOTUSDT_TWO | threshold 0.55 | 61% accurate | 1k -> 83k | sl=1, tp=2
# UNIUSDT_TWO | threshold 0.55 | 41% accurate | 1k -> 3k | sl=1, tp=2
# XRPUSDT_TWO | threshold 0.55 | 45% accurate | 1k -> 3k | sl=1, tp=2
# LINKUSDT_TWO | threshold 0.45 | 45% accurate | 1k -> 19k | sl=1, tp=2
# LTCUSDT_TWO | threshold 0.55 | 45% accurate | 1k -> 3k | sl=1, tp=2
# SOLUSDT_TWO | threshold 0.5 | 51% accurate | 1k -> 9k | sl=1, tp=2
# AVAXUSDT_TWO | threshold 0.5 | 47% accurate | 1k -> 7k | sl=1, tp=2
# DOGEUSDT_TWO | threshold 0.5 | 49% accurate | 1k -> 82k | sl=1, tp=2
# SHIBUSDT_TWO | threshold 0.55 | 44% accurate | 1k -> 2k | sl=1, tp=2
# ADAUSDT_TWO | threshold 0.5 | 55% accurate | 1k -> 21k | sl=1, tp=2
# ETHUSDT_TWO | threshold 0.4 | 63% accurate | 1k -> 45k | sl=1, tp=2
# XLMUSDT_TWO | threshold 0.5 | 46% accurate | 1k -> 21k | sl=1, tp=2
# TRXUSDT _TWO | threshold 0.1 | 100% accurate (2 trades) | 1k -> 1.6k | sl=1, tp=2
# ATOMUSDT_TWO | threshold 0.5 | 50% accurate | 1k -> 3k | sl=1, tp=2
# BTCUSDT_TWO | threshold 0.38 | 68% accurate | 1k -> 7k | sl=1, tp=2

# -------------------------
# Enhanced Feature Engineering (IDENTICAL to live add_features_live, but returns ALL rows)
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

def stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-12))
    d = k.rolling(d_period).mean()
    return k, d

def adx(high, low, close, period=14):
    """Average Directional Index"""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr = true_range(high, low, close)
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr.ewm(alpha=1/period).mean())
    minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / tr.ewm(alpha=1/period).mean())
    
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12))
    adx = dx.ewm(alpha=1/period).mean()
    return adx, plus_di, minus_di

def ichimoku(high, low, close, period=9):
    """Ichimoku Cloud components"""
    tenkan = (high.rolling(period).max() + low.rolling(period).min()) / 2
    kijun = (high.rolling(period*2).max() + low.rolling(period*2).min()) / 2
    senkou_span_a = ((tenkan + kijun) / 2).shift(period)
    senkou_span_b = ((high.rolling(period*4).max() + low.rolling(period*4).min()) / 2).shift(period)
    return tenkan, kijun, senkou_span_a, senkou_span_b

def compute_features(df):
    """
    Simplified feature engineering - focus on core technical indicators that actually work
    """
    g = df.copy()
    g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True)
    g = g.sort_values("timestamp").reset_index(drop=True)

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

    # Only keep essential columns for prediction
    core_cols = ["ema_20", "ema_50", "rsi_14", "bb_position", "vol_sma_20"]
    g = g.dropna(subset=core_cols)
    return g.reset_index(drop=True)

# -------------------------
# Proper TP/SL Labeling (no more stupid "best move" logic)
# -------------------------
def create_simple_labels(df, move_threshold=0.02, max_lookahead=48):
    """
    Simple labeling: look for a move in either direction that exceeds threshold
    - Win (1): Price moves up by threshold within lookahead
    - Loss (0): Price moves down by threshold within lookahead
    - No trade: Neither threshold hit within lookahead
    
    This is much simpler to predict than complex TP/SL logic
    """
    close = df['close'].values
    n = len(close)
    labels = np.full(n, np.nan, dtype=np.float64)

    for i in range(n - max_lookahead):
        current = close[i]
        up_threshold = current * (1 + move_threshold)
        down_threshold = current * (1 - move_threshold)
        
        # Look ahead to see which threshold is hit first
        for j in range(1, max_lookahead + 1):
            if i + j >= n:
                break
                
            future_close = close[i + j]
            
            # Check if either threshold is hit
            if future_close >= up_threshold:
                labels[i] = 1.0  # Up move = win
                break
            elif future_close <= down_threshold:
                labels[i] = 0.0  # Down move = loss
                break
    
    # Set trailing NaNs to prevent look-ahead bias
    labels[-max_lookahead:] = np.nan
    return pd.Series(labels.astype("float32"), index=df.index)

def balance_training_data(X_train, y_train, method='class_weights'):
    """
    Balance training data only - never touch test data
    """
    if method == 'class_weights':
        # Use class weights instead of creating fake data
        print(f"    Using class weights for balancing (no synthetic data)")
        return X_train, y_train
    elif method == 'smote':
        # SMOTE creates fake data - not recommended
        print(f"    Warning: SMOTE creates synthetic data - switching to class weights")
        return X_train, y_train
    else:
        return X_train, y_train

# -------------------------
# Enhanced Models (no SMOTE/RUS; use class weights + better hyperparams)
# -------------------------
def build_models():
    return {
        "hgb": HistGradientBoostingClassifier(
            max_depth=8, max_iter=500, learning_rate=0.03,
            l2_regularization=0.05, min_samples_leaf=20,
            n_iter_no_change=20, random_state=42,
            max_bins=255, categorical_features=None
        ),
        "rf": RandomForestClassifier(
            n_estimators=500, max_depth=12,
            min_samples_split=30, min_samples_leaf=15,
            max_features='sqrt', bootstrap=True,
            n_jobs=-1, random_state=42, class_weight="balanced",
            max_samples=0.8
        ),
        "lr": LogisticRegression(
            C=0.1, max_iter=3000, solver="liblinear",
            class_weight="balanced", random_state=42,
            penalty='l1', tol=1e-4
        ),
    }

def compute_sample_weights(y):
    """Enhanced sample weighting with class balancing"""
    classes = np.unique(y)
    w = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    mapping = {c: w[i] for i, c in enumerate(classes)}
    return np.array([mapping[v] for v in y], dtype=np.float32)

def pick_threshold(y_true, y_proba, metric="f1"):
    """Enhanced threshold selection with more granular search"""
    thresholds = np.linspace(0.05, 0.95, 181)  # More granular search
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
# Enhanced Training routine (strict, no leakage + better validation)
# -------------------------
class Command(BaseCommand):
    help = "Enhanced leak-free training for a single coin; outputs model, scaler, feature list, config used by live pipeline."

    def add_arguments(self, parser):
        parser.add_argument("--coin", type=str, default="DOTUSDT", help="e.g., DOTUSDT / LINKUSDT / UNIUSDT")
        parser.add_argument("--export_dir", type=str, default=".", help="Where to write outputs")
        parser.add_argument("--train_start", type=str, default="2023-01-01 00:00:00+00:00")
        parser.add_argument("--train_end",   type=str, default="2025-06-30 23:55:00+00:00")
        parser.add_argument("--test_start",  type=str, default="2025-07-01 00:00:00+00:00")
        parser.add_argument("--test_end",    type=str, default="2025-08-19 23:55:00+00:00")
        parser.add_argument("--move_threshold", type=float, default=0.02, help="Move threshold for labeling (e.g., 0.02 = 2%)")
        parser.add_argument("--max_lookahead", type=int, default=48, help="Maximum bars to look ahead for labeling")
        parser.add_argument("--k_features", type=int, default=20, help="Number of features to select (keep small for focused model)")
        parser.add_argument("--val_frac", type=float, default=0.2, help="fraction of TRAIN used as validation for threshold/model pick")
        parser.add_argument("--balance_method", type=str, default="class_weights", choices=['smote', 'class_weights'], help="Method to balance training data")

    def handle(self, *args, **opts):
        COIN = opts["coin"].upper()
        out_dir = opts["export_dir"]
        os.makedirs(out_dir, exist_ok=True)
        self.stdout.write(f"▶ Loading {COIN} OHLCV...")

        # Load data with proper timezone handling
        qs = (CoinAPIPrice.objects
              .filter(coin=COIN, timestamp__gte=opts["train_start"], timestamp__lte=opts["test_end"])
              .values("coin","timestamp","open","high","low","close","volume")
              .order_by("timestamp"))
        df = pd.DataFrame.from_records(list(qs))
        if df.empty:
            self.stderr.write("No data returned.")
            return

        # Data preprocessing
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.dropna(subset=["open","high","low","close","volume"]).reset_index(drop=True)

        self.stdout.write("▶ Engineering enhanced features...")
        feat = compute_features(df)

        self.stdout.write("▶ Creating proper TP/SL labels...")
        labels = create_simple_labels(feat, opts["move_threshold"], opts["max_lookahead"])
        feat["label"] = labels
        feat = feat.dropna(subset=["label"]).reset_index(drop=True)
        feat["label"] = feat["label"].astype(int)

        # Check label distribution
        label_counts = feat["label"].value_counts()
        self.stdout.write(f"▶ Label distribution: {label_counts.to_dict()}")

        # Enhanced Train/Test split by fixed timestamps (no leakage)
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

        # Check train/test label distribution
        train_labels = train_df["label"].value_counts()
        test_labels = test_df["label"].value_counts()
        self.stdout.write(f"▶ Train labels: {train_labels.to_dict()}")
        self.stdout.write(f"▶ Test labels: {test_labels.to_dict()}")

        # Enhanced feature selection
        exclude = {'coin','timestamp','open','high','low','close','volume','label'}
        cols = [c for c in train_df.columns if c not in exclude]

        # Keep columns with at least 90% non-NaN in TRAIN ONLY
        cols = [c for c in cols if train_df[c].notna().mean() >= 0.9]
        if not cols:
            self.stderr.write("No usable features after NaN filter.")
            return

        self.stdout.write(f"▶ Selected {len(cols)} features for training")

        # ===== Enhanced Inner split: TRAIN -> inner_train / inner_val =====
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

        # BALANCE ONLY THE INNER TRAINING DATA - never touch validation or test
        self.stdout.write(f"▶ Balancing inner training data using {opts['balance_method']}...")
        if opts["balance_method"] == "smote":
            X_inner_balanced, y_inner_balanced = balance_training_data(X_inner, y_inner, "smote")
            self.stdout.write(f"  Before balancing: {pd.Series(y_inner).value_counts().to_dict()}")
            self.stdout.write(f"  After balancing: {pd.Series(y_inner_balanced).value_counts().to_dict()}")
        else:
            X_inner_balanced, y_inner_balanced = X_inner, y_inner
            self.stdout.write(f"  Using class weights (no resampling)")

        # Enhanced feature selection on INNER TRAIN ONLY
        from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
        from sklearn.ensemble import RandomForestClassifier
        
        # First pass: Use Random Forest feature importance to get top features
        print(f"▶ First pass: Random Forest feature importance selection...")
        rf_selector = RandomForestClassifier(
            n_estimators=100, 
            max_depth=8, 
            random_state=42, 
            class_weight='balanced',
            n_jobs=-1
        )
        rf_selector.fit(X_inner, y_inner)
        
        # Get feature importance scores
        feature_importance = rf_selector.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': X_inner.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"  Top 10 features by importance:")
        for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows()):
            print(f"    {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        # Select top features by importance
        k_importance = min(100, len(cols))  # Get top 100 by importance first
        top_features = feature_importance_df.head(k_importance)['feature'].tolist()
        print(f"  Selected top {len(top_features)} features by importance")
        
        # Second pass: Use mutual information on top features
        print(f"▶ Second pass: Mutual information selection on top features...")
        X_inner_top = X_inner[top_features]
        
        k = min(int(opts["k_features"]), X_inner_top.shape[1])
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X_inner_top, y_inner)
        
        selected_features = X_inner_top.columns[selector.get_support()].tolist()
        if not selected_features:
            self.stderr.write("No features selected.")
            return

        print(f"▶ Final feature selection: {len(selected_features)} features selected")
        print(f"  Selected features: {', '.join(selected_features[:10])}{'...' if len(selected_features) > 10 else ''}")
        
        # Verify feature quality
        feature_scores = selector.scores_[selector.get_support()]
        print(f"  Feature scores range: {feature_scores.min():.4f} to {feature_scores.max():.4f}")
        print(f"  Average feature score: {feature_scores.mean():.4f}")

        # Enhanced scaling on INNER TRAIN ONLY; apply to VAL
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

        # Enhanced model training with better hyperparameter tuning
        models = build_models()
        best_name, best_model, best_f1_val, best_thr = None, None, -1.0, 0.5
        
        # Use original data for training, but compute sample weights for class balancing
        sw_inner = compute_sample_weights(y_inner)

        self.stdout.write("▶ Training model candidates...")
        for name, mdl in models.items():
            try:
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
                    
                self.stdout.write(f"  {name}: F1={f1_at_thr:.4f} | thr={thr:.3f}")
                
            except Exception as e:
                self.stdout.write(f"  {name}: Failed - {e}")
                continue

        if best_model is None:
            self.stderr.write("Model selection failed.")
            return

        self.stdout.write(f"▶ Best model on VAL: {best_name} | F1={best_f1_val:.4f} | thr={best_thr:.3f}")

        # ===== Refit on FULL TRAIN with SAME selected_features; scaler refit on FULL TRAIN ONLY =====
        X_train_full = train_df[selected_features].astype("float32").fillna(0.0)
        y_train_full = train_df["label"].astype(int).values

        # Use class weights instead of fake data balancing
        self.stdout.write(f"▶ Using class weights for full training (no synthetic data)...")
        print(f"  Original class distribution: {dict(zip(*np.unique(y_train_full, return_counts=True)))}")

        scaler_full = RobustScaler()
        X_train_full_scaled = pd.DataFrame(
            scaler_full.fit_transform(X_train_full),
            columns=selected_features, index=X_train_full.index
        )
        assert hasattr(scaler_full, "feature_names_in_") and list(scaler_full.feature_names_in_) == selected_features

        # Reinitialize chosen model type and fit on FULL TRAIN with class weights
        mdl_final = type(best_model)(**best_model.get_params())
        sw_full = compute_sample_weights(y_train_full)
        
        self.stdout.write("▶ Refitting best model on full training data with class weights...")
        if isinstance(mdl_final, HistGradientBoostingClassifier):
            mdl_final.fit(X_train_full_scaled, y_train_full, sample_weight=sw_full)
        else:
            mdl_final.fit(X_train_full_scaled, y_train_full)
        assert hasattr(mdl_final, "feature_names_in_"), "Model missing feature_names_in_; must fit on a DataFrame"

        # ===== Final evaluation on UNSEEN TEST (transform only; DO NOT change threshold) =====
        # TEST DATA IS NEVER BALANCED OR MODIFIED - this is crucial!
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

        # Enhanced metrics calculation
        acc  = accuracy_score(y_test, y_pred_test)
        prec = precision_score(y_test, y_pred_test, zero_division=0)
        rec  = recall_score(y_test, y_pred_test, zero_division=0)
        f1   = f1_score(y_test, y_pred_test, average="binary")
        try:
            auc = roc_auc_score(y_test, proba_test)
        except Exception:
            auc = float('nan')
        cm = confusion_matrix(y_test, y_pred_test).tolist()

        # Additional metrics
        from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
        bal_acc = balanced_accuracy_score(y_test, y_pred_test)
        mcc = matthews_corrcoef(y_test, y_pred_test)

        self.stdout.write("▶ TEST metrics (unseen):")
        self.stdout.write(f"  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
        self.stdout.write(f"  Bal_Acc={bal_acc:.4f}  MCC={mcc:.4f}")
        self.stdout.write(f"  Confusion={cm}")

        # ===== Enhanced artifacts saving (prefix per coin) =====
        prefix = COIN.split("USDT")[0].lower()  # e.g., dot/link/uni
        model_path   = os.path.join(out_dir, f"{prefix}_two_long_hgb_model.joblib") if best_name == "hgb" else os.path.join(out_dir, f"{prefix}_{best_name}_model.joblib")
        scaler_path  = os.path.join(out_dir, f"{prefix}_two_feature_scaler.joblib")
        feats_path   = os.path.join(out_dir, f"{prefix}_two_feature_list.json")
        config_path  = os.path.join(out_dir, f"{prefix}_two_trade_config.json")
        train_csv    = os.path.join(out_dir, f"{prefix}_two_train_dataset.csv")
        test_csv     = os.path.join(out_dir, f"{prefix}_two_test_dataset.csv")
        preds_csv    = os.path.join(out_dir, f"{prefix}_two_predictions.csv")

        # Save model and scaler
        dump(mdl_final, model_path)
        dump(scaler_full, scaler_path)
        
        # Save feature list
        with open(feats_path, "w") as f:
            json.dump(selected_features, f, indent=2)

        # Enhanced configuration
        cfg = {
            "coin": COIN,
            "train_window": {"start": str(TRAIN_START), "end": str(TRAIN_END)},
            "test_window": {"start": str(TEST_START),  "end": str(TEST_END)},
            "move_threshold": float(opts["move_threshold"]),
            "max_lookahead": int(opts["max_lookahead"]),
            "strategy": "Simple move detection for easier prediction",
            "threshold": round(float(best_thr), 3),
            "balance_method": opts["balance_method"],
            "test_metrics": {
                "acc": round(float(acc), 4),
                "prec": round(float(prec), 4),
                "rec": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "auc": round(float(auc), 4) if not math.isnan(auc) else None,
                "balanced_acc": round(float(bal_acc), 4),
                "mcc": round(float(mcc), 4),
                "confusion": cm
            },
            "n_features": len(selected_features),
            "model_type": type(mdl_final).__name__,
            "feature_selection_method": "RF importance + mutual information",
            "scaling_method": "RobustScaler"
        }
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        # Enhanced dataset export with coin column
        train_out = train_df.copy()
        test_out  = test_df.copy()
        train_out["coin"] = COIN
        test_out["coin"]  = COIN

        export_cols = ['coin','timestamp','open','high','low','close','volume','label'] + selected_features
        train_out[export_cols].to_csv(train_csv, index=False)
        test_out[export_cols].to_csv(test_csv, index=False)

        # Enhanced predictions export WITH coin and additional metadata
        predictions_df = pd.DataFrame({
            "coin": COIN,
            "timestamp": test_df["timestamp"].values,
            "label": y_test,
            "pred_prob": proba_test,
            "pred_at_thr": y_pred_test,
            "confidence": np.abs(proba_test - 0.5) * 2,  # Confidence score
            "prediction_error": np.abs(y_test - proba_test),  # Prediction error
            "high_prob_long": (proba_test > 0.7).astype(int),  # High confidence long signals
            "high_prob_short": (proba_test < 0.3).astype(int),  # High confidence short signals
        })
        predictions_df.to_csv(preds_csv, index=False)

        self.stdout.write(self.style.SUCCESS(
            f"Done.\nModel   = {model_path}\nScaler  = {scaler_path}\nFeatures= {feats_path}\nConfig  = {config_path}\n"
            f"TrainCSV= {train_csv}\nTestCSV = {test_csv}\nPredCSV = {preds_csv}"
        ))
        
        # Final summary
        self.stdout.write(f"\n📊 Model Performance Summary:")
        self.stdout.write(f"  • Features: {len(selected_features)} selected from {len(cols)} candidates")
        self.stdout.write(f"  • Training samples: {len(train_df)} (using class weights, no synthetic data)")
        self.stdout.write(f"  • Test samples: {len(test_df)} (completely untouched)")
        self.stdout.write(f"  • Move threshold: {opts['move_threshold']*100:.1f}% | Max lookahead: {opts['max_lookahead']} bars")
        self.stdout.write(f"  • Strategy: Simple move detection for easier prediction")
        self.stdout.write(f"  • Best model: {best_name} with F1={best_f1_val:.4f}")
        self.stdout.write(f"  • Test F1: {f1:.4f} | Test AUC: {auc:.4f}")
        self.stdout.write(f"  • Balance method: class weights (no fake data)")
        self.stdout.write(f"  • Prediction file created with {len(predictions_df)} predictions")
        self.stdout.write(f"  • Test data was NEVER balanced or modified (crucial for real performance)")
        self.stdout.write(f"  • Simplified approach: focused on core technical indicators only")
        self.stdout.write(f"  • Feature selection: RF importance + mutual information for quality")
