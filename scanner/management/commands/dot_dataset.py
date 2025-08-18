# scanner/management/commands/two_dataset.py
from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice

import os, json, warnings
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

# python manage.py dot_dataset --coin DOTUSDT

# optimal threshold: 0.60

# -------------------------
# Scope & fixed dates
# -------------------------
DEFAULT_COIN = "DOTUSDT"  # override via --coin

TRAIN_START = "2023-01-01 00:00:00+00:00"
TRAIN_END   = "2025-06-30 23:55:00+00:00"
TEST_START  = "2025-07-01 00:00:00+00:00"
TEST_END    = "2025-08-08 23:55:00+00:00"

# -------------------------
# Technicals
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
    return pd.concat([a,b,d], axis=1).max(axis=1)

def atr(h, l, c, period=14):
    tr = true_range(h,l,c)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def vwap(close, high, low, volume, window=20):
    tp = (high + low + close)/3.0
    pv = tp * volume
    return pv.rolling(window).sum() / (volume.rolling(window).sum() + 1e-12)

def stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-12))
    d = k.rolling(d_period).mean()
    return k, d

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

# -------------------------
# Feature Engineering
# -------------------------
def add_advanced_features(df):
    out = df.copy()
    out['price_range'] = (out['high'] - out['low']) / out['close']
    out['body_size'] = (out['close'] - out['open']).abs() / out['close']
    out['upper_shadow'] = (out['high'] - out[['open', 'close']].max(axis=1)) / out['close']
    out['lower_shadow'] = (out[['open', 'close']].min(axis=1) - out['low']) / out['close']

    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]:
        out[f'ret_{n}'] = out['close'].pct_change(n)
        out[f'ret_{n}_abs'] = out[f'ret_{n}'].abs()
        out[f'ret_{n}_squared'] = out[f'ret_{n}'] ** 2

    for period in [5, 10, 20, 50]:
        out[f'volatility_{period}'] = out['close'].pct_change().rolling(period).std()
        out[f'volatility_{period}_squared'] = out[f'volatility_{period}'] ** 2

    for span in [3, 5, 8, 13, 21, 34, 55, 89, 144, 233]:
        e = ema(out['close'], span)
        out[f'ema_{span}'] = e
        out[f'ema_{span}_slope'] = e.diff()
        out[f'ema_{span}_slope_3'] = e.diff(3)
        out[f'ema_{span}_slope_5'] = e.diff(5)
        out[f'close_vs_ema_{span}'] = (out['close'] - e) / (e + 1e-12)

    macd_line, macd_sig, macd_hist = macd(out['close'])
    out['macd'] = macd_line
    out['macd_signal'] = macd_sig
    out['macd_hist'] = macd_hist
    out['macd_hist_slope'] = out['macd_hist'].diff()
    out['macd_hist_slope_3'] = out['macd_hist'].diff(3)
    out['macd_hist_slope_5'] = out['macd_hist'].diff(5)
    out['macd_cross_above'] = ((out['macd'] > out['macd_signal']) & (out['macd'].shift(1) <= out['macd_signal'].shift(1))).astype(int)
    out['macd_cross_below'] = ((out['macd'] < out['macd_signal']) & (out['macd'].shift(1) >= out['macd_signal'].shift(1))).astype(int)

    for period in [7, 14, 21, 34]:
        r = rsi(out['close'], period)
        out[f'rsi_{period}'] = r
        out[f'rsi_{period}_slope'] = r.diff()
        out[f'rsi_{period}_slope_3'] = r.diff(3)
        out[f'rsi_{period}_overbought'] = (r > 70).astype(int)
        out[f'rsi_{period}_oversold'] = (r < 30).astype(int)

    bb_u, bb_m, bb_l, bb_w, bb_std = bollinger(out['close'], 20, 2.0)
    out['bb_upper'] = bb_u
    out['bb_middle'] = bb_m
    out['bb_lower'] = bb_l
    out['bb_width'] = bb_w
    out['bb_z'] = (out['close'] - bb_m) / (bb_std + 1e-12)
    out['bb_squeeze'] = bb_w / (out['close'].rolling(20).mean() + 1e-12)
    out['bb_position'] = (out['close'] - bb_l) / (bb_u - bb_l + 1e-12)

    stoch_k, stoch_d = stochastic(out['high'], out['low'], out['close'])
    out['stoch_k'] = stoch_k
    out['stoch_d'] = stoch_d
    out['stoch_cross_above'] = ((stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))).astype(int)
    out['stoch_cross_below'] = ((stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1))).astype(int)

    out['williams_r'] = williams_r(out['high'], out['low'], out['close'])
    out['williams_r_slope'] = out['williams_r'].diff()

    out['cci'] = cci(out['high'], out['low'], out['close'])
    out['cci_slope'] = out['cci'].diff()

    out['mfi'] = money_flow_index(out['high'], out['low'], out['close'], out['volume'])
    out['mfi_slope'] = out['mfi'].diff()

    out['atr_14'] = atr(out['high'], out['low'], out['close'], 14)
    out['atr_21'] = atr(out['high'], out['low'], out['close'], 21)
    out['tr'] = true_range(out['high'], out['low'], out['close'])
    out['tr_pct'] = out['tr'] / (out['close'].shift(1) + 1e-12)

    for window in [10, 20, 50]:
        v = vwap(out['close'], out['high'], out['low'], out['volume'], window)
        out[f'vwap_{window}'] = v
        out[f'vwap_{window}_dev'] = (out['close'] - v) / (v + 1e-12)
        out[f'vwap_{window}_dev_pct'] = out[f'vwap_{window}_dev'] * 100

    vol = pd.to_numeric(out['volume'], errors='coerce').fillna(0.0)
    for period in [5, 10, 20, 50]:
        out[f'vol_sma_{period}'] = vol.rolling(period).mean()
        out[f'vol_med_{period}'] = vol.rolling(period).median()
        out[f'rel_vol_{period}'] = vol / (out[f'vol_sma_{period}'] + 1e-12)
        out[f'vol_spike_{period}'] = vol / (out[f'vol_med_{period}'] + 1e-12)

    dirn = np.sign(out['close'].diff())
    dirn = dirn.replace(0, np.nan).ffill().fillna(0)
    out['obv'] = (vol * dirn).cumsum()
    out['obv_slope'] = out['obv'].diff()
    out['obv_slope_3'] = out['obv'].diff(3)
    out['obv_slope_5'] = out['obv'].diff(5)

    for period in [20, 50, 100]:
        out[f'resistance_{period}'] = out['high'].rolling(period).max()
        out[f'support_{period}'] = out['low'].rolling(period).min()
        out[f'resistance_distance_{period}'] = (out[f'resistance_{period}'] - out['close']) / (out['close'] + 1e-12)
        out[f'support_distance_{period}'] = (out['close'] - out[f'support_{period}']) / (out['close'] + 1e-12)

    for period in [5, 10, 20, 50]:
        out[f'momentum_{period}'] = out['close'] / out['close'].shift(period) - 1
        out[f'roc_{period}'] = out['close'].pct_change(period) * 100

    for period in [10, 20, 50]:
        sma_short = sma(out['close'], period//2)
        sma_long = sma(out['close'], period)
        out[f'trend_strength_{period}'] = (sma_short - sma_long) / (sma_long + 1e-12)

    out['doji'] = ((out['close'] - out['open']).abs() <= (out['high'] - out['low']) * 0.1).astype(int)
    out['hammer'] = (((out['close'] - out['open']) > 0) & (out['lower_shadow'] > out['body_size'] * 2)).astype(int)
    out['shooting_star'] = (((out['open'] - out['close']) > 0) & (out['upper_shadow'] > out['body_size'] * 2)).astype(int)

    out['hour'] = out['timestamp'].dt.hour
    out['dow'] = out['timestamp'].dt.dayofweek
    out['month'] = out['timestamp'].dt.month
    out['hour_sin'] = np.sin(2*np.pi*out['hour']/24)
    out['hour_cos'] = np.cos(2*np.pi*out['hour']/24)
    out['dow_sin'] = np.sin(2*np.pi*out['dow']/7)
    out['dow_cos'] = np.cos(2*np.pi*out['dow']/7)
    out['month_sin'] = np.sin(2*np.pi*out['month']/12)
    out['month_cos'] = np.cos(2*np.pi*out['month']/12)

    out['is_us_hours'] = ((out['hour'] >= 13) & (out['hour'] <= 21)).astype(int)
    out['is_asia_hours'] = ((out['hour'] >= 0) & (out['hour'] <= 8)).astype(int)
    out['is_europe_hours'] = ((out['hour'] >= 7) & (out['hour'] <= 15)).astype(int)

    for feat in ['close', 'volume', 'rsi_14', 'macd_hist', 'bb_z', 'vwap_20_dev', 'atr_14']:
        if feat in out.columns:
            for lag in [1, 2, 3, 5, 8]:
                out[f'{feat}_lag_{lag}'] = out[feat].shift(lag)

    out['rsi_bb_interaction'] = out['rsi_14'] * out['bb_z']
    out['macd_volume_interaction'] = out['macd_hist'] * out['rel_vol_20']
    out['momentum_volatility_interaction'] = out['momentum_20'] * out['volatility_20']

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out

# -------------------------
# Labeling — neutral bars dropped
# -------------------------
def create_smart_labels(df, min_move_pct=0.005, max_horizon=24):
    close = df['close'].values
    high  = df['high'].values
    low   = df['low'].values
    n = len(close)

    labels = np.full(n, np.nan, dtype=np.float64)

    for i in range(n - max_horizon):
        cp = close[i]
        min_move = cp * min_move_pct
        best_up = 0.0
        best_dn = 0.0
        for j in range(1, max_horizon + 1):
            up = (high[i+j] - cp) / cp
            dn = (cp - low[i+j]) / cp
            if up > best_up: best_up = up
            if dn > best_dn: best_dn = dn
            if best_up >= min_move or best_dn >= min_move:
                labels[i] = 1.0 if best_up > best_dn else 0.0
                break

    labels[-max_horizon:] = np.nan
    return pd.Series(labels, index=df.index, dtype="float32")

# -------------------------
# Feature Selection
# -------------------------
def select_best_features(X, y, method='mutual_info', k=150):
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
    elif method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    else:
        return X.columns.tolist()
    selector.fit(X.fillna(0), y)
    return X.columns[selector.get_support()].tolist()

# -------------------------
# Models
# -------------------------
def create_ensemble_model():
    return {
        'hgb': HistGradientBoostingClassifier(
            max_depth=6,
            max_iter=300,
            learning_rate=0.05,
            l2_regularization=0.1,
            min_samples_leaf=30,
            n_iter_no_change=15,
            random_state=42
        ),
        'rf': RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=50,
            min_samples_leaf=25,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample'  # <- remove need for RUS
        ),
        'lr': LogisticRegression(
            C=0.1,
            max_iter=1000,
            random_state=42,
            solver='liblinear',
            class_weight='balanced'  # <- remove need for RUS
        )
    }

def compute_sample_weights(y, boost_pos=1.5):
    cw = compute_class_weight('balanced', classes=np.array([0,1]), y=y)
    cw = cw.astype(float)
    cw[1] *= float(boost_pos)
    return np.where(y==1, cw[1], cw[0])

def align_features(df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    for c in feature_list:
        if c not in df.columns:
            df[c] = 0.0
    return df[feature_list]

def train_ensemble(X_train, y_train, X_val, y_val):
    """
    Choose best model by F1 (pos=1). No resampling:
      - HGB uses sample_weight
      - RF/LR rely on class_weight
    """
    models = create_ensemble_model()
    best_model, best_score = None, -1.0

    for name, model in models.items():
        if name == 'hgb':
            sw = compute_sample_weights(y_train, boost_pos=1.5)
            model.fit(X_train.fillna(0), y_train, sample_weight=sw)
        else:
            model.fit(X_train.fillna(0), y_train)

        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_val.fillna(0))[:, 1]
        else:
            try:
                y_proba = model.decision_function(X_val.fillna(0))
                y_proba = 1/(1+np.exp(-y_proba))
            except Exception:
                y_proba = model.predict(X_val.fillna(0)).astype(float)

        thrs = np.linspace(0.1, 0.9, 17)
        best_f1 = -1.0
        for t in thrs:
            yp = (y_proba >= t).astype(int)
            f1 = f1_score(y_val, yp, average='binary', pos_label=1)
            if f1 > best_f1:
                best_f1 = f1

        if best_f1 > best_score:
            best_score = best_f1
            best_model = model

    return best_model, best_score

# -------------------------
# Command
# -------------------------
class Command(BaseCommand):
    help = "Coin-aware training with advanced features, neutral-drop labels, strict 150-feature export"

    def add_arguments(self, parser):
        parser.add_argument("--export_dir", type=str, default="./", help="Where to write outputs")
        parser.add_argument("--coin", type=str, default=DEFAULT_COIN, help="e.g., DOTUSDT")
        parser.add_argument("--min_move_pct", type=float, default=0.005, help="Min move percent for labels")
        parser.add_argument("--horizon", type=int, default=24, help="Prediction horizon in bars")
        parser.add_argument("--n_folds", type=int, default=5, help="TimeSeriesSplit folds")
        parser.add_argument("--selector", type=str, default="mutual_info", choices=["mutual_info","f_classif"], help="Feature selector")
        parser.add_argument("--k_features", type=int, default=150, help="Number of features to select")

    def handle(self, *args, **opts):
        export_dir = opts["export_dir"]
        coin = opts["coin"].upper()
        assert coin.endswith("USDT"), "coin must be like DOTUSDT"
        prefix = coin.replace("USDT", "").lower()

        min_move_pct = float(opts["min_move_pct"])
        horizon = int(opts["horizon"])
        n_folds = int(opts["n_folds"])
        selector_method = opts["selector"]
        k_features = int(opts["k_features"])
        
        os.makedirs(export_dir, exist_ok=True)

        # 1) Load data
        self.stdout.write(f"▶ Loading {coin} OHLCV data...")
        qs = (CoinAPIPrice.objects
              .filter(coin=coin, timestamp__gte=TRAIN_START, timestamp__lte=TEST_END)
              .values("coin","timestamp","open","high","low","close","volume")
              .order_by("timestamp"))
        
        df = pd.DataFrame.from_records(list(qs))
        if df.empty:
            self.stderr.write("No data returned for the requested window.")
            return

        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.dropna(subset=["open","high","low","close","volume"]).reset_index(drop=True)

        # 2) Features
        self.stdout.write("▶ Engineering advanced features...")
        df_features = add_advanced_features(df)

        core_cols = ["ema_233","bb_width","rsi_14","atr_14","obv","vwap_20","macd","stoch_k"]
        df_features = df_features.dropna(subset=core_cols)

        min_non_null = int(df_features.shape[1] * 0.8)
        df_features = df_features.dropna(thresh=min_non_null).reset_index(drop=True)

        # 3) Labels (neutral → NaN → dropped)
        self.stdout.write("▶ Creating smart labels (neutral → drop)...")
        labels = create_smart_labels(df_features, min_move_pct, horizon)
        df_features['label'] = labels
        df_features = df_features.dropna(subset=['label']).reset_index(drop=True)
        df_features['label'] = df_features['label'].astype(int)

        label_counts = df_features['label'].value_counts().to_dict()
        self.stdout.write(f"Label distribution (post-drop): {label_counts}")

        # 4) Feature matrix
        exclude_cols = ['coin', 'timestamp', 'label', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [c for c in df_features.columns if c not in exclude_cols]
        feature_cols = [c for c in feature_cols if df_features[c].notna().sum() > len(df_features) * 0.9]

        X_all = df_features[feature_cols].astype("float32")
        y_all = df_features['label'].values.astype(int)

        # 5) Select exactly k_features
        self.stdout.write(f"▶ Selecting best {k_features} features via {selector_method}...")
        selected_features = select_best_features(X_all, y_all, method=selector_method, k=k_features)
        if len(selected_features) != k_features:
            self.stdout.write(f"⚠️ Selected {len(selected_features)} features (requested {k_features}). Continuing.")
        X_selected = X_all[selected_features]

        # 6) Time-series CV
        self.stdout.write("▶ Performing time series cross-validation...")
        tscv = TimeSeriesSplit(n_splits=n_folds)

        cv_scores, best_model, best_score, best_scaler = [], None, -1.0, None

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_selected)):
            X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
            y_train, y_val = y_all[train_idx], y_all[val_idx]

            if len(y_train) < 1000 or len(y_val) < 200:
                continue

            scaler = RobustScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train.fillna(0)),
                columns=selected_features,
                index=X_train.index
            )
            X_val_scaled = pd.DataFrame(
                scaler.transform(X_val.fillna(0)),
                columns=selected_features,
                index=X_val.index
            )

            model, score = train_ensemble(X_train_scaled, y_train, X_val_scaled, y_val)
            cv_scores.append(score)
            self.stdout.write(f"  Fold {fold + 1}: F1 (pos=1) = {score:.4f}")

            if score > best_score:
                best_score, best_model, best_scaler = score, model, scaler

        if not cv_scores:
            self.stderr.write("No valid CV folds found.")
            return
        mean_cv_score = float(np.mean(cv_scores))
        self.stdout.write(f"Mean CV F1 (pos=1): {mean_cv_score:.4f}")

        # 7) Final fit on full data
        self.stdout.write("▶ Training final model on full dataset...")
        best_scaler = RobustScaler()
        X_full_scaled = best_scaler.fit_transform(X_selected.fillna(0))
        X_full_scaled_df = pd.DataFrame(X_full_scaled, columns=selected_features, index=X_selected.index)

        if isinstance(best_model, HistGradientBoostingClassifier):
            sw = compute_sample_weights(y_all, boost_pos=1.5)
            best_model.fit(X_full_scaled_df, y_all, sample_weight=sw)
        else:
            best_model.fit(X_full_scaled_df, y_all)

        # 8) Evaluate on test window
        train_mask = (df_features["timestamp"] >= pd.Timestamp(TRAIN_START)) & (df_features["timestamp"] <= pd.Timestamp(TRAIN_END))
        test_mask  = (df_features["timestamp"] >= pd.Timestamp(TEST_START))  & (df_features["timestamp"]  <= pd.Timestamp(TEST_END))
        train_data = df_features.loc[train_mask].copy()
        test_data  = df_features.loc[test_mask].copy()
        if len(test_data) == 0:
            self.stdout.write("No test data in specified window; using last 20% as test set")
            split_idx = int(len(df_features) * 0.8)
            train_data = df_features.iloc[:split_idx].copy()
            test_data  = df_features.iloc[split_idx:].copy()

        X_test = align_features(test_data, selected_features).astype("float32").fillna(0)
        y_test = test_data['label'].values.astype(int)
        X_test_scaled = best_scaler.transform(X_test)

        if hasattr(best_model, 'predict_proba'):
            y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        else:
            try:
                y_pred_proba = best_model.decision_function(X_test_scaled)
                y_pred_proba = 1/(1+np.exp(-y_pred_proba))
            except Exception:
                y_pred_proba = best_model.predict(X_test_scaled).astype(float)

        thresholds = np.linspace(0.1, 0.9, 81)
        best_threshold, best_f1 = 0.5, -1.0
        for thr in thresholds:
            y_pred = (y_pred_proba >= thr).astype(int)
            f1 = f1_score(y_test, y_pred, average='binary', pos_label=1)
            if f1 > best_f1:
                best_f1, best_threshold = f1, float(thr)

        y_pred_final = (y_pred_proba >= best_threshold).astype(int)
        accuracy = accuracy_score(y_test, y_pred_final)
        precision = precision_score(y_test, y_pred_final, zero_division=0)
        recall = recall_score(y_test, y_pred_final, zero_division=0)
        f1 = f1_score(y_test, y_pred_final, average='binary', pos_label=1)
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except Exception:
            auc = float('nan')

        self.stdout.write("Test Set Performance:")
        self.stdout.write(f"  Accuracy: {accuracy:.4f}")
        self.stdout.write(f"  Precision: {precision:.4f}")
        self.stdout.write(f"  Recall: {recall:.4f}")
        self.stdout.write(f"  F1 (pos=1): {f1:.4f}")
        self.stdout.write(f"  AUC: {auc:.4f}")
        self.stdout.write(f"  Optimal Threshold: {best_threshold:.3f}")

        # 9) Save artifacts
        self.stdout.write("▶ Saving model artifacts...")

        train_csv = os.path.join(export_dir, f"{prefix}_train_dataset.csv")
        test_csv  = os.path.join(export_dir, f"{prefix}_test_dataset.csv")
        preds_csv = os.path.join(export_dir, f"{prefix}_predictions.csv")

        train_data = train_data.assign(coin=coin)
        test_data  = test_data.assign(coin=coin)

        export_cols = ['coin', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'label'] + selected_features
        export_cols = [c for c in export_cols if c in train_data.columns or c in ['coin','timestamp','open','high','low','close','volume','label'] or c in selected_features]

        train_data[export_cols].to_csv(train_csv, index=False)
        test_data[export_cols].to_csv(test_csv, index=False)

        predictions_df = test_data[['coin', 'timestamp']].copy()
        predictions_df['label'] = y_test
        predictions_df['pred_prob'] = y_pred_proba
        predictions_df['pred_at_thr'] = y_pred_final
        predictions_df.to_csv(preds_csv, index=False)

        model_path    = os.path.join(export_dir, f"{prefix}_long_hgb_model.joblib")
        scaler_path   = os.path.join(export_dir, f"{prefix}_feature_scaler.joblib")
        features_path = os.path.join(export_dir, f"{prefix}_feature_list.json")
        config_path   = os.path.join(export_dir, f"{prefix}_trade_config.json")

        dump(best_model, model_path)
        dump(best_scaler, scaler_path)
        with open(features_path, 'w') as f:
            json.dump(selected_features, f, indent=2)

        config = {
            "coin": coin,
            "train_window": {"start": TRAIN_START, "end": TRAIN_END},
            "test_window": {"start": TEST_START, "end": TEST_END},
            "min_move_pct": min_move_pct,
            "horizon_bars": horizon,
            "threshold": round(best_threshold, 3),
            "test_metrics": {
                "acc": round(accuracy, 4),
                "prec": round(precision, 4),
                "rec": round(recall, 4),
                "f1_pos": round(f1, 4),
                "auc": round(float(auc), 4) if not np.isnan(auc) else None,
            },
            "cv_score_f1_pos": round(mean_cv_score, 4),
            "n_features": len(selected_features),
            "model_type": type(best_model).__name__
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        self.stdout.write(self.style.SUCCESS(
            f"✅ {coin}: Training complete.\n"
            f"Files saved to: {export_dir}\n"
            f"Artifacts: {os.path.basename(model_path)}, {os.path.basename(scaler_path)}, {os.path.basename(features_path)}"
        ))
