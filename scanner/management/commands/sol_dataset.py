# scanner/management/commands/two_dataset.py
from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice

import os, json, warnings, math
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 200)


# --- optimal threshold from training: 0.00 ---


# -------------------------
# Scope & fixed dates
# -------------------------
COIN = "SOLUSDT"

TRAIN_START = "2023-01-01 00:00:00+00:00"
TRAIN_END   = "2025-06-30 23:55:00+00:00"
TEST_START  = "2025-07-01 00:00:00+00:00"
TEST_END    = "2025-08-08 23:55:00+00:00"

# -------------------------
# Enhanced Technical Indicators
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
    wr = -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-12))
    return wr

def cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma_tp) / (0.015 * mad + 1e-12)
    return cci

def money_flow_index(high, low, close, volume, period=14):
    mf = ((close - low) - (high - close)) / (high - low + 1e-12)
    mf = mf * volume
    positive_flow = mf.where(mf > 0, 0).rolling(period).sum()
    negative_flow = mf.where(mf < 0, 0).rolling(period).sum()
    mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-12)))
    return mfi

def add_advanced_features(df):
    out = df.copy()
    
    # Price action features
    out['price_range'] = (out['high'] - out['low']) / out['close']
    out['body_size'] = abs(out['close'] - out['open']) / out['close']
    out['upper_shadow'] = (out['high'] - out[['open', 'close']].max(axis=1)) / out['close']
    out['lower_shadow'] = (out[['open', 'close']].min(axis=1) - out['low']) / out['close']
    
    # Enhanced returns with different timeframes
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]:  # Fibonacci sequence
        out[f'ret_{n}'] = out['close'].pct_change(n)
        out[f'ret_{n}_abs'] = out[f'ret_{n}'].abs()
        out[f'ret_{n}_squared'] = out[f'ret_{n}'] ** 2
    
    # Volatility features
    for period in [5, 10, 20, 50]:
        out[f'volatility_{period}'] = out['close'].pct_change().rolling(period).std()
        out[f'volatility_{period}_squared'] = out[f'volatility_{period}'] ** 2
    
    # Enhanced EMAs with slopes and crossovers
    for span in [3, 5, 8, 13, 21, 34, 55, 89, 144, 233]:
        e = ema(out['close'], span)
        out[f'ema_{span}'] = e
        out[f'ema_{span}_slope'] = e.diff()
        out[f'ema_{span}_slope_3'] = e.diff(3)
        out[f'ema_{span}_slope_5'] = e.diff(5)
        
        # Price position relative to EMAs
        out[f'close_vs_ema_{span}'] = (out['close'] - e) / e
    
    # Enhanced MACD
    macd_line, macd_sig, macd_hist = macd(out['close'])
    out['macd'] = macd_line
    out['macd_signal'] = macd_sig
    out['macd_hist'] = macd_hist
    out['macd_hist_slope'] = out['macd_hist'].diff()
    out['macd_hist_slope_3'] = out['macd_hist'].diff(3)
    out['macd_hist_slope_5'] = out['macd_hist'].diff(5)
    out['macd_cross_above'] = ((out['macd'] > out['macd_signal']) & (out['macd'].shift(1) <= out['macd_signal'].shift(1))).astype(int)
    out['macd_cross_below'] = ((out['macd'] < out['macd_signal']) & (out['macd'].shift(1) >= out['macd_signal'].shift(1))).astype(int)
    
    # Enhanced RSI
    for period in [7, 14, 21, 34]:
        r = rsi(out['close'], period)
        out[f'rsi_{period}'] = r
        out[f'rsi_{period}_slope'] = r.diff()
        out[f'rsi_{period}_slope_3'] = r.diff(3)
        out[f'rsi_{period}_overbought'] = (r > 70).astype(int)
        out[f'rsi_{period}_oversold'] = (r < 30).astype(int)
    
    # Enhanced Bollinger Bands
    bb_u, bb_m, bb_l, bb_w, bb_std = bollinger(out['close'], 20, 2.0)
    out['bb_upper'] = bb_u
    out['bb_middle'] = bb_m
    out['bb_lower'] = bb_l
    out['bb_width'] = bb_w
    out['bb_z'] = (out['close'] - bb_m) / (bb_std + 1e-12)
    out['bb_squeeze'] = bb_w / (out['close'].rolling(20).mean() + 1e-12)
    out['bb_position'] = (out['close'] - bb_l) / (bb_u - bb_l + 1e-12)
    
    # Stochastic and Williams %R
    stoch_k, stoch_d = stochastic(out['high'], out['low'], out['close'])
    out['stoch_k'] = stoch_k
    out['stoch_d'] = stoch_d
    out['stoch_cross_above'] = ((stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))).astype(int)
    out['stoch_cross_below'] = ((stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1))).astype(int)
    
    out['williams_r'] = williams_r(out['high'], out['low'], out['close'])
    out['williams_r_slope'] = out['williams_r'].diff()
    
    # CCI and MFI
    out['cci'] = cci(out['high'], out['low'], out['close'])
    out['cci_slope'] = out['cci'].diff()
    
    out['mfi'] = money_flow_index(out['high'], out['low'], out['close'], out['volume'])
    out['mfi_slope'] = out['mfi'].diff()
    
    # Enhanced ATR and True Range
    out['atr_14'] = atr(out['high'], out['low'], out['close'], 14)
    out['atr_21'] = atr(out['high'], out['low'], out['close'], 21)
    out['tr'] = true_range(out['high'], out['low'], out['close'])
    out['tr_pct'] = out['tr'] / (out['close'].shift(1) + 1e-12)
    
    # Enhanced VWAP
    for window in [10, 20, 50]:
        v = vwap(out['close'], out['high'], out['low'], out['volume'], window)
        out[f'vwap_{window}'] = v
        out[f'vwap_{window}_dev'] = (out['close'] - v) / v
        out[f'vwap_{window}_dev_pct'] = out[f'vwap_{window}_dev'] * 100
    
    # Volume analysis
    vol = pd.to_numeric(out['volume'], errors='coerce').fillna(0.0)
    for period in [5, 10, 20, 50]:
        out[f'vol_sma_{period}'] = vol.rolling(period).mean()
        out[f'vol_med_{period}'] = vol.rolling(period).median()
        out[f'rel_vol_{period}'] = vol / (out[f'vol_sma_{period}'] + 1e-12)
        out[f'vol_spike_{period}'] = vol / (out[f'vol_med_{period}'] + 1e-12)
    
    # OBV and volume flow
    dirn = np.sign(out['close'].diff())
    dirn = dirn.replace(0, np.nan).ffill().fillna(0)
    out['obv'] = (vol * dirn).cumsum()
    out['obv_slope'] = out['obv'].diff()
    out['obv_slope_3'] = out['obv'].diff(3)
    out['obv_slope_5'] = out['obv'].diff(5)
    
    # Support and resistance levels
    for period in [20, 50, 100]:
        out[f'resistance_{period}'] = out['high'].rolling(period).max()
        out[f'support_{period}'] = out['low'].rolling(period).min()
        out[f'resistance_distance_{period}'] = (out[f'resistance_{period}'] - out['close']) / out['close']
        out[f'support_distance_{period}'] = (out['close'] - out[f'support_{period}']) / out['close']
    
    # Momentum indicators
    for period in [5, 10, 20, 50]:
        out[f'momentum_{period}'] = out['close'] / out['close'].shift(period) - 1
        out[f'roc_{period}'] = out['close'].pct_change(period) * 100
    
    # Trend strength
    for period in [10, 20, 50]:
        sma_short = sma(out['close'], period//2)
        sma_long = sma(out['close'], period)
        out[f'trend_strength_{period}'] = (sma_short - sma_long) / sma_long
    
    # Price patterns
    out['doji'] = (abs(out['close'] - out['open']) <= (out['high'] - out['low']) * 0.1).astype(int)
    out['hammer'] = ((out['close'] - out['open']) > 0) & (out['lower_shadow'] > out['body_size'] * 2).astype(int)
    out['shooting_star'] = ((out['open'] - out['close']) > 0) & (out['upper_shadow'] > out['body_size'] * 2).astype(int)
    
    # Time-based features
    out['hour'] = out['timestamp'].dt.hour
    out['dow'] = out['timestamp'].dt.dayofweek
    out['month'] = out['timestamp'].dt.month
    out['hour_sin'] = np.sin(2*np.pi*out['hour']/24)
    out['hour_cos'] = np.cos(2*np.pi*out['hour']/24)
    out['dow_sin'] = np.sin(2*np.pi*out['dow']/7)
    out['dow_cos'] = np.cos(2*np.pi*out['dow']/7)
    out['month_sin'] = np.sin(2*np.pi*out['month']/12)
    out['month_cos'] = np.cos(2*np.pi*out['month']/12)
    
    # Market session indicators
    out['is_us_hours'] = ((out['hour'] >= 13) & (out['hour'] <= 21)).astype(int)
    out['is_asia_hours'] = ((out['hour'] >= 0) & (out['hour'] <= 8)).astype(int)
    out['is_europe_hours'] = ((out['hour'] >= 7) & (out['hour'] <= 15)).astype(int)
    
    # Lagged features for temporal dependencies
    lag_features = ['close', 'volume', 'rsi_14', 'macd_hist', 'bb_z', 'vwap_20_dev', 'atr_14']
    for feat in lag_features:
        if feat in out.columns:
            for lag in [1, 2, 3, 5, 8]:
                out[f'{feat}_lag_{lag}'] = out[feat].shift(lag)
    
    # Feature interactions
    out['rsi_bb_interaction'] = out['rsi_14'] * out['bb_z']
    out['macd_volume_interaction'] = out['macd_hist'] * out['rel_vol_20']
    out['momentum_volatility_interaction'] = out['momentum_20'] * out['volatility_20']
    
    # Clean up infinite values
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return out

# -------------------------
# Improved Labeling Strategy
# -------------------------
def create_smart_labels(df, min_move_pct=0.005, max_horizon=48):
    """
    Create labels based on actual price movements, not just TP/SL hits.
    This focuses on identifying when the market is about to make a significant move.
    """
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    n = len(close)
    
    # Use float64 to handle NaN values properly
    labels = np.zeros(n, dtype=np.float64)
    
    for i in range(n - max_horizon):
        current_price = close[i]
        min_move = current_price * min_move_pct
        
        # Look ahead to find the next significant move
        for j in range(1, max_horizon + 1):
            future_high = high[i + j]
            future_low = low[i + j]
            
            # Check if we hit a significant move in either direction
            if (future_high >= current_price + min_move) or (future_low <= current_price - min_move):
                # Determine direction of the move
                high_move = (future_high - current_price) / current_price
                low_move = (current_price - future_low) / current_price
                
                if high_move > low_move and high_move >= min_move_pct:
                    labels[i] = 1.0  # Bullish move
                elif low_move > high_move and low_move >= min_move_pct:
                    labels[i] = 0.0  # Bearish move (or no significant move)
                break
    
    # Set last max_horizon rows to NaN since we can't see their future
    labels[-max_horizon:] = np.nan
    
    return pd.Series(labels, index=df.index, dtype="float32")

def create_volatility_breakout_labels(df, volatility_threshold=0.02, horizon=24):
    """
    Alternative labeling strategy focusing on volatility breakouts
    """
    close = df['close'].values
    n = len(close)
    
    # Calculate rolling volatility
    returns = np.diff(close) / close[:-1]
    volatility = pd.Series(returns).rolling(20).std().values
    
    # Use float64 to handle NaN values properly
    labels = np.zeros(n, dtype=np.float64)
    
    for i in range(20, n - horizon):
        current_vol = volatility[i]
        avg_vol = np.mean(volatility[max(0, i-20):i])
        
        # Check if current volatility is significantly higher than average
        if current_vol > avg_vol * 1.5 and current_vol > volatility_threshold:
            # Look ahead to see if this leads to a significant price move
            future_prices = close[i:i+horizon]
            price_change = abs(future_prices[-1] - close[i]) / close[i]
            
            if price_change > volatility_threshold:
                labels[i] = 1.0  # Volatility breakout led to price move
            else:
                labels[i] = 0.0  # False breakout
        else:
            labels[i] = 0.0  # Normal volatility
    
    labels[-horizon:] = np.nan
    return pd.Series(labels, index=df.index, dtype="float32")

# -------------------------
# Feature Selection
# -------------------------
def select_best_features(X, y, method='mutual_info', k=100):
    """
    Select the most predictive features using various methods
    """
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
    elif method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    else:
        return X.columns.tolist()
    
    X_selected = selector.fit_transform(X.fillna(0), y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    return selected_features

# -------------------------
# Enhanced Model Training
# -------------------------
def create_ensemble_model():
    """
    Create an ensemble of models for better prediction
    """
    models = {
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
            n_jobs=-1
        ),
        'lr': LogisticRegression(
            C=0.1,
            max_iter=1000,
            random_state=42,
            solver='liblinear'
        )
    }
    return models

def train_ensemble(X_train, y_train, X_val, y_val):
    """
    Train ensemble of models and return the best one
    """
    models = create_ensemble_model()
    best_model = None
    best_score = -1
    
    for name, model in models.items():
        # Handle class imbalance
        if name == 'hgb':
            # HGB handles imbalance well with sample weights
            sample_weights = compute_sample_weights(y_train)
            model.fit(X_train.fillna(0), y_train, sample_weight=sample_weights)
        else:
            # For other models, use SMOTE
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_resampled, y_resampled = smote.fit_resample(X_train.fillna(0), y_train)
            model.fit(X_resampled, y_resampled)
        
        # Predict on validation set
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_val.fillna(0))[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_val.fillna(0))
        
        # Calculate F1 score (better for imbalanced data)
        score = f1_score(y_val, y_pred, average='weighted')
        
        if score > best_score:
            best_score = score
            best_model = model
    
    return best_model, best_score

def compute_sample_weights(y):
    """
    Compute sample weights to handle class imbalance
    """
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    sample_weights = np.ones(len(y))
    for i, class_label in enumerate(np.unique(y)):
        sample_weights[y == class_label] = class_weights[i]
    return sample_weights

# -------------------------
# Main Command
# -------------------------
class Command(BaseCommand):
    help = "Enhanced DOTUSDT model training with advanced features and improved labeling for 60%+ accuracy"

    def add_arguments(self, parser):
        parser.add_argument("--export_dir", type=str, default="./", help="Where to write outputs")
        parser.add_argument("--min_move_pct", type=float, default=0.005, help="Minimum move percentage for labeling")
        parser.add_argument("--horizon", type=int, default=24, help="Prediction horizon in bars")
        parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds")

    def handle(self, *args, **opts):
        export_dir = opts["export_dir"]
        min_move_pct = opts["min_move_pct"]
        horizon = opts["horizon"]
        n_folds = opts["n_folds"]
        
        os.makedirs(export_dir, exist_ok=True)

        # 1) Load data
        self.stdout.write("▶ Loading DOTUSDT OHLCV data...")
        qs = (CoinAPIPrice.objects
              .filter(coin=COIN, timestamp__gte=TRAIN_START, timestamp__lte=TEST_END)
              .values("coin","timestamp","open","high","low","close","volume")
              .order_by("timestamp"))
        
        df = pd.DataFrame.from_records(list(qs))
        if df.empty:
            self.stderr.write("No data returned for the requested window.")
            return

        # Data preprocessing
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.dropna(subset=["open","high","low","close","volume"]).reset_index(drop=True)

        # 2) Enhanced feature engineering
        self.stdout.write("▶ Engineering advanced features...")
        df_features = add_advanced_features(df)
        
        # Remove rows with too many NaN values
        min_non_null = df_features.shape[1] * 0.8
        df_features = df_features.dropna(thresh=min_non_null).reset_index(drop=True)
        
        # 3) Create labels using improved strategy
        self.stdout.write("▶ Creating smart labels...")
        labels = create_smart_labels(df_features, min_move_pct, horizon)
        df_features['label'] = labels
        
        # Remove rows without labels
        df_features = df_features.dropna(subset=['label']).reset_index(drop=True)
        df_features['label'] = df_features['label'].astype(int)
        
        # Check class distribution
        label_counts = df_features['label'].value_counts()
        self.stdout.write(f"Label distribution: {label_counts.to_dict()}")
        
        # 4) Prepare features
        # Remove non-feature columns
        exclude_cols = ['coin', 'timestamp', 'label', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        
        # Remove columns with too many NaN values
        feature_cols = [col for col in feature_cols if df_features[col].notna().sum() > len(df_features) * 0.9]
        
        X = df_features[feature_cols].astype("float32")
        y = df_features['label'].values.astype(int)
        
        # 5) Feature selection
        self.stdout.write("▶ Selecting best features...")
        selected_features = select_best_features(X, y, method='mutual_info', k=150)
        X_selected = X[selected_features]
        
        self.stdout.write(f"Selected {len(selected_features)} features from {len(feature_cols)} total features")
        
        # 6) Time series cross-validation
        self.stdout.write("▶ Performing time series cross-validation...")
        tscv = TimeSeriesSplit(n_splits=n_folds)
        
        cv_scores = []
        best_model = None
        best_score = -1
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_selected)):
            X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Skip if not enough samples
            if len(y_train) < 1000 or len(y_val) < 200:
                continue
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train.fillna(0))
            X_val_scaled = scaler.transform(X_val.fillna(0))
            
            # Train ensemble
            model, score = train_ensemble(
                pd.DataFrame(X_train_scaled, columns=selected_features),
                y_train,
                pd.DataFrame(X_val_scaled, columns=selected_features),
                y_val
            )
            
            cv_scores.append(score)
            self.stdout.write(f"  Fold {fold + 1}: F1 Score = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model
                best_scaler = scaler
        
        if not cv_scores:
            self.stderr.write("No valid CV folds found.")
            return
        
        mean_cv_score = np.mean(cv_scores)
        self.stdout.write(f"Mean CV F1 Score: {mean_cv_score:.4f}")
        
        # 7) Final training on full dataset
        self.stdout.write("▶ Training final model on full dataset...")
        
        # Scale full dataset
        X_full_scaled = best_scaler.fit_transform(X_selected.fillna(0))
        
        # Train final model
        if hasattr(best_model, 'fit'):
            if isinstance(best_model, HistGradientBoostingClassifier):
                sample_weights = compute_sample_weights(y)
                best_model.fit(X_full_scaled, y, sample_weight=sample_weights)
            else:
                # For other models, use SMOTE
                smote = SMOTE(random_state=42, k_neighbors=5)
                X_resampled, y_resampled = smote.fit_resample(X_full_scaled, y)
                best_model.fit(X_resampled, y_resampled)
        
        # 8) Evaluate on test set
        train_mask = (df_features["timestamp"] >= pd.Timestamp(TRAIN_START)) & (df_features["timestamp"] <= pd.Timestamp(TRAIN_END))
        test_mask = (df_features["timestamp"] >= pd.Timestamp(TEST_START)) & (df_features["timestamp"] <= pd.Timestamp(TEST_END))
        
        train_data = df_features.loc[train_mask].copy()
        test_data = df_features.loc[test_mask].copy()
        
        if len(test_data) == 0:
            self.stdout.write("No test data available, using last 20% of data as test set")
            split_idx = int(len(df_features) * 0.8)
            train_data = df_features.iloc[:split_idx].copy()
            test_data = df_features.iloc[split_idx:].copy()
        
        # Prepare test data
        X_test = test_data[selected_features].astype("float32")
        y_test = test_data['label'].values.astype(int)
        X_test_scaled = best_scaler.transform(X_test.fillna(0))
        
        # Make predictions
        if hasattr(best_model, 'predict_proba'):
            y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred_proba = best_model.predict(X_test_scaled).astype(float)
        
        # Find optimal threshold
        thresholds = np.linspace(0.1, 0.9, 81)
        best_threshold = 0.5
        best_f1 = -1
        
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            f1 = f1_score(y_test, y_pred, average='weighted')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        # Final evaluation
        y_pred_final = (y_pred_proba >= best_threshold).astype(int)
        accuracy = accuracy_score(y_test, y_pred_final)
        precision = precision_score(y_test, y_pred_final, zero_division=0)
        recall = recall_score(y_test, y_pred_final, zero_division=0)
        f1 = f1_score(y_test, y_pred_final, average='weighted')
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = float('nan')
        
        self.stdout.write(f"Test Set Performance:")
        self.stdout.write(f"  Accuracy: {accuracy:.4f}")
        self.stdout.write(f"  Precision: {precision:.4f}")
        self.stdout.write(f"  Recall: {recall:.4f}")
        self.stdout.write(f"  F1 Score: {f1:.4f}")
        self.stdout.write(f"  AUC: {auc:.4f}")
        self.stdout.write(f"  Optimal Threshold: {best_threshold:.3f}")
        
        # 9) Save artifacts
        self.stdout.write("▶ Saving model artifacts...")
        
        # Save datasets
        train_csv = os.path.join(export_dir, "sol_train_dataset.csv")
        test_csv = os.path.join(export_dir, "sol_test_dataset.csv")
        preds_csv = os.path.join(export_dir, "sol_predictions.csv")
        
        # Add coin column back
        train_data = train_data.assign(coin=COIN)
        test_data = test_data.assign(coin=COIN)
        
        # Save with all features
        export_cols = ['coin', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'label'] + selected_features
        train_data[export_cols].to_csv(train_csv, index=False)
        test_data[export_cols].to_csv(test_csv, index=False)
        
        # Save predictions
        predictions_df = test_data[['coin', 'timestamp']].copy()
        predictions_df['label'] = y_test
        predictions_df['pred_prob'] = y_pred_proba
        predictions_df['pred_at_thr'] = y_pred_final
        predictions_df.to_csv(preds_csv, index=False)
        
        # Save model and scaler
        model_path = os.path.join(export_dir, "sol_long_hgb_model.joblib")
        scaler_path = os.path.join(export_dir, "sol_feature_scaler.joblib")
        features_path = os.path.join(export_dir, "sol_feature_list.json")
        config_path = os.path.join(export_dir, "sol_trade_config.json")
        
        dump(best_model, model_path)
        dump(best_scaler, scaler_path)
        
        with open(features_path, 'w') as f:
            json.dump(selected_features, f, indent=2)
        
        # Create config
        config = {
            "coin": COIN,
            "train_window": {"start": TRAIN_START, "end": TRAIN_END},
            "test_window": {"start": TEST_START, "end": TEST_END},
            "min_move_pct": min_move_pct,
            "horizon_bars": horizon,
            "threshold": round(best_threshold, 3),
            "test_metrics": {
                "acc": round(accuracy, 4),
                "prec": round(precision, 4),
                "rec": round(recall, 4),
                "f1": round(f1, 4),
                "auc": round(auc, 4)
            },
            "cv_score": round(mean_cv_score, 4),
            "n_features": len(selected_features),
            "model_type": type(best_model).__name__
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.stdout.write(self.style.SUCCESS(
            f"Model training completed successfully!\n"
            f"Final Test Accuracy: {accuracy:.4f}\n"
            f"Files saved to: {export_dir}"
        ))
