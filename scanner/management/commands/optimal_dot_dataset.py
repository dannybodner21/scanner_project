# Optimal DOTUSDT ML model: 20-30 highest-quality features
# Research-backed feature selection for maximum profitability

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

# Core technical analysis functions
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

def vwap(close, high, low, volume, window=20):
    tp = (high + low + close) / 3.0
    pv = tp * volume
    return pv.rolling(window).sum() / (volume.rolling(window).sum() + 1e-12)

def money_flow_index(high, low, close, volume, period=14):
    """Research-proven volume-price indicator"""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    
    positive_flow = pd.Series(0.0, index=close.index)
    negative_flow = pd.Series(0.0, index=close.index)
    
    price_changes = typical_price.diff()
    positive_flow = pd.where(price_changes > 0, raw_money_flow, 0)
    negative_flow = pd.where(price_changes < 0, raw_money_flow, 0)
    
    positive_mf = pd.Series(positive_flow).rolling(period).sum()
    negative_mf = pd.Series(negative_flow).rolling(period).sum()
    
    return 100 - (100 / (1 + positive_mf / (negative_mf + 1e-12)))

def chaikin_money_flow(high, low, close, volume, period=20):
    """Accumulation/Distribution pressure indicator"""
    mfm = ((close - low) - (high - close)) / (high - low + 1e-12)
    mfv = mfm * volume
    return mfv.rolling(period).sum() / (volume.rolling(period).sum() + 1e-12)

def parkinson_volatility(high, low):
    """Superior volatility estimator using intraday range"""
    return np.sqrt((np.log(high / low) ** 2) / (4 * np.log(2)))

def compute_optimal_features(df):
    """Curated 25-30 highest-impact features from research"""
    g = df.copy()
    g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True)
    g = g.sort_values("timestamp").reset_index(drop=True)
    
    F = {}
    vol = pd.to_numeric(g['volume'], errors='coerce').fillna(0.0)
    
    # === CORE PRICE ACTION (5 features) ===
    F['price_range'] = (g['high'] - g['low']) / (g['close'] + 1e-12)
    F['body_size'] = (g['close'] - g['open']).abs() / (g['close'] + 1e-12)
    F['close_position'] = (g['close'] - g['low']) / (g['high'] - g['low'] + 1e-12)  # Where close sits in range
    
    # Key momentum returns (research-proven periods)
    F['ret_1'] = g['close'].pct_change()
    F['ret_5'] = g['close'].pct_change(5)
    
    # === MOMENTUM & TREND (8 features) ===
    # EMA slopes (strongest trend indicators)
    ema_21 = ema(g['close'], 21)
    ema_55 = ema(g['close'], 55)
    F['ema_21'] = ema_21
    F['ema_21_slope'] = ema_21.diff()
    F['ema_55_slope'] = ema_55.diff()
    F['close_vs_ema_21'] = (g['close'] - ema_21) / (ema_21 + 1e-12)
    
    # RSI (optimized single period)
    rsi_14 = rsi(g['close'], 14)
    F['rsi_14'] = rsi_14
    F['rsi_14_slope'] = rsi_14.diff()
    
    # MACD histogram slope (momentum change)
    _, _, macd_hist = macd(g['close'])
    F['macd_hist_slope'] = macd_hist.diff()
    
    # === VOLATILITY REGIME (3 features) ===
    vol_20 = g['close'].pct_change().rolling(20).std()
    F['volatility_20'] = vol_20
    F['vol_percentile'] = vol_20.rolling(100).rank(pct=True)  # Critical: vol regime detection
    F['parkinson_vol'] = parkinson_volatility(g['high'], g['low'])  # Superior vol estimator
    
    # === VOLUME ANALYSIS (5 features) ===
    vol_ma_20 = vol.rolling(20).mean()
    F['rel_volume'] = vol / (vol_ma_20 + 1e-12)  # Volume surge detection
    F['mfi'] = money_flow_index(g['high'], g['low'], g['close'], vol)  # Research-proven
    F['cmf'] = chaikin_money_flow(g['high'], g['low'], g['close'], vol)  # Accumulation pressure
    
    # On-Balance Volume slope (volume-price momentum)
    dirn = np.sign(g['close'].diff()).fillna(0)
    obv = (vol * dirn).cumsum()
    F['obv_slope'] = obv.diff()
    
    # VWAP deviation (institutional trading reference)
    vwap_20 = vwap(g['close'], g['high'], g['low'], vol, 20)
    F['vwap_deviation'] = (g['close'] - vwap_20) / (vwap_20 + 1e-12)
    
    # === CRYPTOQUANT-INSPIRED ON-CHAIN SIMULATION (4 features) ===
    # Simulated MVRV: Current price vs volume-weighted realized price
    vwap_100 = vwap(g['close'], g['high'], g['low'], vol, 100)
    F['mvrv_sim'] = g['close'] / (vwap_100 + 1e-12)
    
    # Volume flow regime (exchange flow simulation)
    vol_ma_50 = vol.rolling(50).mean()
    F['vol_flow_ratio'] = vol_ma_20 / (vol_ma_50 + 1e-12)
    
    # Order flow imbalance simulation
    buying_pressure = F['close_position'] * vol
    selling_pressure = (1 - F['close_position']) * vol
    F['order_flow_imbalance'] = (buying_pressure - selling_pressure) / (vol + 1e-12)
    
    # Network value simulation (NVT-inspired)
    network_value = g['close'] * vol
    F['nvt_sim'] = network_value / (vol_ma_20 + 1e-12)
    
    # === BOLLINGER BANDS (2 features) ===
    bb_u, bb_m, bb_l, bb_w, bb_std = bollinger(g['close'], 20, 2.0)
    F['bb_position'] = (g['close'] - bb_l) / (bb_u - bb_l + 1e-12)  # Position in bands
    F['bb_zscore'] = (g['close'] - bb_m) / (bb_std + 1e-12)  # Standardized distance
    
    # === TIME FEATURES (3 features) ===
    hour = g['timestamp'].dt.hour
    F['hour_sin'] = np.sin(2*np.pi*hour/24)  # Cyclical encoding
    F['hour_cos'] = np.cos(2*np.pi*hour/24)
    F['is_us_hours'] = ((hour >= 13) & (hour <= 21)).astype(int)  # US trading session
    
    # Convert to DataFrame
    feat_df = pd.DataFrame(F, index=g.index)
    
    # Add cross-validation features (momentum rank)
    feat_df['momentum_rank'] = feat_df['ret_5'].rolling(50).rank(pct=True)
    
    # Combine with original data
    g = pd.concat([g, feat_df], axis=1)
    g.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows with critical NaN values
    core_cols = ["ema_21", "rsi_14", "volatility_20", "rel_volume", "mfi", "vwap_deviation"]
    g = g.dropna(subset=core_cols)
    return g.reset_index(drop=True)

def create_smart_labels(df, min_move_pct=0.025, max_horizon=24):
    """Optimized labeling: TP before SL"""
    close = df['close'].values
    high  = df['high'].values
    low   = df['low'].values
    n = len(close)
    labels = np.full(n, np.nan, dtype=np.float64)

    for i in range(n - max_horizon):
        current = close[i]
        tp_level = current * (1 + min_move_pct)
        sl_level = current * (1 - min_move_pct * 0.6)  # Asymmetric: 2.5% TP, 1.5% SL
        
        for j in range(1, max_horizon + 1):
            if high[i + j] >= tp_level:
                labels[i] = 1.0  # Win
                break
            elif low[i + j] <= sl_level:
                labels[i] = 0.0  # Loss
                break
    
    labels[-max_horizon:] = np.nan
    return pd.Series(labels.astype("float32"), index=df.index)

def build_models():
    """Research-optimized models for crypto"""
    return {
        "hgb": HistGradientBoostingClassifier(
            max_depth=4, max_iter=200, learning_rate=0.08,
            l2_regularization=0.2, min_samples_leaf=50,
            n_iter_no_change=15, random_state=42
        ),
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=8,
            min_samples_split=80, min_samples_leaf=30,
            max_features=0.6, n_jobs=-1, random_state=42, 
            class_weight="balanced"
        ),
        "lr": LogisticRegression(
            C=0.5, max_iter=1000, solver="liblinear",
            class_weight="balanced", random_state=42
        ),
    }

def compute_sample_weights(y):
    classes = np.unique(y)
    w = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    mapping = {c: w[i] for i, c in enumerate(classes)}
    return np.array([mapping[v] for v in y], dtype=np.float32)

def pick_threshold(y_true, y_proba, metric="f1"):
    thresholds = np.linspace(0.4, 0.8, 41)  # Optimized range for crypto
    best_t, best = 0.5, -1.0
    for t in thresholds:
        yp = (y_proba >= t).astype(int)
        if yp.sum() < 10:  # Need minimum trades
            continue
        if metric == "f1":
            score = f1_score(y_true, yp, average='binary', zero_division=0)
        else:
            score = average_precision_score(y_true, y_proba)
        if score > best:
            best, best_t = score, t
    return float(best_t), float(best)

class Command(BaseCommand):
    help = "Optimal 25-30 feature DOT model based on quantitative research"

    def add_arguments(self, parser):
        parser.add_argument("--coin", type=str, default="DOTUSDT")
        parser.add_argument("--export_dir", type=str, default=".")
        parser.add_argument("--train_start", type=str, default="2023-01-01 00:00:00+00:00")
        parser.add_argument("--train_end", type=str, default="2025-03-31 23:55:00+00:00")
        parser.add_argument("--test_start", type=str, default="2025-04-01 00:00:00+00:00")
        parser.add_argument("--test_end", type=str, default="2025-08-18 23:55:00+00:00")
        parser.add_argument("--min_move_pct", type=float, default=0.025)
        parser.add_argument("--horizon", type=int, default=24)
        parser.add_argument("--k_features", type=int, default=25)
        parser.add_argument("--val_frac", type=float, default=0.2)

    def handle(self, *args, **opts):
        COIN = opts["coin"].upper()
        out_dir = opts["export_dir"]
        os.makedirs(out_dir, exist_ok=True)
        
        self.stdout.write(f"🎯 Training OPTIMAL {COIN} model (25-30 research-backed features)")

        # Load data
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

        self.stdout.write("🔧 Computing optimal features...")
        feat = compute_optimal_features(df)
        
        # Count actual features
        exclude = {'coin','timestamp','open','high','low','close','volume'}
        actual_features = [c for c in feat.columns if c not in exclude]
        self.stdout.write(f"✅ Generated {len(actual_features)} optimal features")

        self.stdout.write("🎯 Creating smart labels...")
        labels = create_smart_labels(feat, opts["min_move_pct"], opts["horizon"])
        feat["label"] = labels
        feat = feat.dropna(subset=["label"]).reset_index(drop=True)
        feat["label"] = feat["label"].astype(int)

        # Train/Test split
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

        # Feature columns (all are carefully selected, no further filtering needed)
        exclude = {'coin','timestamp','open','high','low','close','volume','label'}
        cols = [c for c in train_df.columns if c not in exclude]
        
        # Only keep columns with sufficient data
        cols = [c for c in cols if train_df[c].notna().mean() >= 0.95]
        self.stdout.write(f"📊 Using {len(cols)} features (optimal set)")

        # Inner split for model selection
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

        # Scale features
        scaler = RobustScaler()
        X_inner_scaled = pd.DataFrame(
            scaler.fit_transform(X_inner),
            columns=cols, index=X_inner.index
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=cols, index=X_val.index
        )

        # Model selection
        models = build_models()
        best_name, best_model, best_f1_val, best_thr = None, None, -1.0, 0.5
        sw_inner = compute_sample_weights(y_inner)

        self.stdout.write("🔄 Model selection on validation set...")
        for name, mdl in models.items():
            if isinstance(mdl, HistGradientBoostingClassifier):
                mdl.fit(X_inner_scaled, y_inner, sample_weight=sw_inner)
            else:
                mdl.fit(X_inner_scaled, y_inner)

            if hasattr(mdl, "predict_proba"):
                proba_val = mdl.predict_proba(X_val_scaled)[:, 1]
            else:
                raw = mdl.decision_function(X_val_scaled)
                proba_val = 1.0 / (1.0 + np.exp(-raw))

            thr, f1_at_thr = pick_threshold(y_val, proba_val, metric="f1")
            
            # Calculate hit rate at threshold
            pred_val = (proba_val >= thr).astype(int)
            hit_rate = precision_score(y_val, pred_val, zero_division=0) if pred_val.sum() > 0 else 0
            
            self.stdout.write(f"   {name}: F1={f1_at_thr:.3f} | Hit Rate={hit_rate:.3f} | Trades={pred_val.sum()} | Thr={thr:.2f}")
            
            if f1_at_thr > best_f1_val:
                best_f1_val = f1_at_thr
                best_name = name
                best_model = mdl
                best_thr = thr

        if best_model is None:
            self.stderr.write("Model selection failed.")
            return

        self.stdout.write(f"🏆 Best model: {best_name} | F1={best_f1_val:.4f} | Threshold={best_thr:.3f}")

        # Refit on full training data
        X_train_full = train_df[cols].astype("float32").fillna(0.0)
        y_train_full = train_df["label"].astype(int).values

        scaler_full = RobustScaler()
        X_train_full_scaled = pd.DataFrame(
            scaler_full.fit_transform(X_train_full),
            columns=cols, index=X_train_full.index
        )

        # Final model
        mdl_final = type(best_model)(**best_model.get_params())
        sw_full = compute_sample_weights(y_train_full)
        if isinstance(mdl_final, HistGradientBoostingClassifier):
            mdl_final.fit(X_train_full_scaled, y_train_full, sample_weight=sw_full)
        else:
            mdl_final.fit(X_train_full_scaled, y_train_full)

        # Test evaluation
        X_test = test_df[cols].astype("float32").fillna(0.0)
        X_test_scaled = pd.DataFrame(
            scaler_full.transform(X_test),
            columns=cols, index=X_test.index
        )
        y_test = test_df["label"].astype(int).values

        if hasattr(mdl_final, "predict_proba"):
            proba_test = mdl_final.predict_proba(X_test_scaled)[:, 1]
        else:
            raw = mdl_final.decision_function(X_test_scaled)
            proba_test = 1.0 / (1.0 + np.exp(-raw))

        y_pred_test = (proba_test >= best_thr).astype(int)

        # Metrics
        acc  = accuracy_score(y_test, y_pred_test)
        prec = precision_score(y_test, y_pred_test, zero_division=0)
        rec  = recall_score(y_test, y_pred_test, zero_division=0)
        f1   = f1_score(y_test, y_pred_test, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_test, proba_test)
        except:
            auc = float('nan')
        cm = confusion_matrix(y_test, y_pred_test).tolist()
        trades = y_pred_test.sum()

        self.stdout.write("📊 TEST RESULTS (unseen data):")
        self.stdout.write(f"   Hit Rate (Precision): {prec:.1%}")
        self.stdout.write(f"   Accuracy: {acc:.1%} | Recall: {rec:.1%} | F1: {f1:.3f}")
        self.stdout.write(f"   AUC: {auc:.3f} | Trades: {trades}")
        self.stdout.write(f"   Confusion Matrix: {cm}")

        # Save artifacts
        prefix = COIN.split("USDT")[0].lower()
        model_path   = os.path.join(out_dir, f"{prefix}_optimal_model.joblib")
        scaler_path  = os.path.join(out_dir, f"{prefix}_optimal_scaler.joblib")
        feats_path   = os.path.join(out_dir, f"{prefix}_optimal_features.json")
        config_path  = os.path.join(out_dir, f"{prefix}_optimal_config.json")
        preds_path   = os.path.join(out_dir, f"{prefix}_optimal_predictions.csv")

        dump(mdl_final, model_path)
        dump(scaler_full, scaler_path)
        with open(feats_path, "w") as f:
            json.dump(cols, f, indent=2)

        cfg = {
            "coin": COIN,
            "model_type": type(mdl_final).__name__,
            "train_window": {"start": str(TRAIN_START), "end": str(TRAIN_END)},
            "test_window": {"start": str(TEST_START), "end": str(TEST_END)},
            "min_move_pct": float(opts["min_move_pct"]),
            "horizon_bars": int(opts["horizon"]),
            "threshold": round(float(best_thr), 3),
            "test_metrics": {
                "hit_rate": round(float(prec), 4),
                "accuracy": round(float(acc), 4),
                "recall": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "auc": round(float(auc), 4) if not math.isnan(auc) else None,
                "trades": int(trades),
                "confusion": cm
            },
            "n_features": len(cols),
            "feature_list": cols
        }
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        # Save predictions
        pd.DataFrame({
            "coin": COIN,
            "timestamp": test_df["timestamp"].values,
            "pred_prob": proba_test
        }).to_csv(preds_path, index=False)

        self.stdout.write(self.style.SUCCESS(
            f"✅ OPTIMAL MODEL COMPLETE!\n"
            f"   Features: {len(cols)} research-backed indicators\n"
            f"   Hit Rate: {prec:.1%} | Trades: {trades}\n"
            f"   Model: {model_path}\n"
            f"   Predictions: {preds_path}"
        ))