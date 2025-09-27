# scanner/management/commands/short_dataset_simple.py
# SIMPLE and EFFECTIVE short trading model - focus on what actually works
# 
# Key principles:
# 1. Fewer, better features (15-20 max)
# 2. Simple, proven technical indicators
# 3. Clear short trade logic
# 4. Conservative thresholds
# 5. Focus on trend following, not prediction

from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice

import os, json, math, warnings
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    f1_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight



# XRPUSDT | threshold 0.6 | 48% accurate | 299 trades | 1k -> 9M | sl=1, tp=2

# ADAUSDT | threshold 0.55 | 43% accurate | 301 trades | 1k -> 14k | sl=1, tp=2
# ATOMUSDT | threshold 0.5 | 49% accurate | 232 trades | 1k -> 3M | sl=1, tp=2
# AVAXUSDT | threshold 0.5 | 47% accurate | 267 trades | 1k -> 1M | sl=1, tp=2
# BTCUSDT | threshold 0.5 | 58% accurate | 43 trades | 1k -> 23k | sl=1, tp=2
# DOGEUSDT | threshold 0.5 | 47% accurate | 354 trades | 1k -> 16M | sl=1, tp=2
# DOTUSDT | threshold 0.5 | 50% accurate | 247 trades | 1k -> 25M | sl=1, tp=2
# ETHUSDT | threshold 0.5 | 54% accurate | 147 trades | 1k -> 4M | sl=1, tp=2
# LINKUSDT | threshold 0.5 | 43% accurate | 349 trades | 1k -> 60k | sl=1, tp=2
# LTCUSDT | threshold 0.5 | 44% accurate | 270 trades | 1k -> 74k | sl=1, tp=2
# SHIBUSDT | threshold 0.5 | 48% accurate | 242 trades | 1k -> 3M | sl=1, tp=2
# SOLUSDT | threshold 0.5 | 50% accurate | 320 trades | 1k -> 500M | sl=1, tp=2
# UNIUSDT | threshold 0.5 | 43% accurate | 390 trades | 1k -> 26k | sl=1, tp=2
# TRXUSDT | threshold 0.5 | 75% accurate | 24 trades | 1k -> 34k | sl=1, tp=2
# XLMUSDT | threshold 0.6 | 45% accurate | 42 trades | 1k -> 2k | sl=1, tp=2



warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)

def rsi(close, period=14):
    """Simple RSI calculation"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def compute_simple_features(df):
    """
    SIMPLE features that actually work for short trading
    Focus: Trend following, momentum, volatility
    """
    g = df.copy()
    g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True)
    g = g.sort_values("timestamp").reset_index(drop=True)

    F = {}
    
    # 1. TREND FEATURES (4 features)
    F['ema_20'] = g['close'].ewm(span=20).mean()
    F['ema_50'] = g['close'].ewm(span=50).mean()
    F['ema_200'] = g['close'].ewm(span=200).mean()
    F['trend_strength'] = (F['ema_20'] - F['ema_200']) / F['ema_200']  # Strong trend indicator
    
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
    F['bb_squeeze'] = (bb_std < bb_std.rolling(50).quantile(0.25)).astype(int)  # Low vol
    
    # 6. VOLUME FEATURES (2 features)
    vol = pd.to_numeric(g['volume'], errors='coerce').fillna(0.0)
    F['vol_sma_20'] = vol.rolling(20).mean()
    F['rel_vol'] = vol / F['vol_sma_20']
    
    # 7. SHORT-SPECIFIC SIGNALS (3 features)
    F['price_below_ema_20'] = (g['close'] < F['ema_20']).astype(int)
    F['ema_trend_down'] = (F['ema_20'] < F['ema_50']).astype(int)
    F['bearish_momentum'] = ((F['ret_5'] < 0) & (F['ret_20'] < 0)).astype(int)  # Both timeframes down
    
    # TOTAL: 18 features - simple and focused
    
    feat_df = pd.DataFrame(F, index=g.index)
    g = pd.concat([g, feat_df], axis=1)
    g.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Core columns for prediction
    core_cols = ["ema_20", "ema_50", "rsi_14", "bb_position", "vol_sma_20"]
    g = g.dropna(subset=core_cols)
    return g.reset_index(drop=True)

def create_simple_short_labels(df, move_threshold=0.02, max_lookahead=24):
    """
    SIMPLE short trade labeling - conservative and clear
    - Win (1): Price drops by threshold within lookahead
    - Loss (0): Price rises by threshold within lookahead
    - No trade: Neither threshold hit
    
    Conservative approach: smaller moves, shorter lookahead
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
            
            # Simple logic: which threshold hit first
            if future_close <= down_threshold:
                labels[i] = 1.0  # Down move = short wins
                break
            elif future_close >= up_threshold:
                labels[i] = 0.0  # Up move = short loses
                break
    
    # Set trailing NaNs to prevent look-ahead bias
    labels[-max_lookahead:] = np.nan
    return pd.Series(labels.astype("float32"), index=df.index)

def build_simple_models():
    """Simple, effective models for short trading"""
    return {
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=8,
            min_samples_split=50, min_samples_leaf=25,
            max_features='sqrt', bootstrap=True,
            n_jobs=-1, random_state=42, class_weight="balanced"
        )
    }

class Command(BaseCommand):
    help = "SIMPLE and EFFECTIVE short trading model - focus on what actually works"

    def add_arguments(self, parser):
        parser.add_argument("--coin", type=str, default="DOTUSDT", help="e.g., DOTUSDT / LINKUSDT / UNIUSDT")
        parser.add_argument("--export_dir", type=str, default=".", help="Where to write outputs")
        parser.add_argument("--train_start", type=str, default="2023-01-01 00:00:00+00:00")
        parser.add_argument("--train_end",   type=str, default="2024-12-31 23:55:00+00:00")
        parser.add_argument("--test_start",  type=str, default="2025-01-01 00:00:00+00:00")
        parser.add_argument("--test_end",    type=str, default="2025-09-02 23:55:00+00:00")
        parser.add_argument("--move_threshold", type=float, default=0.015, help="Move threshold (1.5% default)")
        parser.add_argument("--max_lookahead", type=int, default=24, help="Max bars to look ahead (24 = 2 hours)")
        parser.add_argument("--k_features", type=int, default=15, help="Number of features to select (keep small)")

    def handle(self, *args, **opts):
        COIN = opts["coin"].upper()
        out_dir = opts["export_dir"]
        os.makedirs(out_dir, exist_ok=True)
        
        self.stdout.write(f"🚀 SIMPLE SHORT TRADING MODEL for {COIN}")
        self.stdout.write(f"📊 Strategy: Conservative trend following with clear signals")
        self.stdout.write(f"🎯 Move threshold: {opts['move_threshold']*100:.1f}% | Lookahead: {opts['max_lookahead']} bars")

        # Load data
        self.stdout.write("▶ Loading data...")
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

        self.stdout.write("▶ Computing simple features...")
        feat = compute_simple_features(df)

        self.stdout.write("▶ Creating simple short labels...")
        labels = create_simple_short_labels(feat, opts["move_threshold"], opts["max_lookahead"])
        feat["label"] = labels
        feat = feat.dropna(subset=["label"]).reset_index(drop=True)
        feat["label"] = feat["label"].astype(int)

        # Check label distribution
        label_counts = feat["label"].value_counts()
        self.stdout.write(f"▶ Label distribution: {label_counts.to_dict()}")

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

        # Feature selection
        exclude = {'coin','timestamp','open','high','low','close','volume','label'}
        cols = [c for c in train_df.columns if c not in exclude]
        cols = [c for c in cols if train_df[c].notna().mean() >= 0.95]  # Stricter NaN filter

        self.stdout.write(f"▶ Selected {len(cols)} features for training")

        # Simple train/validation split
        n_train = len(train_df)
        val_len = max(100, int(n_train * 0.15))  # Smaller validation set
        inner_train_df = train_df.iloc[:-val_len].copy()
        inner_val_df   = train_df.iloc[-val_len:].copy()

        X_inner = inner_train_df[cols].astype("float32").fillna(0.0)
        y_inner = inner_train_df["label"].astype(int).values
        X_val   = inner_val_df[cols].astype("float32").fillna(0.0)
        y_val   = inner_val_df["label"].astype(int).values

        # Simple feature selection
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        
        k = min(int(opts["k_features"]), X_inner.shape[1])
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X_inner, y_inner)
        
        selected_features = X_inner.columns[selector.get_support()].tolist()
        if not selected_features:
            self.stderr.write("No features selected.")
            return

        self.stdout.write(f"▶ Final features: {len(selected_features)}")
        self.stdout.write(f"  Selected: {', '.join(selected_features)}")

        # Scaling
        scaler = RobustScaler()
        X_inner_sel = pd.DataFrame(
            scaler.fit_transform(X_inner[selected_features]),
            columns=selected_features, index=X_inner.index
        )
        X_val_sel = pd.DataFrame(
            scaler.transform(X_val[selected_features]),
            columns=selected_features, index=X_val.index
        )

        # Model training
        models = build_simple_models()
        best_name, best_model, best_f1_val, best_thr = None, None, -1.0, 0.5

        self.stdout.write("▶ Training simple model...")
        for name, mdl in models.items():
            try:
                mdl.fit(X_inner_sel, y_inner)
                
                if hasattr(mdl, "predict_proba"):
                    proba_val = mdl.predict_proba(X_val_sel)[:, 1]
                else:
                    raw = mdl.decision_function(X_val_sel)
                    proba_val = 1.0 / (1.0 + np.exp(-raw))

                # Find best threshold
                thresholds = np.linspace(0.3, 0.7, 41)  # Conservative range
                best_t, best_score = 0.5, -1.0
                
                for t in thresholds:
                    yp = (proba_val >= t).astype(int)
                    score = f1_score(y_val, yp, average='binary', zero_division=0)
                    if score > best_score:
                        best_score, best_t = score, t

                if best_score > best_f1_val:
                    best_f1_val = best_score
                    best_name = name
                    best_model = mdl
                    best_thr = best_t
                    
                self.stdout.write(f"  {name}: F1={best_score:.4f} | thr={best_t:.3f}")
                
            except Exception as e:
                self.stdout.write(f"  {name}: Failed - {e}")
                continue

        if best_model is None:
            self.stderr.write("Model training failed.")
            return

        # Refit on full training data
        X_train_full = train_df[selected_features].astype("float32").fillna(0.0)
        y_train_full = train_df["label"].astype(int).values

        scaler_full = RobustScaler()
        X_train_full_scaled = pd.DataFrame(
            scaler_full.fit_transform(X_train_full),
            columns=selected_features, index=X_train_full.index
        )

        mdl_final = type(best_model)(**best_model.get_params())
        mdl_final.fit(X_train_full_scaled, y_train_full)

        # Final evaluation on test
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

        # Metrics
        acc  = accuracy_score(y_test, y_pred_test)
        prec = precision_score(y_test, y_pred_test, zero_division=0)
        rec  = recall_score(y_test, y_pred_test, zero_division=0)
        f1   = f1_score(y_test, y_pred_test, average="binary")
        try:
            auc = roc_auc_score(y_test, proba_test)
        except Exception:
            auc = float('nan')

        self.stdout.write("▶ TEST metrics:")
        self.stdout.write(f"  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

        # Save artifacts
        prefix = COIN.split("USDT")[0].lower()
        model_path   = os.path.join(out_dir, f"{prefix}_simple_short_rf_model.joblib")
        scaler_path  = os.path.join(out_dir, f"{prefix}_simple_short_scaler.joblib")
        feats_path   = os.path.join(out_dir, f"{prefix}_simple_short_features.json")
        config_path  = os.path.join(out_dir, f"{prefix}_simple_short_config.json")
        preds_csv    = os.path.join(out_dir, f"{prefix}_simple_short_predictions.csv")
        test_csv     = os.path.join(out_dir, f"{prefix}_simple_short_test_dataset.csv")

        dump(mdl_final, model_path)
        dump(scaler_full, scaler_path)
        
        with open(feats_path, "w") as f:
            json.dump(selected_features, f, indent=2)
        
        # Export test dataset for trade simulator baseline
        test_out = test_df.copy()
        test_out["coin"] = COIN
        export_cols = ['coin','timestamp','open','high','low','close','volume','label'] + selected_features
        test_out[export_cols].to_csv(test_csv, index=False)

        cfg = {
            "coin": COIN,
            "strategy": "SIMPLE short trading - conservative trend following",
            "move_threshold": float(opts["move_threshold"]),
            "max_lookahead": int(opts["max_lookahead"]),
            "threshold": round(float(best_thr), 3),
            "n_features": len(selected_features),
            "test_metrics": {
                "acc": round(float(acc), 4),
                "prec": round(float(prec), 4),
                "rec": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "auc": round(float(auc), 4) if not math.isnan(auc) else None
            }
        }
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        # Simplified predictions export with only necessary columns (matching long_dataset.py)
        predictions_df = pd.DataFrame({
            "coin": COIN,
            "timestamp": test_df["timestamp"].values,
            "pred_prob": proba_test,
            "confidence": np.abs(proba_test - 0.5) * 2,  # Confidence score
        })
        predictions_df.to_csv(preds_csv, index=False)

        self.stdout.write(self.style.SUCCESS(
            f"✅ SIMPLE SHORT MODEL COMPLETE\n"
            f"Model: {model_path}\n"
            f"Scaler: {scaler_path}\n"
            f"Features: {feats_path}\n"
            f"Config: {config_path}\n"
            f"Predictions: {preds_csv}\n"
            f"Test Dataset: {test_csv}"
        ))
        
        self.stdout.write(f"\n📊 SIMPLE MODEL SUMMARY:")
        self.stdout.write(f"  • Features: {len(selected_features)} (focused and proven)")
        self.stdout.write(f"  • Strategy: Conservative trend following")
        self.stdout.write(f"  • Move threshold: {opts['move_threshold']*100:.1f}% (conservative)")
        self.stdout.write(f"  • Lookahead: {opts['max_lookahead']} bars (2 hours)")
        self.stdout.write(f"  • Test F1: {f1:.4f} | Test AUC: {auc:.4f}")
        self.stdout.write(f"  • Key principle: Fewer, better features that actually work")
