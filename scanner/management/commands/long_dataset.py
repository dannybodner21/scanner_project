

# ETHUSDT | threshold 0.45 | 81% accurate | 22 trades | 1k -> 52k | sl=1, tp=2

# UNIUSDT | threshold 0.55 | 43% accurate | 131 trades | 1k -> 8k | sl=1, tp=2
# LINKUSDT | threshold 0.5 | 58% accurate | 74 trades | 1k -> 338kk | sl=1, tp=2
# LTCUSDT | threshold 0.5 | 47% accurate | 40 trades | 1k -> 3k | sl=1, tp=2
# AVAXUSDT | threshold 0.45 | 47% accurate | 94 trades | 1k -> 26k | sl=1, tp=2
# ADAUSDT | threshold 0.5 | 51% accurate | 47 trades | 1k -> 9k | sl=1, tp=2
# XLMUSDT | threshold 0.5 | 44% accurate | 107 trades | 1k -> 10k | sl=1, tp=2
# ATOMUSDT | threshold 0.5 | 49% accurate | 61 trades | 1k -> 11k | sl=1, tp=2
# BTCUSDT | threshold 0.3 | 57% accurate | 14 trades | 1k -> 2k | sl=1, tp=2
# TRXUSDT | threshold 0.3 | 100% accurate | 1 trade | 1k -> 1.2k | sl=1, tp=2



# XRPUSDT | threshold 0.8 | 41% accurate | 291 trades | 1k -> 61k | sl=1, tp=2


# DOTUSDT | threshold 0.65 | 0% accurate | 67 trades | 1k -> 3k | sl=1, tp=2
# SOLUSDT | threshold 0.5 | 0% accurate | 43 trades | 1k -> 3k | sl=1, tp=2
# SHIBUSDT | threshold 0.5 | 0% accurate | 61 trades | 1k -> 2k | sl=1, tp=2
# DOGEUSDT | threshold 0.5 | 0% accurate | 62 trades | 1k -> 10k | sl=1, tp=2

# scanner/management/commands/train_coin.py
from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice

import os, json, warnings, math
import numpy as np
import pandas as pd
from joblib import dump
from datetime import datetime

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 300)

# ===== Optional: LightGBM (preferred). Fallback = sklearn HGB =====
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_LGBM = False

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, balanced_accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight


# =========================
# Feature Engineering
# =========================
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()

def _rsi_ewm(series, n=14):
    d = series.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-12)
    return 100 - (100/(1+rs))

def _atr_like(df, n=14):
    # classic TR but cheaper; trailing/causal
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(),
                    (high-prev_close).abs(),
                    (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _bollinger_pos(df, n=20, k=2):
    m = df["close"].rolling(n).mean()
    s = df["close"].rolling(n).std(ddof=0)
    upper = m + k*s
    lower = m - k*s
    width = (upper - lower) / (m + 1e-12)
    pos = (df["close"] - m) / (s + 1e-12)
    return pos, width

def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy().sort_values("timestamp").reset_index(drop=True)

    # EMAs & slopes
    g["ema_9"] = _ema(g["close"], 9)
    g["ema_21"] = _ema(g["close"], 21)
    g["ema_50"] = _ema(g["close"], 50)
    g["ema_200"] = _ema(g["close"], 200)
    g["ema_9_slope3"] = g["ema_9"].diff(3)
    g["ema_21_slope3"] = g["ema_21"].diff(3)

    # MACD
    fast = g["close"].ewm(span=12, adjust=False).mean()
    slow = g["close"].ewm(span=26, adjust=False).mean()
    macd = fast - slow
    g["macd"] = macd
    g["macd_sig"] = macd.ewm(span=9, adjust=False).mean()
    g["macd_hist"] = g["macd"] - g["macd_sig"]

    # RSI
    g["rsi_14"] = _rsi_ewm(g["close"], 14)

    # Bollinger
    g["bb_pos"], g["bb_width"] = _bollinger_pos(g, 20, 2)

    # Volatility & returns
    g["ret_1"] = np.log(g["close"] / g["close"].shift(1))
    g["ret_3"] = np.log(g["close"] / g["close"].shift(3))
    g["ret_6"] = np.log(g["close"] / g["close"].shift(6))
    g["ret_12"] = np.log(g["close"] / g["close"].shift(12))
    g["vol_24"] = g["ret_1"].rolling(24).std()
    g["vol_72"] = g["ret_1"].rolling(72).std()

    # Candles
    g["body"] = (g["close"] - g["open"]) / (g["open"] + 1e-12)
    g["range"] = (g["high"] - g["low"]) / (g["open"] + 1e-12)
    g["upper_wick"] = (g["high"] - g[["open","close"]].max(axis=1)) / (g["open"] + 1e-12)
    g["lower_wick"] = (g[["open","close"]].min(axis=1) - g["low"]) / (g["open"] + 1e-12)

    # ATR-like
    g["atr_14"] = _atr_like(g, 14)
    g["atr_norm"] = g["atr_14"] / (g["close"] + 1e-12)

    # Volume
    vol = pd.to_numeric(g["volume"], errors="coerce").fillna(0.0)
    g["vol_sma_20"] = vol.rolling(20).mean()
    g["rel_vol_5"]  = vol / (vol.rolling(5).mean() + 1e-12)

    # Position vs recent extremes (trailing, causal)
    g["hi_24h"] = g["high"].rolling(288).max()
    g["lo_24h"] = g["low"].rolling(288).min()
    g["pos_in_24h"] = (g["close"] - g["lo_24h"]) / (g["hi_24h"] - g["lo_24h"] + 1e-12)

    # Distance to EMAs
    g["close_vs_ema21"] = (g["close"] - g["ema_21"]) / (g["ema_21"] + 1e-12)
    g["close_vs_ema50"] = (g["close"] - g["ema_50"]) / (g["ema_50"] + 1e-12)

    g.replace([np.inf, -np.inf], np.nan, inplace=True)
    return g


# =========================
# Labeling — EXACT simulator parity (next-bar entry; first-touch; SL-first)
# =========================
def make_long_labels(df: pd.DataFrame, tp=0.02, sl=0.01, horizon=12, entry_lag=1, same_bar_policy="sl-first") -> pd.Series:
    """
    Label at bar i using your simulator's semantics:
      • enter at bar i+entry_lag OPEN
      • scan forward for up to `horizon` bars
      • whichever TP/SL is hit FIRST decides
      • if both hit within the same bar, apply same_bar_policy ('sl-first'|'tp-first')
    Returns float32 Series aligned to index i with {1.0, 0.0, nan}
    """
    high = df["high"].to_numpy(dtype=float)
    low  = df["low"].to_numpy(dtype=float)
    openp= df["open"].to_numpy(dtype=float)
    n = len(df)
    y = np.full(n, np.nan, dtype=np.float32)
    last_i = n - (entry_lag + horizon)
    if last_i <= 0:
        return pd.Series(y, index=df.index, dtype="float32")

    sl_first = (same_bar_policy == "sl-first")

    for i in range(last_i):
        ei = i + entry_lag
        entry_price = openp[ei]
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        tp_px = entry_price * (1.0 + tp)
        sl_px = entry_price * (1.0 - sl)
        outcome = None

        # scan over bars [ei .. ei+horizon-1]
        for j in range(ei, ei + horizon):
            hi = high[j]; lo = low[j]
            if not (np.isfinite(hi) and np.isfinite(lo)):
                continue

            tp_hit = (hi >= tp_px)
            sl_hit = (lo <= sl_px)

            if tp_hit and sl_hit:
                outcome = 0.0 if sl_first else 1.0
                break
            elif tp_hit:
                outcome = 1.0
                break
            elif sl_hit:
                outcome = 0.0
                break

        y[i] = outcome if outcome is not None else np.nan

    return pd.Series(y, index=df.index, dtype="float32")


# =========================
# EV-based threshold with min-trades guard
# =========================
def pick_threshold_ev(y_true, proba, tp=0.02, sl=0.01, fees=0.0006, slip=0.0007,
                      n_min=400, floor=0.58, cap_quantile=0.75):
    """
    Sweep thresholds; choose one that maximizes expected value, subject to:
      • at least n_min predicted trades
      • threshold between [floor, quantile_cap] to avoid starvation
    EV = win_rate*tp - (1 - win_rate)*sl - fees - slip
    Returns (thr, wr_at_thr, ev_at_thr, n_trades)
    """
    if len(y_true) == 0:
        return floor, 0.0, -1.0, 0
    qcap = float(np.quantile(proba, cap_quantile)) if 0 < cap_quantile < 1 else 1.0

    # Candidate thresholds high→low
    cand = np.unique(np.round(proba, 6))[::-1]
    best = (floor, 0.0, -999.0, 0)
    for t in cand:
        if t < floor: break
        if t > qcap: t_eff = qcap
        else: t_eff = t
        pred = (proba >= t_eff).astype(int)
        n = int(pred.sum())
        if n < n_min:
            continue
        wins = int(((pred == 1) & (y_true == 1)).sum())
        if n == 0:
            continue
        wr = wins / n
        ev = wr*tp - (1-wr)*sl - fees - slip
        if ev > best[2]:
            best = (float(t_eff), float(wr), float(ev), n)

    # Fallback: relax n_min gradually to get a usable threshold
    if best[2] == -999.0:
        for req in [int(n_min*0.75), int(n_min*0.5), int(n_min*0.25), 50]:
            for t in cand:
                if t < floor: break
                t_eff = min(max(t, floor), qcap)
                pred = (proba >= t_eff).astype(int)
                n = int(pred.sum())
                if n < req: continue
                wins = int(((pred == 1) & (y_true == 1)).sum())
                if n == 0: continue
                wr = wins / n
                ev = wr*tp - (1-wr)*sl - fees - slip
                if ev > best[2]:
                    best = (float(t_eff), float(wr), float(ev), n)
            if best[2] != -999.0:
                break

    # If still nothing, return floor
    if best[2] == -999.0:
        return float(floor), 0.0, -1.0, 0
    return best


# =========================
# Helpers
# =========================
def _fetch_ohlcv(coin: str, start_ts: str, end_ts: str) -> pd.DataFrame:
    qs = (CoinAPIPrice.objects
          .filter(coin=coin, timestamp__gte=start_ts, timestamp__lte=end_ts)
          .values("timestamp","open","high","low","close","volume")
          .order_by("timestamp"))
    df = pd.DataFrame(list(qs))
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("timestamp")
    # force a clean 5m grid; ffill gaps to avoid implicit lookahead
    df = df.set_index("timestamp").asfreq("5min").ffill().reset_index()
    # normalize to naive UTC (your sim uses naive UTC)
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def _sample_weights(y):
    classes = np.unique(y)
    w = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    mp = {c: w[i] for i, c in enumerate(classes)}
    return np.array([mp[v] for v in y], dtype=np.float32)


# =========================
# Command
# =========================
class Command(BaseCommand):
    help = "Train a LONG model for one coin with labels matching the simulator exactly (next-bar entry, first-touch, SL-first)."

    def add_arguments(self, parser):
        parser.add_argument("--coin", type=str, default="XRPUSDT")
        parser.add_argument("--export_dir", type=str, default=".")
        parser.add_argument("--train_start", type=str, default="2023-01-01 00:00:00+00:00")
        parser.add_argument("--train_end",   type=str, default="2024-12-31 23:55:00+00:00")
        parser.add_argument("--test_start",  type=str, default="2025-01-01 00:00:00+00:00")
        parser.add_argument("--test_end",    type=str, default="2025-09-02 23:55:00+00:00")

        # Label params (must mirror simulator)
        parser.add_argument("--tp", type=float, default=0.02, help="Take-profit % (e.g., 0.02 = +2%).")
        parser.add_argument("--sl", type=float, default=0.01, help="Stop-loss % (e.g., 0.01 = -1%).")
        parser.add_argument("--horizon", type=int, default=12, help="Bars to look ahead (5m bars).")
        parser.add_argument("--entry_lag", type=int, default=1, help="Bars to delay entry after signal.")
        parser.add_argument("--same_bar_policy", type=str, default="sl-first", choices=["sl-first","tp-first"])

        # Threshold search
        parser.add_argument("--thr_floor", type=float, default=0.58)
        parser.add_argument("--thr_cap_quantile", type=float, default=0.75)
        parser.add_argument("--min_val_trades", type=int, default=400)

        # Costs for EV (match your simulator fees+slip defaults if you pass them there)
        parser.add_argument("--fees", type=float, default=0.0006)
        parser.add_argument("--slip", type=float, default=0.0007)

    def handle(self, *args, **o):
        coin = o["coin"].upper()
        out_dir = o["export_dir"]
        os.makedirs(out_dir, exist_ok=True)

        # ===== Load data =====
        self.stdout.write(f"▶ Loading {coin} OHLCV …")
        df_all = _fetch_ohlcv(coin, o["train_start"], o["test_end"])
        if df_all.empty:
            self.stderr.write("No data returned for coin/time range.")
            return

        # ===== Features FIRST, then split (avoid rolling NaN leakage) =====
        self.stdout.write("▶ Building features …")
        feat = _add_features(df_all)

        # ===== Label EXACTLY like simulator =====
        self.stdout.write("▶ Labeling (first-touch, next-bar entry, SL-first) …")
        feat["label"] = make_long_labels(
            feat,
            tp=float(o["tp"]), sl=float(o["sl"]),
            horizon=int(o["horizon"]), entry_lag=int(o["entry_lag"]),
            same_bar_policy=o["same_bar_policy"]
        )
        # Drop unlabeled rows
        feat = feat.dropna(subset=["label"]).reset_index(drop=True)



        # --- put this just before the time-split masks ---
        def _as_utc_ts(s):
            t = pd.Timestamp(s)
            return t.tz_convert('UTC') if t.tzinfo is not None else t.tz_localize('UTC')

        # ensure feature timestamps are UTC tz-aware (not naive)
        feat["timestamp"] = pd.to_datetime(feat["timestamp"], utc=True)

        TRAIN_START = _as_utc_ts(o["train_start"])
        TRAIN_END   = _as_utc_ts(o["train_end"])
        TEST_START  = _as_utc_ts(o["test_start"])
        TEST_END    = _as_utc_ts(o["test_end"])



        mask_train = (feat["timestamp"] >= TRAIN_START) & (feat["timestamp"] <= TRAIN_END)
        mask_test  = (feat["timestamp"] >= TEST_START)  & (feat["timestamp"] <= TEST_END)

        train_df = feat.loc[mask_train].copy()
        test_df  = feat.loc[mask_test].copy()
        if train_df.empty or test_df.empty:
            self.stderr.write("Train or test slice empty after split.")
            return

        # ===== Feature columns =====
        exclude = {"timestamp","open","high","low","close","volume","label","hi_24h","lo_24h"}
        cols = [c for c in train_df.columns if c not in exclude]
        # Keep features with ≥98% non-NaN in TRAIN
        cols = [c for c in cols if train_df[c].notna().mean() >= 0.98]
        if not cols:
            self.stderr.write("No usable features after NaN filter.")
            return

        self.stdout.write(f"▶ Using {len(cols)} features.")

        # ===== Inner validation (last 20% of train) =====
        n_train = len(train_df)
        val_len = max(2000, int(n_train * 0.2))
        if n_train <= val_len:
            self.stderr.write("Train too small for inner validation.")
            return
        inner_train = train_df.iloc[:-val_len].copy()
        inner_val   = train_df.iloc[-val_len:].copy()

        X_tr = inner_train[cols].astype("float32").fillna(0.0)
        y_tr = inner_train["label"].astype(int).to_numpy()

        X_val = inner_val[cols].astype("float32").fillna(0.0)
        y_val = inner_val["label"].astype(int).to_numpy()

        # ===== Model =====
        if HAS_LGBM:
            model = LGBMClassifier(
                n_estimators=1200,
                learning_rate=0.03,
                max_depth=-1,
                num_leaves=127,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0, reg_lambda=0.0,
                random_state=42
            )
        else:
            model = HistGradientBoostingClassifier(
                max_iter=1200,
                learning_rate=0.035,
                max_depth=None,
                l2_regularization=0.0,
                min_samples_leaf=25,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )

        sw_tr = _sample_weights(y_tr)
        self.stdout.write("▶ Training …")
        if HAS_LGBM:
            model.fit(X_tr, y_tr, sample_weight=sw_tr)
        else:
            model.fit(X_tr, y_tr, sample_weight=sw_tr)

        # ===== Validation threshold (EV) =====
        self.stdout.write("▶ Selecting threshold by EV on validation …")
        if hasattr(model, "predict_proba"):
            proba_val = model.predict_proba(X_val)[:,1]
        else:
            raw = model.decision_function(X_val)
            proba_val = 1.0 / (1.0 + np.exp(-raw))

        thr, wr, ev, ntr = pick_threshold_ev(
            y_val, proba_val,
            tp=float(o["tp"]), sl=float(o["sl"]),
            fees=float(o["fees"]), slip=float(o["slip"]),
            n_min=int(o["min_val_trades"]),
            floor=float(o["thr_floor"]),
            cap_quantile=float(o["thr_cap_quantile"])
        )
        self.stdout.write(f"   → thr={thr:.3f} | WR={wr:.3f} | EV={ev:.5f} | N={ntr}")

        # ===== Refit on FULL TRAIN =====
        X_full = train_df[cols].astype("float32").fillna(0.0)
        y_full = train_df["label"].astype(int).to_numpy()
        sw_full = _sample_weights(y_full)

        self.stdout.write("▶ Re-fitting on full training data …")
        model_final = type(model)(**getattr(model, "get_params", lambda: {})())
        if HAS_LGBM:
            model_final.fit(X_full, y_full, sample_weight=sw_full)
        else:
            model_final.fit(X_full, y_full, sample_weight=sw_full)

        # ===== Test inference =====
        X_test = test_df[cols].astype("float32").fillna(0.0)
        y_test = test_df["label"].astype(int).to_numpy()

        if hasattr(model_final, "predict_proba"):
            proba_test = model_final.predict_proba(X_test)[:,1]
        else:
            raw = model_final.decision_function(X_test)
            proba_test = 1.0 / (1.0 + np.exp(-raw))

        y_pred_test = (proba_test >= thr).astype(int)

        # ===== Metrics =====
        def _safe_auc(y, p):
            try: return roc_auc_score(y, p)
            except Exception: return float("nan")

        acc  = accuracy_score(y_test, y_pred_test)
        prec = precision_score(y_test, y_pred_test, zero_division=0)
        rec  = recall_score(y_test, y_pred_test, zero_division=0)
        f1   = f1_score(y_test, y_pred_test)
        auc  = _safe_auc(y_test, proba_test)
        bal  = balanced_accuracy_score(y_test, y_pred_test)
        cm   = confusion_matrix(y_test, y_pred_test).tolist()

        self.stdout.write("▶ TEST metrics:")
        self.stdout.write(f"   Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}  Bal={bal:.4f}")
        self.stdout.write(f"   Confusion={cm}")

        # ===== Exports =====
        prefix = coin.replace("USDT","").lower()
        model_path  = os.path.join(out_dir, f"{prefix}_long_model.joblib")
        meta_path   = os.path.join(out_dir, f"{prefix}_long_meta.json")
        feats_path  = os.path.join(out_dir, f"{prefix}_feature_list.json")
        train_csv   = os.path.join(out_dir, f"{prefix}_train_dataset.csv")
        test_csv    = os.path.join(out_dir, f"{prefix}_test_dataset.csv")
        preds_csv   = os.path.join(out_dir, f"{prefix}_predictions.csv")

        dump(model_final, model_path)

        with open(feats_path, "w") as f:
            json.dump(cols, f, indent=2)

        meta = {
            "coin": coin,
            "model_type": type(model_final).__name__,
            "features": cols,
            "threshold": round(float(thr), 4),
            "val_ev": {"win_rate": round(float(wr),4), "ev": round(float(ev),6), "n_trades": int(ntr)},
            "train_window": {"start": str(TRAIN_START), "end": str(TRAIN_END)},
            "test_window": {"start": str(TEST_START), "end": str(TEST_END)},
            "labeling": {
                "tp": float(o["tp"]), "sl": float(o["sl"]),
                "horizon_bars": int(o["horizon"]),
                "entry_lag": int(o["entry_lag"]),
                "same_bar_policy": o["same_bar_policy"]
            },
            "ev_costs": {"fees": float(o["fees"]), "slip": float(o["slip"])},
            "test_metrics": {
                "acc": round(float(acc),4), "prec": round(float(prec),4),
                "rec": round(float(rec),4), "f1": round(float(f1),4),
                "auc": round(float(auc),4) if not math.isnan(auc) else None,
                "balanced_acc": round(float(bal),4),
                "confusion": cm
            }
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Datasets (include coin for traceability)
        export_cols = ["timestamp","open","high","low","close","volume","label"] + cols
        tr_out = train_df[export_cols].copy(); tr_out.insert(0, "coin", coin)
        te_out = test_df[export_cols].copy();  te_out.insert(0, "coin", coin)
        tr_out.to_csv(train_csv, index=False); te_out.to_csv(test_csv, index=False)

        # Predictions CSV — EXACT schema required by your simulator
        pd.DataFrame({
            "coin": coin,
            "timestamp": test_df["timestamp"].values,
            "pred_prob": proba_test
        }).to_csv(preds_csv, index=False)

        self.stdout.write(self.style.SUCCESS(
            f"Done: {coin}\n"
            f"  Model: {model_path}\n"
            f"  Meta:  {meta_path}\n"
            f"  Feats: {feats_path}\n"
            f"  Train: {train_csv}\n"
            f"  Test:  {test_csv}\n"
            f"  Preds: {preds_csv}\n"
        ))
