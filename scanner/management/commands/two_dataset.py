from django.core.management.base import BaseCommand
from django.conf import settings
from scanner.models import CoinAPIPrice

import os, json, math, warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from joblib import dump

warnings.filterwarnings("ignore", category=UserWarning)

COINS = [
    "BTCUSDT","ETHUSDT","XRPUSDT","LTCUSDT","SOLUSDT","DOGEUSDT",
    "LINKUSDT","DOTUSDT","SHIBUSDT","ADAUSDT","UNIUSDT","AVAXUSDT",
    "XLMUSDT","TRXUSDT","ATOMUSDT"
]

# Fixed split per your spec
TRAIN_START = "2023-01-01 00:00:00+00:00"
TRAIN_END   = "2025-06-30 23:55:00+00:00"
TEST_START  = "2025-07-01 00:00:00+00:00"
TEST_END    = "2025-08-08 23:55:00+00:00"

# Default trade/label params (override with CLI flags if you want)
DEFAULT_TP = 0.03     # +3%
DEFAULT_SL = 0.02     # -2%
DEFAULT_H  = 36       # 36 x 5min = 3 hours

# -------------------------
# Feature Engineering
# -------------------------
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

def add_features(df):
    out = df.copy()
    for n in [1,3,6,12,24,48]:
        out[f"ret_{n}"] = out["close"].pct_change(n)

    for span in [9,21,50,200]:
        out[f"ema_{span}"] = ema(out["close"], span)

    macd_line, macd_sig, macd_hist = macd(out["close"])
    out["macd"] = macd_line
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd_hist

    out["rsi_14"] = rsi(out["close"], 14)

    bb_u, bb_m, bb_l, bb_w = bollinger(out["close"], 20, 2.0)
    out["bb_upper"] = bb_u
    out["bb_middle"] = bb_m
    out["bb_lower"]  = bb_l
    out["bb_width"]  = bb_w

    out["atr_14"] = atr(out["high"], out["low"], out["close"], 14)
    out["obv"] = obv(out["close"], out["volume"])
    out["vwap_20"] = vwap(out["close"], out["high"], out["low"], out["volume"], 20)

    out["close_above_ema_9"]   = (out["close"] > out["ema_9"]).astype(int)
    out["close_above_ema_21"]  = (out["close"] > out["ema_21"]).astype(int)
    out["close_above_ema_50"]  = (out["close"] > out["ema_50"]).astype(int)
    out["close_above_ema_200"] = (out["close"] > out["ema_200"]).astype(int)
    out["above_bb_mid"]        = (out["close"] > out["bb_middle"]).astype(int)

    out["hour"] = out["timestamp"].dt.hour
    out["dow"]  = out["timestamp"].dt.dayofweek
    out["hour_sin"] = np.sin(2*np.pi*out["hour"]/24)
    out["hour_cos"] = np.cos(2*np.pi*out["hour"]/24)
    out["dow_sin"]  = np.sin(2*np.pi*out["dow"]/7)
    out["dow_cos"]  = np.cos(2*np.pi*out["dow"]/7)

    out.replace([np.inf,-np.inf], np.nan, inplace=True)
    return out

# -------------------------
# Labeling (first-hit)
# -------------------------
def label_first_hit(close, high, low, tp=0.03, sl=0.02, horizon=36):
    n = len(close)
    y = np.zeros(n, dtype=np.int8)
    tp_levels = close.values * (1.0 + tp)
    sl_levels = close.values * (1.0 - sl)
    H = high.values; L = low.values

    for i in range(n - horizon):
        t_idx = -1; s_idx = -1
        t_level = tp_levels[i]; s_level = sl_levels[i]
        for j in range(1, horizon+1):
            if t_idx == -1 and H[i+j] >= t_level: t_idx = j
            if s_idx == -1 and L[i+j] <= s_level: s_idx = j
            if t_idx != -1 and s_idx != -1: break
        if t_idx == -1 and s_idx == -1: y[i] = 0
        elif t_idx != -1 and s_idx == -1: y[i] = 1
        elif t_idx == -1 and s_idx != -1: y[i] = 0
        else: y[i] = 1 if t_idx < s_idx else 0

    lab = pd.Series(y, index=close.index, dtype="float32")
    lab.iloc[-horizon:] = np.nan
    return lab

def build_dataset(df_all, tp, sl, horizon):
    df = df_all.copy()
    df["label"] = (
        df.groupby("coin", group_keys=False)
          .apply(lambda g: label_first_hit(g["close"], g["high"], g["low"], tp, sl, horizon))
    )
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    feat_cols = [
        "ret_1","ret_3","ret_6","ret_12","ret_24","ret_48",
        "ema_9","ema_21","ema_50","ema_200",
        "macd","macd_signal","macd_hist",
        "rsi_14","bb_upper","bb_middle","bb_lower","bb_width",
        "atr_14","obv","vwap_20",
        "close_above_ema_9","close_above_ema_21","close_above_ema_50","close_above_ema_200",
        "above_bb_mid",
        "hour_sin","hour_cos","dow_sin","dow_cos",
    ]
    X = df[feat_cols].astype("float32")
    y = df["label"].values.astype(int)
    return df, X, y, feat_cols

def balanced_weights(y):
    cls, cnt = np.unique(y, return_counts=True)
    freq = dict(zip(cls, cnt))
    total = len(y); k = len(cls)
    return np.array([total/(k*freq[v]) for v in y], dtype="float32")

def metric_pack(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    try: auc = roc_auc_score(y_true, y_prob)
    except Exception: auc = float("nan")
    return acc, prec, rec, auc

class Command(BaseCommand):
    help = "Build train/test datasets (fixed dates), train a long model, and output test predictions CSV with confidences."

    def add_arguments(self, parser):
        parser.add_argument("--export_dir", type=str, default="./exports", help="Where to write outputs")
        parser.add_argument("--tp", type=float, default=DEFAULT_TP)
        parser.add_argument("--sl", type=float, default=DEFAULT_SL)
        parser.add_argument("--h", type=int, default=DEFAULT_H)

    def handle(self, *args, **opts):
        export_dir = opts["export_dir"]; os.makedirs(export_dir, exist_ok=True)
        tp = float(opts["tp"]); sl = float(opts["sl"]); horizon = int(opts["h"])

        # 1) Load all relevant OHLCV once (2023-01-01 .. 2025-08-31)
        qs = (CoinAPIPrice.objects
              .filter(coin__in=COINS, timestamp__gte=TRAIN_START, timestamp__lte=TEST_END)
              .values("coin","timestamp","open","high","low","close","volume")
              .order_by("coin","timestamp"))
        df = pd.DataFrame.from_records(list(qs))
        if df.empty:
            self.stderr.write("No data returned for the requested window.")
            return

        # to numeric
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # 2) Features per coin
        dfs = []
        for coin, g in df.groupby("coin", sort=False):
            g = g.sort_values("timestamp").reset_index(drop=True)
            gf = add_features(g)
            dfs.append(gf.assign(coin=coin))
        df_all = pd.concat(dfs, ignore_index=True)

        # require core features present
        df_all = df_all.dropna(subset=["ema_200","bb_width","rsi_14","atr_14","obv","vwap_20"]).reset_index(drop=True)

        # 3) Labels with your TP/SL/H
        self.stdout.write(f"Labeling: TP={tp*100:.1f}% SL={sl*100:.1f}% Horizon={horizon} bars")
        df_lab, X_all, y_all, feat_cols = build_dataset(df_all, tp, sl, horizon)

        # 4) Fixed time split
        train_mask = (df_lab["timestamp"] >= pd.Timestamp(TRAIN_START)) & (df_lab["timestamp"] <= pd.Timestamp(TRAIN_END))
        test_mask  = (df_lab["timestamp"] >= pd.Timestamp(TEST_START))  & (df_lab["timestamp"] <= pd.Timestamp(TEST_END))

        train_rows = df_lab.loc[train_mask].copy()
        test_rows  = df_lab.loc[test_mask].copy()

        if len(train_rows) < 5000 or len(test_rows) < 1000:
            self.stderr.write(f"Not enough data after split. Train={len(train_rows)} Test={len(test_rows)}")
            return

        X_train = train_rows[feat_cols].astype("float32")
        y_train = train_rows["label"].values.astype(int)
        X_test  = test_rows[feat_cols].astype("float32")
        y_test  = test_rows["label"].values.astype(int)

        # 5) Scale continuous features only
        cont = [c for c in feat_cols if not c.startswith("close_above_") and c not in ("above_bb_mid",)]
        scaler = StandardScaler()
        X_train_s = X_train.copy()
        X_test_s  = X_test.copy()
        X_train_s[cont] = scaler.fit_transform(X_train[cont].fillna(0.0))
        X_test_s[cont]  = scaler.transform(X_test[cont].fillna(0.0))

        # 6) Train model (fast, decent baseline)
        w = balanced_weights(y_train)
        clf = HistGradientBoostingClassifier(
            max_depth=6, max_iter=300, learning_rate=0.06,
            l2_regularization=0.0, min_samples_leaf=40, random_state=42
        )
        clf.fit(X_train_s.fillna(0.0), y_train, sample_weight=w)

        # 7) Evaluate (on July/Aug test)
        test_prob = clf.predict_proba(X_test_s.fillna(0.0))[:,1]
        acc, prec, rec, auc = metric_pack(y_test, test_prob, thr=0.5)
        self.stdout.write(f"Test (Jul–Aug 2025) => acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} auc={auc:.3f} n={len(y_test)}")

        # Per-coin sanity
        tmp = test_rows[["coin"]].copy()
        tmp["y_true"] = y_test
        tmp["y_pred"] = (test_prob >= 0.5).astype(int)
        per_coin = tmp.groupby("coin").apply(lambda g: accuracy_score(g["y_true"], g["y_pred"])).to_dict()
        for c, v in per_coin.items():
            self.stdout.write(f"  {c}: acc={v:.3f}")

        # 8) Persist datasets, model, config, predictions
        train_csv = os.path.join(export_dir, "two_train_dataset.csv")
        test_csv  = os.path.join(export_dir, "two_test_dataset.csv")
        preds_csv = os.path.join(export_dir, "two_predictions.csv")
        model_pth = os.path.join(export_dir, "two_long_hgb_model.joblib")
        scaler_pth= os.path.join(export_dir, "two_feature_scaler.joblib")
        feats_pth = os.path.join(export_dir, "two_feature_list.json")
        cfg_pth   = os.path.join(export_dir, "two_trade_config.json")

        # minimal train/test exports
        cols_export = ["coin","timestamp","open","high","low","close","volume","label"] + feat_cols
        train_rows[cols_export].to_csv(train_csv, index=False)
        test_rows[cols_export].to_csv(test_csv, index=False)

        dump(clf, model_pth)
        dump(scaler, scaler_pth)
        with open(feats_pth, "w") as f: json.dump(feat_cols, f, indent=2)

        # Predictions file for your simulator
        out = test_rows[["coin","timestamp"]].copy().reset_index(drop=True)
        out["label"] = y_test
        out["pred_prob"] = test_prob
        out["pred"] = (out["pred_prob"] >= 0.5).astype(int)
        out.to_csv(preds_csv, index=False)

        # basic trading suggestion (expectancy proxy)
        winrate = float((out["pred"] == out["label"]).mean())
        expectancy = winrate*tp - (1-winrate)*sl
        # keep worst-case loss <= 20% equity at SL hit
        lev_cap = 0.20/sl if sl > 0 else 1.0
        leverage = float(round(min(lev_cap, 15.0), 2))
        pos_frac = float(round(min(max(expectancy/(sl+1e-9),0.0), 0.5), 3))

        cfg = {
            "train_window": {"start": TRAIN_START, "end": TRAIN_END},
            "test_window":  {"start": TEST_START,  "end": TEST_END},
            "tp": tp, "sl": sl, "horizon_bars": horizon, "threshold": 0.50,
            "suggested_leverage_x": leverage,
            "suggested_position_fraction": pos_frac,
            "expectancy_per_trade_pct": round(expectancy*100.0, 3),
            "test_metrics": {"acc": round(acc,4), "prec": round(prec,4), "rec": round(rec,4), "auc": round(auc,4)},
            "per_coin_accuracy": {k: float(v) for k,v in per_coin.items()}
        }
        with open(cfg_pth, "w") as f: json.dump(cfg, f, indent=2)

        self.stdout.write(self.style.SUCCESS(f"Saved:\n  {train_csv}\n  {test_csv}\n  {preds_csv}\n  {model_pth}\n  {scaler_pth}\n  {feats_pth}\n  {cfg_pth}"))
