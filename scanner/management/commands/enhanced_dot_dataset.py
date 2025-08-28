# Enhanced DOTUSDT ML model with CryptoQuant-style on-chain indicators
# Target: >60% accuracy with advanced technical and simulated on-chain metrics

from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice

import os, json
import numpy as np
import pandas as pd
from copy import deepcopy
from joblib import dump

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold, f_classif
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)


class Command(BaseCommand):
    help = "Enhanced DOT ML model with CryptoQuant-style indicators for >60% accuracy"
    
    def calculate_mfi(self, high, low, close, volume, period=14):
        """Money Flow Index calculation"""
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        
        positive_flow = pd.Series(0.0, index=close.index)
        negative_flow = pd.Series(0.0, index=close.index)
        
        price_changes = typical_price.diff()
        positive_flow = pd.where(price_changes > 0, raw_money_flow, 0)
        negative_flow = pd.where(price_changes < 0, raw_money_flow, 0)
        
        positive_mf = pd.Series(positive_flow).rolling(period).sum()
        negative_mf = pd.Series(negative_flow).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-12)))
        return mfi
    
    def calculate_cmf(self, high, low, close, volume, period=20):
        """Chaikin Money Flow calculation"""
        mfm = ((close - low) - (high - close)) / (high - low + 1e-12)
        mfv = mfm * volume
        cmf = mfv.rolling(period).sum() / volume.rolling(period).sum()
        return cmf
    
    def calculate_tsi(self, close, long=25, short=13, signal=13):
        """True Strength Index"""
        pc = close.diff(1)
        double_smoothed_pc = pc.ewm(span=long).mean().ewm(span=short).mean()
        double_smoothed_apc = pc.abs().ewm(span=long).mean().ewm(span=short).mean()
        
        tsi = 100 * (double_smoothed_pc / double_smoothed_apc)
        tsi_signal = tsi.ewm(span=signal).mean()
        
        return tsi, tsi_signal
    
    def calculate_squeeze_momentum(self, high, low, close, period=20, mult=2):
        """Bollinger Band Squeeze Momentum"""
        # Bollinger Bands
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper_bb = sma + (mult * std)
        lower_bb = sma - (mult * std)
        
        # Keltner Channels
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        ema = close.ewm(span=period).mean()
        upper_kc = ema + (mult * atr)
        lower_kc = ema - (mult * atr)
        
        # Squeeze condition
        squeeze = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        
        # Momentum
        mom = close - close.rolling(period).mean().shift(int(period/2))
        
        return squeeze.astype(int), mom

    def add_arguments(self, parser):
        parser.add_argument("--coin", type=str, default="DOTUSDT", help="e.g., DOTUSDT")
        parser.add_argument("--export_dir", type=str, default=".", help="Output directory")
        parser.add_argument("--tp_pct", type=float, default=0.025, help="Take profit fraction (0.025 = 2.5%)")
        parser.add_argument("--sl_pct", type=float, default=0.015, help="Stop loss fraction (0.015 = 1.5%)")
        parser.add_argument("--max_bars", type=int, default=24, help="Max hold bars (e.g., 24=2h on 5m)")
        parser.add_argument("--test_months", type=int, default=3, help="Reserve last N months for held-out test")
        parser.add_argument("--min_trades", type=int, default=100, help="Min predicted trades required per fold/phase")
        parser.add_argument("--k_features", type=int, default=25, help="Top-K features after variance filter")
        parser.add_argument("--cv_folds", type=int, default=5, help="Number of walk-forward folds before final test")

    def handle(self, *args, **opts):
        COIN        = opts["coin"].upper()
        EXPORT_DIR  = opts["export_dir"]
        TP_PCT      = float(opts["tp_pct"])
        SL_PCT      = float(opts["sl_pct"])
        MAX_BARS    = int(opts["max_bars"])
        TEST_MONTHS = int(opts["test_months"])
        MIN_TRADES  = int(opts["min_trades"])
        K_FEATS     = int(opts["k_features"])
        CV_FOLDS    = int(opts["cv_folds"])
        RNG_SEED    = 42
        np.random.seed(RNG_SEED)
        EPS = 1e-12

        print(f"🎯 Enhanced {COIN} | TP {TP_PCT*100:.1f}% | SL {SL_PCT*100:.1f}% | Max {MAX_BARS} bars | Test last {TEST_MONTHS} mo | CV folds={CV_FOLDS}")

        # -------------------
        # Load OHLCV
        # -------------------
        print("📊 Loading OHLCV…")
        qs = (
            CoinAPIPrice.objects
            .filter(coin=COIN, timestamp__gte="2023-01-01", timestamp__lte="2025-08-31")
            .values("timestamp","open","high","low","close","volume")
            .order_by("timestamp")
        )
        df = pd.DataFrame.from_records(list(qs))
        if df.empty:
            print("❌ No data found")
            return

        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.dropna().reset_index(drop=True)
        print(f"✅ {len(df):,} candles | {df['timestamp'].min()} → {df['timestamp'].max()}")

        # -------------------
        # Enhanced Feature Engineering
        # -------------------
        def ema(s, span): return s.ewm(span=span, adjust=False).mean()
        def sma(s, span): return s.rolling(span).mean()

        def rsi(close, period=14):
            d = close.diff()
            up = d.clip(lower=0.0)
            dn = -d.clip(upper=0.0)
            ru = up.ewm(alpha=1/period, adjust=False).mean()
            rd = dn.ewm(alpha=1/period, adjust=False).mean()
            rs = ru / (rd + EPS)
            return 100 - (100 / (1 + rs))

        def stoch_rsi(close, period=14, k=3, d=3):
            rs = rsi(close, period)
            lo = rs.rolling(period).min()
            hi = rs.rolling(period).max()
            st = (rs - lo) / (hi - lo + EPS)
            k_val = st.rolling(k).mean() * 100
            d_val = k_val.rolling(d).mean()
            return k_val, d_val

        def williams_r(high, low, close, period=14):
            hh = high.rolling(period).max()
            ll = low.rolling(period).min()
            return -100 * (hh - close) / (hh - ll + EPS)

        def atr(high, low, close, period=14):
            h_l = (high - low)
            h_c = (high - close.shift(1)).abs()
            l_c = (low - close.shift(1)).abs()
            tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
            return tr.rolling(period).mean()

        def adx(high, low, close, period=14):
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            tr = atr(high, low, close, 1)
            plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / (tr.ewm(alpha=1/period, adjust=False).mean() + EPS))
            minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / (tr.ewm(alpha=1/period, adjust=False).mean() + EPS))
            dx = (plus_di - minus_di).abs() / (plus_di + minus_di + EPS) * 100
            return dx.ewm(alpha=1/period, adjust=False).mean(), plus_di, minus_di

        def macd(close, fast=12, slow=26, signal=9):
            exp1 = ema(close, fast)
            exp2 = ema(close, slow)
            macd_line = exp1 - exp2
            signal_line = ema(macd_line, signal)
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram

        def compute_enhanced_features(df_in: pd.DataFrame):
            g = df_in.copy().sort_values("timestamp").reset_index(drop=True)
            vol = pd.to_numeric(g["volume"], errors='coerce').fillna(0.0)

            # Core price features
            g["ret_1"] = g["close"].pct_change()
            g["log_ret_1"] = np.log((g["close"] + EPS) / (g["close"].shift(1) + EPS))
            g["body_pct"] = (g["close"] - g["open"]).abs() / (g["close"].abs() + EPS)
            g["range_pct"] = (g["high"] - g["low"]) / (g["close"].abs() + EPS)
            g["upper_shadow"] = (g["high"] - np.maximum(g["open"], g["close"])) / (g["close"].abs() + EPS)
            g["lower_shadow"] = (np.minimum(g["open"], g["close"]) - g["low"]) / (g["close"].abs() + EPS)
            g["close_pos"] = (g["close"] - g["low"]) / (g["high"] - g["low"] + EPS)

            # Multi-timeframe returns with enhanced periods
            for n in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]:
                g[f"ret_{n}"] = g["close"].pct_change(n)
                g[f"ret_{n}_zscore"] = (g[f"ret_{n}"] - g[f"ret_{n}"].rolling(100).mean()) / (g[f"ret_{n}"].rolling(100).std() + EPS)
                g[f"ret_{n}_rank"] = g[f"ret_{n}"].rolling(100).rank(pct=True)

            # Enhanced volatility suite
            for period in [5, 10, 20, 50, 100]:
                vol_p = g["ret_1"].rolling(period).std()
                g[f"volatility_{period}"] = vol_p
                g[f"vol_zscore_{period}"] = (vol_p - vol_p.rolling(100).mean()) / (vol_p.rolling(100).std() + EPS)
                g[f"vol_percentile_{period}"] = vol_p.rolling(200).rank(pct=True)

            # Advanced moving averages with comprehensive analysis
            ma_periods = [3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
            for p in ma_periods:
                sma_val = sma(g["close"], p)
                ema_val = ema(g["close"], p)
                g[f"sma_{p}"] = sma_val
                g[f"ema_{p}"] = ema_val
                g[f"price_sma_{p}_ratio"] = g["close"] / (sma_val + EPS)
                g[f"price_ema_{p}_ratio"] = g["close"] / (ema_val + EPS)
                
                # MA slopes and momentum
                g[f"sma_{p}_slope_1"] = sma_val.diff(1)
                g[f"sma_{p}_slope_3"] = sma_val.diff(3)
                g[f"sma_{p}_slope_5"] = sma_val.diff(5)
                g[f"ema_{p}_slope_1"] = ema_val.diff(1)
                g[f"ema_{p}_slope_3"] = ema_val.diff(3)
                g[f"ema_{p}_slope_5"] = ema_val.diff(5)
                
                # Distance from MAs
                g[f"price_distance_sma_{p}"] = (g["close"] - sma_val) / (g["close"] + EPS)
                g[f"price_distance_ema_{p}"] = (g["close"] - ema_val) / (g["close"] + EPS)

            # MA crosses and relationships
            g["ema_8_21_cross"] = (g["ema_8"] > g["ema_21"]).astype(int)
            g["ema_21_55_cross"] = (g["ema_21"] > g["ema_55"]).astype(int)
            g["sma_13_55_cross"] = (g["sma_13"] > g["sma_55"]).astype(int)
            g["golden_cross"] = (g["sma_55"] > g["sma_233"]).astype(int)

            # Enhanced RSI suite
            for period in [7, 14, 21, 28]:
                rsi_val = rsi(g["close"], period)
                g[f"rsi_{period}"] = rsi_val
                g[f"rsi_{period}_slope"] = rsi_val.diff()
                g[f"rsi_{period}_momentum"] = rsi_val.diff(3)
                g[f"rsi_{period}_overbought"] = (rsi_val > 70).astype(int)
                g[f"rsi_{period}_oversold"] = (rsi_val < 30).astype(int)
                g[f"rsi_{period}_extreme"] = ((rsi_val > 80) | (rsi_val < 20)).astype(int)

            # Stochastic RSI
            g["stoch_rsi_k"], g["stoch_rsi_d"] = stoch_rsi(g["close"])
            g["stoch_cross_up"] = ((g["stoch_rsi_k"] > g["stoch_rsi_d"]) & (g["stoch_rsi_k"].shift(1) <= g["stoch_rsi_d"].shift(1))).astype(int)

            # Enhanced MACD
            macd_line, macd_sig, macd_hist = macd(g["close"])
            g["macd"] = macd_line
            g["macd_sig"] = macd_sig
            g["macd_hist"] = macd_hist
            g["macd_hist_slope"] = macd_hist.diff()
            g["macd_hist_momentum"] = macd_hist.diff(3)
            g["macd_zero_cross"] = ((macd_line > 0) & (macd_line.shift(1) <= 0)).astype(int)

            # Williams %R
            g["williams_r"] = williams_r(g["high"], g["low"], g["close"])
            g["williams_r_slope"] = g["williams_r"].diff()

            # ADX and directional indicators
            g["adx"], g["plus_di"], g["minus_di"] = adx(g["high"], g["low"], g["close"])
            g["adx_rising"] = (g["adx"].diff() > 0).astype(int)
            g["strong_trend"] = (g["adx"] > 25).astype(int)
            g["directional_bias"] = g["plus_di"] - g["minus_di"]

            # True Strength Index
            g["tsi"], g["tsi_signal"] = self.calculate_tsi(g["close"])
            g["tsi_cross"] = (g["tsi"] > g["tsi_signal"]).astype(int)

            # ATR and volatility
            g["atr"] = atr(g["high"], g["low"], g["close"])
            g["atr_ratio"] = g["atr"] / (g["close"] + EPS)
            g["atr_percentile"] = g["atr"].rolling(100).rank(pct=True)

            # Enhanced Bollinger Bands
            for period, mult in [(10, 2), (20, 2), (20, 1.5), (20, 2.5)]:
                mid = sma(g["close"], period)
                sd = g["close"].rolling(period).std()
                upper, lower = mid + mult * sd, mid - mult * sd
                g[f"bb_{period}_{mult}_pos"] = (g["close"] - lower) / (upper - lower + EPS)
                g[f"bb_{period}_{mult}_width"] = (upper - lower) / (mid + EPS)
                g[f"bb_{period}_{mult}_zscore"] = (g["close"] - mid) / (sd + EPS)
                g[f"bb_{period}_{mult}_squeeze"] = (g[f"bb_{period}_{mult}_width"] < g[f"bb_{period}_{mult}_width"].rolling(50).quantile(0.2)).astype(int)
                
                # Bollinger Band touches
                g[f"bb_{period}_{mult}_upper_touch"] = (g["high"] >= upper * 0.99).astype(int)
                g[f"bb_{period}_{mult}_lower_touch"] = (g["low"] <= lower * 1.01).astype(int)

            # Squeeze Momentum
            g["squeeze"], g["squeeze_momentum"] = self.calculate_squeeze_momentum(g["high"], g["low"], g["close"])

            # CryptoQuant-style On-Chain Inspired Features
            # Volume flow analysis
            vol_ma_20 = vol.rolling(20).mean()
            vol_ma_50 = vol.rolling(50).mean()
            g["vol_flow_ratio"] = vol_ma_20 / (vol_ma_50 + EPS)
            g["vol_surge_intensity"] = (vol / vol_ma_20).clip(0, 10)
            g["vol_accumulation"] = ((vol > vol_ma_20) & (g["close"] > g["open"])).astype(int)
            g["vol_distribution"] = ((vol > vol_ma_20) & (g["close"] < g["open"])).astype(int)

            # Advanced volume analysis
            for period in [5, 10, 20, 50]:
                vol_sma = vol.rolling(period).mean()
                vol_std = vol.rolling(period).std()
                g[f"vol_zscore_{period}"] = (vol - vol_sma) / (vol_std + EPS)
                g[f"vol_surge_{period}"] = (vol > vol_sma + 2 * vol_std).astype(int)
                g[f"vol_drought_{period}"] = (vol < vol_sma - vol_std).astype(int)

            # Money Flow Index and Chaikin Money Flow
            g["mfi"] = self.calculate_mfi(g["high"], g["low"], g["close"], vol)
            g["cmf"] = self.calculate_cmf(g["high"], g["low"], g["close"], vol)

            # Price-Volume correlation
            g["pv_correlation_20"] = g["ret_1"].rolling(20).corr(vol.pct_change())
            g["pv_correlation_50"] = g["ret_1"].rolling(50).corr(vol.pct_change())

            # Order flow simulation
            g["buying_pressure"] = g["close_pos"] * vol
            g["selling_pressure"] = (1 - g["close_pos"]) * vol
            g["order_flow_imbalance"] = (g["buying_pressure"] - g["selling_pressure"]) / (vol + EPS)
            g["cumulative_delta"] = g["order_flow_imbalance"].rolling(50).sum()

            # VWAP variants
            for period in [20, 50]:
                vwap = (g["close"] * vol).rolling(period).sum() / (vol.rolling(period).sum() + EPS)
                g[f"vwap_{period}"] = vwap
                g[f"price_vwap_{period}_ratio"] = g["close"] / (vwap + EPS)
                g[f"vwap_{period}_distance"] = (g["close"] - vwap) / (g["close"] + EPS)

            # Simulated MVRV
            vwap_100 = (g["close"] * vol).rolling(100).sum() / (vol.rolling(100).sum() + EPS)
            g["mvrv_sim"] = g["close"] / (vwap_100 + EPS)
            g["mvrv_zscore"] = (g["mvrv_sim"] - g["mvrv_sim"].rolling(200).mean()) / (g["mvrv_sim"].rolling(200).std() + EPS)

            # Network value simulation
            network_value = g["close"] * vol
            transaction_volume = vol.rolling(20).mean()
            g["nvt_ratio"] = network_value / (transaction_volume + EPS)
            g["nvt_signal"] = g["nvt_ratio"].rolling(20).mean()

            # Ichimoku Cloud
            high_9 = g["high"].rolling(9).max()
            low_9 = g["low"].rolling(9).min()
            high_26 = g["high"].rolling(26).max()
            low_26 = g["low"].rolling(26).min()
            high_52 = g["high"].rolling(52).max()
            low_52 = g["low"].rolling(52).min()

            g["tenkan_sen"] = (high_9 + low_9) / 2
            g["kijun_sen"] = (high_26 + low_26) / 2
            g["senkou_span_a"] = (g["tenkan_sen"] + g["kijun_sen"]) / 2
            g["senkou_span_b"] = (high_52 + low_52) / 2

            # Cloud analysis
            cloud_top = np.maximum(g["senkou_span_a"], g["senkou_span_b"])
            cloud_bottom = np.minimum(g["senkou_span_a"], g["senkou_span_b"])
            g["price_vs_cloud"] = np.where(g["close"] > cloud_top, 1,
                                         np.where(g["close"] < cloud_bottom, -1, 0))
            g["cloud_thickness"] = (cloud_top - cloud_bottom) / (g["close"] + EPS)
            g["tenkan_kijun_cross"] = (g["tenkan_sen"] > g["kijun_sen"]).astype(int)

            # Market structure
            g["fractal_high"] = ((g["high"] > g["high"].shift(2)) & (g["high"] > g["high"].shift(1)) & 
                               (g["high"] > g["high"].shift(-1)) & (g["high"] > g["high"].shift(-2))).astype(int)
            g["fractal_low"] = ((g["low"] < g["low"].shift(2)) & (g["low"] < g["low"].shift(1)) & 
                              (g["low"] < g["low"].shift(-1)) & (g["low"] < g["low"].shift(-2))).astype(int)

            # Support and resistance levels
            for period in [20, 50, 100]:
                resistance = g["high"].rolling(period).max()
                support = g["low"].rolling(period).min()
                g[f"resistance_distance_{period}"] = (resistance - g["close"]) / (g["close"] + EPS)
                g[f"support_distance_{period}"] = (g["close"] - support) / (g["close"] + EPS)

            # Candlestick patterns
            g["doji"] = (g["body_pct"] < g["range_pct"] * 0.1).astype(int)
            g["hammer"] = ((g["lower_shadow"] > g["body_pct"] * 2) & (g["upper_shadow"] < g["body_pct"])).astype(int)
            g["shooting_star"] = ((g["upper_shadow"] > g["body_pct"] * 2) & (g["lower_shadow"] < g["body_pct"])).astype(int)
            g["engulfing"] = ((g["body_pct"] > g["body_pct"].shift(1) * 1.5) & 
                            ((g["close"] > g["open"]) != (g["close"].shift(1) > g["open"].shift(1)))).astype(int)

            # Time features with enhanced cyclical encoding
            g["hour"] = g["timestamp"].dt.hour
            g["dow"] = g["timestamp"].dt.dayofweek
            g["month"] = g["timestamp"].dt.month
            g["quarter"] = g["timestamp"].dt.quarter

            # Cyclical encoding
            g["hour_sin"] = np.sin(2*np.pi*g["hour"]/24)
            g["hour_cos"] = np.cos(2*np.pi*g["hour"]/24)
            g["dow_sin"] = np.sin(2*np.pi*g["dow"]/7)
            g["dow_cos"] = np.cos(2*np.pi*g["dow"]/7)
            g["month_sin"] = np.sin(2*np.pi*g["month"]/12)
            g["month_cos"] = np.cos(2*np.pi*g["month"]/12)

            # Trading session indicators
            g["is_weekend"] = (g["dow"] >= 5).astype(int)
            g["is_asia_session"] = ((g["hour"] >= 0) & (g["hour"] <= 8)).astype(int)
            g["is_london_session"] = ((g["hour"] >= 8) & (g["hour"] <= 16)).astype(int)
            g["is_ny_session"] = ((g["hour"] >= 13) & (g["hour"] <= 21)).astype(int)
            g["session_overlap"] = ((g["hour"] >= 13) & (g["hour"] <= 16)).astype(int)

            # Advanced volatility regimes
            vol_20 = g["ret_1"].rolling(20).std()
            g["vol_regime"] = vol_20.rolling(200).rank(pct=True)
            g["high_vol_regime"] = (g["vol_regime"] > 0.8).astype(int)
            g["low_vol_regime"] = (g["vol_regime"] < 0.2).astype(int)

            # Trend regimes
            trend_signals = [
                g["golden_cross"],
                g["price_vs_cloud"],
                (g["adx"] > 25).astype(int) * np.sign(g["directional_bias"]),
                (g["ema_21"] > g["ema_55"]).astype(int) * 2 - 1
            ]
            g["trend_consensus"] = np.mean(trend_signals, axis=0)
            g["bullish_regime"] = (g["trend_consensus"] > 0.3).astype(int)
            g["bearish_regime"] = (g["trend_consensus"] < -0.3).astype(int)

            # Clean up
            g = g.replace([np.inf, -np.inf], np.nan)
            g = g.fillna(method="ffill").fillna(0)

            essential = ["timestamp","open","high","low","close","volume"]
            feature_cols = [c for c in g.columns if c not in essential]
            return g[essential + feature_cols].copy(), feature_cols

        print("🔧 Computing enhanced features on full dataset…")
        df_feat, feature_cols_all = compute_enhanced_features(df)
        print(f"✅ Generated {len(feature_cols_all)} features")

        # -------------------
        # Labels (optimized for DOT)
        # -------------------
        def make_labels(df_in: pd.DataFrame, tp_pct: float, sl_pct: float, max_bars: int) -> pd.Series:
            close = df_in["close"].values
            high  = df_in["high"].values
            low   = df_in["low"].values
            n = len(close)
            usable = n - max_bars
            if usable <= 0:
                return pd.Series(np.full(n, np.nan, dtype=np.float32), index=df_in.index, name="label")

            entry = close[:usable]
            tp = entry * (1.0 + tp_pct)
            sl = entry * (1.0 - sl_pct)

            labels = np.full(usable, -1, dtype=np.int8)
            decided = np.zeros(usable, dtype=bool)

            for j in range(1, max_bars + 1):
                hi_j = high[j:j+usable]
                lo_j = low[j:j+usable]
                tp_hit = (~decided) & (hi_j >= tp)
                if tp_hit.any():
                    labels[tp_hit] = 1
                    decided[tp_hit] = True
                sl_hit = (~decided) & (lo_j <= sl)
                if sl_hit.any():
                    labels[sl_hit] = 0
                    decided[sl_hit] = True
                if decided.all():
                    break

            if (~decided).any():
                final = close[max_bars:max_bars+usable]
                undec = ~decided
                labels[undec] = (final[undec] > entry[undec]).astype(np.int8)

            out = np.full(n, np.nan, dtype=np.float32)
            out[:usable] = labels
            return pd.Series(out, index=df_in.index, name="label")

        print("🎯 Creating optimized labels…")
        df_feat["label"] = make_labels(df_feat, TP_PCT, SL_PCT, MAX_BARS)

        # -------------------
        # Enhanced preprocessing with feature selection
        # -------------------
        def enhanced_preprocessing_pipeline(X_tr: np.ndarray, y_tr: np.ndarray, k_cap: int):
            """Enhanced preprocessing with multiple stages"""
            # Stage 1: Remove constant and near-constant features
            vt = VarianceThreshold(threshold=0.01)
            X_tr_v = vt.fit_transform(X_tr)
            
            # Stage 2: Remove highly correlated features
            if X_tr_v.shape[1] > k_cap * 2:
                corr_matrix = pd.DataFrame(X_tr_v).corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [c for c in upper_tri.columns if any(upper_tri[c] > 0.85)]
                keep_cols = [i for i in range(X_tr_v.shape[1]) if i not in to_drop]
                X_tr_v = X_tr_v[:, keep_cols]
                print(f"   Removed {len(to_drop)} highly correlated features")
            else:
                keep_cols = None
                
            # Stage 3: Feature selection using multiple methods
            k_use = min(k_cap, X_tr_v.shape[1])
            if k_use > 1:
                # Use f_classif for stability
                selector = SelectKBest(score_func=f_classif, k=k_use)
                X_tr_s = selector.fit_transform(X_tr_v, y_tr)
                selected_idx = selector.get_support(indices=True)
            else:
                selector = None
                X_tr_s = X_tr_v
                selected_idx = list(range(X_tr_v.shape[1]))
                
            # Stage 4: Robust scaling (better for outliers)
            scaler = RobustScaler()
            X_tr_final = scaler.fit_transform(X_tr_s)
            
            return vt, selector, scaler, X_tr_final, selected_idx, keep_cols

        def apply_preprocessing_pipeline(vt, selector, scaler, X_te: np.ndarray, keep_cols=None):
            X_te_v = vt.transform(X_te)
            if keep_cols is not None:
                X_te_v = X_te_v[:, keep_cols]
            X_te_s = X_te_v if selector is None else selector.transform(X_te_v)
            X_te_final = scaler.transform(X_te_s)
            return X_te_final

        # -------------------
        # Train/Test split
        # -------------------
        max_date = df_feat["timestamp"].max()
        split_date = max_date - pd.DateOffset(months=TEST_MONTHS)
        test_mask = df_feat["timestamp"] >= split_date
        
        test_start_idx = int(df_feat.index[test_mask][0])
        safe_train_end = test_start_idx - 1 - MAX_BARS
        
        if safe_train_end < 1000:
            print("❌ Insufficient training data")
            return

        train_idx = np.arange(0, safe_train_end + 1)
        test_idx = np.arange(test_start_idx, len(df_feat))

        essential = ["timestamp","open","high","low","close","volume","label"]
        feature_cols = [c for c in df_feat.columns if c not in essential]

        X_train = df_feat.iloc[train_idx][feature_cols].astype("float32").reset_index(drop=True)
        y_train = df_feat.iloc[train_idx]["label"].dropna().astype("int64").reset_index(drop=True)
        X_train = X_train.iloc[:len(y_train)]  # Align lengths

        X_test = df_feat.iloc[test_idx][feature_cols].astype("float32").reset_index(drop=True)
        ts_test = df_feat.iloc[test_idx]["timestamp"].reset_index(drop=True)

        print(f"📚 Train: {len(X_train):,} | Test: {len(X_test):,}")

        # -------------------
        # Enhanced model ensemble
        # -------------------
        models = {
            "lgb": LGBMClassifier(
                objective="binary",
                boosting_type="gbdt",
                num_leaves=20,
                learning_rate=0.05,
                feature_fraction=0.7,
                bagging_fraction=0.8,
                bagging_freq=5,
                min_data_in_leaf=100,
                n_estimators=300,
                max_depth=5,
                reg_alpha=0.3,
                reg_lambda=0.3,
                random_state=RNG_SEED,
                verbose=-1,
                class_weight='balanced'
            ),
            "xgb": xgb.XGBClassifier(
                n_estimators=250,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.3,
                reg_lambda=0.3,
                random_state=RNG_SEED,
                eval_metric='logloss',
                scale_pos_weight=1
            ),
            "hgb": HistGradientBoostingClassifier(
                max_depth=4,
                max_iter=200,
                learning_rate=0.05,
                min_samples_leaf=50,
                l2_regularization=0.3,
                random_state=RNG_SEED,
                early_stopping=True,
                validation_fraction=0.2
            )
        }

        # Walkforward CV with enhanced evaluation
        def walk_forward_splits(n_train: int, n_folds: int, embargo: int):
            fold_size = n_train // (n_folds + 1)
            for i in range(n_folds):
                train_end = (i + 1) * fold_size
                valid_start = train_end + embargo
                valid_end = min(valid_start + fold_size, n_train)
                if valid_start >= valid_end:
                    continue
                yield np.arange(0, train_end), np.arange(valid_start, valid_end)

        def finance_metrics(y_true, y_pred, tp_pct, sl_pct):
            trades = int(y_pred.sum())
            if trades == 0:
                return {"hit_rate": 0.0, "trades": 0, "expectancy": -sl_pct}
            
            hit_rate = precision_score(y_true, y_pred, zero_division=0)
            expectancy = hit_rate * tp_pct - (1 - hit_rate) * sl_pct
            
            return {
                "hit_rate": float(hit_rate),
                "trades": trades,
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "expectancy": float(expectancy)
            }

        print("🔄 Enhanced walk-forward CV…")
        best_model_name = None
        best_cv_score = -1
        cv_results = {}

        for name, base_model in models.items():
            fold_scores = []
            
            for fold, (tr_idx, va_idx) in enumerate(walk_forward_splits(len(X_train), CV_FOLDS, MAX_BARS)):
                X_tr_fold, y_tr_fold = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
                X_va_fold, y_va_fold = X_train.iloc[va_idx], y_train.iloc[va_idx]
                
                # Preprocessing
                vt, sel, scaler, X_tr_proc, _, keep_cols = enhanced_preprocessing_pipeline(
                    X_tr_fold.values, y_tr_fold.values, K_FEATS
                )
                X_va_proc = apply_preprocessing_pipeline(vt, sel, scaler, X_va_fold.values, keep_cols)
                
                # Train model
                model = deepcopy(base_model)
                model.fit(X_tr_proc, y_tr_fold)
                
                # Predictions
                proba = model.predict_proba(X_va_proc)[:, 1]
                
                # Optimize threshold
                best_score = -1
                for th in np.arange(0.4, 0.8, 0.02):
                    pred = (proba >= th).astype(int)
                    if pred.sum() < MIN_TRADES // 2:
                        continue
                    metrics = finance_metrics(y_va_fold, pred, TP_PCT, SL_PCT)
                    score = metrics["hit_rate"] * 0.7 + metrics["expectancy"] * 30  # Combined score
                    if score > best_score:
                        best_score = score
                
                fold_scores.append(best_score)
                
            cv_score = np.mean(fold_scores) if fold_scores else 0
            cv_results[name] = cv_score
            print(f"   {name}: CV Score = {cv_score:.4f}")
            
            if cv_score > best_cv_score:
                best_cv_score = cv_score
                best_model_name = name

        print(f"🏆 Best model: {best_model_name} (Score: {best_cv_score:.4f})")

        # Final training
        vt, sel, scaler, X_tr_final, sel_idx, keep_cols = enhanced_preprocessing_pipeline(
            X_train.values, y_train.values, K_FEATS
        )
        
        final_model = deepcopy(models[best_model_name])
        final_model.fit(X_tr_final, y_train)

        # Optimize threshold on training set
        train_proba = final_model.predict_proba(X_tr_final)[:, 1]
        best_threshold = 0.5
        best_train_score = -1
        
        for th in np.arange(0.4, 0.8, 0.01):
            pred = (train_proba >= th).astype(int)
            if pred.sum() < MIN_TRADES:
                continue
            metrics = finance_metrics(y_train, pred, TP_PCT, SL_PCT)
            score = metrics["hit_rate"] * 0.7 + metrics["expectancy"] * 30
            if score > best_train_score:
                best_train_score = score
                best_threshold = th

        print(f"✅ Optimal threshold: {best_threshold:.3f}")

        # Test predictions
        X_test_proc = apply_preprocessing_pipeline(vt, sel, scaler, X_test.values, keep_cols)
        test_proba = final_model.predict_proba(X_test_proc)[:, 1]

        # Save results
        prefix = COIN.replace("USDT", "").lower()
        
        # Model artifacts
        model_path = os.path.join(EXPORT_DIR, f"{prefix}_enhanced_model.joblib")
        pipeline_path = os.path.join(EXPORT_DIR, f"{prefix}_enhanced_pipeline.joblib")
        config_path = os.path.join(EXPORT_DIR, f"{prefix}_enhanced_config.json")
        predictions_path = os.path.join(EXPORT_DIR, f"{prefix}_enhanced_predictions.csv")
        
        # Save model and pipeline
        dump(final_model, model_path)
        dump({
            "var_threshold": vt,
            "selector": sel, 
            "scaler": scaler,
            "keep_cols": keep_cols,
            "feature_names": feature_cols
        }, pipeline_path)
        
        # Predictions
        predictions_df = pd.DataFrame({
            "coin": COIN,
            "timestamp": ts_test,
            "pred_prob": test_proba
        })
        predictions_df.to_csv(predictions_path, index=False)
        
        # Configuration
        config = {
            "coin": COIN,
            "model_type": best_model_name,
            "tp_percent": TP_PCT * 100,
            "sl_percent": SL_PCT * 100,
            "max_bars": MAX_BARS,
            "threshold": best_threshold,
            "n_features": len(sel_idx) if sel else X_tr_final.shape[1],
            "cv_results": cv_results,
            "feature_count": len(feature_cols_all)
        }
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\n✅ Enhanced model complete!")
        print(f"   Model: {model_path}")
        print(f"   Pipeline: {pipeline_path}") 
        print(f"   Predictions: {predictions_path}")
        print(f"   Features: {len(feature_cols_all)} → {X_tr_final.shape[1]} selected")