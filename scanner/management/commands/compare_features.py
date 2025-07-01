import os
import json
import pandas as pd
import numpy as np
import ta
from scipy.stats import linregress
from datetime import timedelta, datetime, timezone

from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice

JSON_PATH = "live_predictions_log.jsonl"

FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume', 'adx_14', 'ma_200', 'returns_5m', 'returns_15m',
    'returns_1h', 'returns_4h', 'momentum', 'volume_ma_20', 'vol_spike', 'rsi_14', 'macd',
    'macd_signal', 'macd_hist', 'bb_upper', 'bb_lower', 'atr_14', 'obv', 'obv_slope',
    'ema_9', 'ema_21', 'ema_diff', 'volatility', 'bull_regime', 'bear_regime', 'sideways_regime',
    'slope_1h', 'dist_from_high_24h', 'dist_from_low_24h', 'stoch_k', 'stoch_d',
    'price_change_5', 'volume_change_5', 'high_1h', 'low_1h', 'pos_in_range_1h',
    'vwap_1h', 'pos_vs_vwap'
]

class Command(BaseCommand):
    help = "Compare features in JSON log to recomputed features"

    def handle(self, *args, **kwargs):
        if not os.path.exists(JSON_PATH):
            self.stderr.write(f"JSON log file not found at {JSON_PATH}")
            return

        with open(JSON_PATH, "r") as f:
            lines = f.readlines()

        for line in lines:
            try:
                record = json.loads(line)
                coin = record["coin"]
                ts = pd.to_datetime(record["timestamp"])
                stored = record["features"]

                recomputed = self.compute_features(coin, ts)
                if not recomputed:
                    self.stderr.write(f"Could not recompute features for {coin} at {ts}")
                    continue

                mismatches = []
                for col in FEATURE_COLUMNS:
                    v1 = stored.get(col)
                    v2 = recomputed.get(col)
                    if v1 is None or v2 is None:
                        mismatches.append((col, v1, v2))
                        continue
                    if abs(float(v1) - float(v2)) > 1e-5:
                        mismatches.append((col, v1, v2))

                if mismatches:
                    self.stdout.write(f"\n🔍 {ts} - {coin} - {len(mismatches)} mismatches:")
                    for col, old, new in mismatches:
                        self.stdout.write(f"  ❌ {col}: stored={old}, recomputed={new}")
                else:
                    self.stdout.write(f"✅ {ts} - {coin} - All features match.")

            except Exception as e:
                self.stderr.write(f"Error on line: {e}")

    def compute_features(self, coin, ts):
        start = ts - timedelta(hours=24)
        end = ts + timedelta(minutes=5)

        queryset = CoinAPIPrice.objects.filter(
            coin=coin,
            timestamp__gte=start,
            timestamp__lte=end
        ).order_by("timestamp")

        df = pd.DataFrame(list(queryset.values("timestamp", "open", "high", "low", "close", "volume")))
        if df.empty or len(df) < 200:
            return None

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df.set_index("timestamp", inplace=True)
        df = self.add_features(df)

        if ts not in df.index:
            return None

        row = df.loc[ts]
        return row[FEATURE_COLUMNS].to_dict()

    def calculate_trend_slope(self, prices):
        if len(prices) < 12:
            return np.nan
        x = np.arange(len(prices))
        slope, _, _, _, _ = linregress(x, prices)
        return slope

    def add_features(self, df):
        df = df.copy()

        df['adx_14'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df['ma_200'] = ta.trend.sma_indicator(df['close'], window=200)

        df['returns_5m'] = df['close'].pct_change(1).clip(-1, 1)
        df['returns_15m'] = df['close'].pct_change(3).clip(-1, 1)
        df['returns_1h'] = df['close'].pct_change(12).clip(-1, 1)
        df['returns_4h'] = df['close'].pct_change(48).clip(-1, 1)
        df['momentum'] = df['close'] - df['close'].shift(5)

        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['vol_spike'] = df['volume'] / df['volume_ma_20']

        df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()

        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()

        df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['obv_slope'] = df['obv'].diff()

        df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['ema_diff'] = df['ema_9'] - df['ema_21']

        df['volatility'] = df['close'].rolling(20).std()

        adx_thresh = df['adx_14'].quantile(0.5)
        df['bull_regime'] = ((df['adx_14'] > adx_thresh) & (df['close'] > df['ma_200'])).astype(int)
        df['bear_regime'] = ((df['adx_14'] > adx_thresh) & (df['close'] < df['ma_200'])).astype(int)
        df['sideways_regime'] = (df['adx_14'] <= adx_thresh).astype(int)

        df['slope_1h'] = df['close'].rolling(12).apply(self.calculate_trend_slope, raw=False)
        df['dist_from_high_24h'] = ((df['close'] - df['high'].rolling(288).max()) / df['high'].rolling(288).max()).clip(-1, 1)
        df['dist_from_low_24h'] = ((df['close'] - df['low'].rolling(288).min()) / df['low'].rolling(288).min()).clip(-1, 5)

        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        df['price_change_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        df['volume_change_5'] = (df['volume'] - df['volume'].shift(5)) / df['volume'].shift(5)

        df['high_1h'] = df['high'].rolling(12).max()
        df['low_1h'] = df['low'].rolling(12).min()
        df['pos_in_range_1h'] = (df['close'] - df['low_1h']) / (df['high_1h'] - df['low_1h'])
        df['vwap_1h'] = (df['close'] * df['volume']).rolling(12).sum() / df['volume'].rolling(12).sum()
        df['pos_vs_vwap'] = df['close'] - df['vwap_1h']

        return df.dropna()
