import pandas as pd
import ta
from datetime import datetime, timezone
from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice
import numpy as np
from sklearn.utils import shuffle

class Command(BaseCommand):
    help = 'Generate training and 2025 test dataset CSVs with features, regime, and labels for long trades'

    def handle(self, *args, **options):
        coins = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'LTCUSDT']
        dfs = []

        print("Loading and processing coins...")
        for coin in coins:
            print(f"Processing {coin}...")
            df = self.load_data(coin)
            df = self.add_features_and_regime(df)
            df = self.generate_labels(df, tp=0.06, sl=0.03, window=288)  # 24h window
            df['coin'] = coin
            dfs.append(df)

        full_df = pd.concat(dfs)
        full_df = full_df.sort_index()

        # Split training and 2025 test data
        split_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        train_df = full_df[full_df.index < split_date]
        test_df = full_df[full_df.index >= split_date]

        print(f"Total rows: {len(full_df)}")
        print(f"Training rows: {len(train_df)}")
        print(f"2025 Test rows: {len(test_df)}")

        # Balance training dataset wins and losses
        wins = train_df[train_df['label'] == 1]
        losses = train_df[train_df['label'] == 0]

        min_len = min(len(wins), len(losses))
        wins_bal = wins.sample(min_len, random_state=42)
        losses_bal = losses.sample(min_len, random_state=42)

        train_balanced = pd.concat([wins_bal, losses_bal])
        train_balanced = shuffle(train_balanced, random_state=42)

        print(f"Balanced training data size: {len(train_balanced)} (wins={min_len}, losses={min_len})")

        # Save to CSV
        train_balanced.to_csv('new_long_training_data.csv')
        test_df.to_csv('new_long_test_data.csv')

        print("Saved training and test CSV files.")

    def load_data(self, coin):
        start_date = datetime(2019, 1, 1, tzinfo=timezone.utc)
        end_date = datetime.now(timezone.utc)

        queryset = CoinAPIPrice.objects.filter(
            coin=coin,
            timestamp__gte=start_date,
            timestamp__lte=end_date
        ).order_by('timestamp')

        df = pd.DataFrame(list(queryset.values('timestamp', 'open', 'high', 'low', 'close', 'volume')))
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df = df.set_index('timestamp').sort_index()
        return df

    def add_features_and_regime(self, df):
        df['returns_5m'] = df['close'].pct_change(1)
        df['returns_15m'] = df['close'].pct_change(3)
        df['returns_1h'] = df['close'].pct_change(12)
        df['returns_4h'] = df['close'].pct_change(48)

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
        df['adx_14'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)

        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['obv_slope'] = df['obv'].diff(5)

        df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['ema_diff'] = df['ema_9'] - df['ema_21']

        # Market regime (bull/bear/sideways)
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df['ma_200'] = df['close'].rolling(200).mean()
        median_vol = df['volatility'].median()

        df['bull_regime'] = ((df['close'] > df['ma_200']) & (df['volatility'] < median_vol)).astype(int)
        df['bear_regime'] = ((df['close'] < df['ma_200']) & (df['volatility'] > median_vol)).astype(int)
        df['sideways_regime'] = 1 - df['bull_regime'] - df['bear_regime']

        return df.dropna()

    def generate_labels(self, df, tp=0.06, sl=0.03, window=288):
        df = df.copy()
        df['label'] = 0
        high_rolling = df['high'].rolling(window).max().shift(-window + 1)
        low_rolling = df['low'].rolling(window).min().shift(-window + 1)
        close = df['close']

        tp_hit = high_rolling >= close * (1 + tp)
        sl_hit = low_rolling <= close * (1 - sl)

        df.loc[tp_hit & (~sl_hit), 'label'] = 1
        return df.dropna()
