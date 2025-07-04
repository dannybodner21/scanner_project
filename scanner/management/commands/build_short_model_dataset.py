import pandas as pd
import numpy as np
import ta
from scipy.stats import linregress
from datetime import datetime, timezone
from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice

class Command(BaseCommand):
    help = 'Build dataset with engineered features for short trade model'

    def handle(self, *args, **options):

        coins = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'LTCUSDT', 'SOLUSDT', 'DOGEUSDT', 'LINKUSDT', 'DOTUSDT', 'SHIBUSDT', 'ADAUSDT', 'UNIUSDT', 'AVAXUSDT', 'XLMUSDT']
        start_date = datetime(2022, 1, 1, tzinfo=timezone.utc)
        #end_date = datetime.now(timezone.utc)
        end_date = datetime(2025, 6, 30, 23, 55, tzinfo=timezone.utc)

        dfs = []
        for coin in coins:
            self.stdout.write(f"Loading data for {coin}...")
            df = self.load_data(coin, start_date, end_date)
            if df.empty:
                self.stdout.write(f"No data for {coin}, skipping.")
                continue
            df['coin'] = coin
            dfs.append(df)

        full_df = pd.concat(dfs).sort_index()
        self.stdout.write(f"Loaded total {len(full_df)} rows for all coins.")

        full_df = self.add_features(full_df)
        self.stdout.write("Features engineered.")


        full_df = self.generate_labels(full_df, tp=0.04, sl=0.02, window=288)
        self.stdout.write("Labels generated.")

        test_df = full_df[full_df.index >= datetime(2025, 6, 15, tzinfo=timezone.utc)].copy()
        train_df = full_df[full_df.index < datetime(2025, 6, 15, tzinfo=timezone.utc)].copy()

        balanced_train_df = self.balance_data(train_df)

        self.stdout.write(f"Training data rows (balanced): {len(balanced_train_df)}")
        self.stdout.write(f"Test data rows (unbalanced): {len(test_df)}")

        balanced_train_df.to_csv("seven_short_training_data.csv")
        test_df.to_csv("seven_short_testing_data.csv")
        self.stdout.write("Training and test CSV files saved.")


    def load_data(self, coin, start, end):
        queryset = CoinAPIPrice.objects.filter(
            coin=coin,
            timestamp__gte=start,
            timestamp__lte=end
        ).order_by('timestamp')

        df = pd.DataFrame(list(queryset.values('timestamp','open','high','low','close','volume')))
        if df.empty:
            return df

        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)

        df = df.set_index('timestamp').sort_index()
        return df

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
        df = df[df['adx_14'].notna() & df['ma_200'].notna()].copy()

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

        df = df.dropna()
        return df

    def generate_labels(self, df, tp=0.04, sl=0.02, window=288):
        df = df.copy()
        df['label'] = 0

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        labels = []
        for i in range(len(df) - window):
            entry = close[i]
            label = 0
            for j in range(1, window):
                future_high = high[i + j]
                future_low = low[i + j]
                if future_low <= entry * (1 - tp):
                    label = 1
                    break
                if future_high >= entry * (1 + sl):
                    label = 0
                    break
            labels.append(label)

        df = df.iloc[:len(labels)]
        df['label'] = labels
        return df.dropna()

    def balance_data(self, df):
        wins = df[df['label'] == 1]
        losses = df[df['label'] == 0]

        min_len = min(len(wins), len(losses))

        if min_len == 0:
            self.stdout.write("No balance possible (zero of one class). Returning unbalanced data.")
            return df

        wins_sampled = wins.sample(min_len, random_state=42)
        losses_sampled = losses.sample(min_len, random_state=42)

        balanced = pd.concat([wins_sampled, losses_sampled]).sample(frac=1, random_state=42)
        return balanced
