import pandas as pd
import ta
from datetime import datetime, timezone
from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice
import random

class Command(BaseCommand):
    help = 'Search indicator thresholds on BTC filtered by bullish market regime'

    def handle(self, *args, **options):
        print("Loading BTC data...")
        df = self.load_data('BTCUSDT')
        print(f"Loaded {len(df)} rows.")

        print("Calculating indicators and regime...")
        df = self.add_indicators(df)
        df = self.add_market_regime(df)
        print("Indicators and regime added.")

        tp_pct = 0.04
        sl_pct = 0.02
        window = 288  # 24h ahead for labels

        print("Generating labels...")
        df = self.generate_labels(df, tp=tp_pct, sl=sl_pct, window=window)
        print(f"Total labeled rows: {len(df)}")

        # Filter only bullish regime rows
        df = df[df['bull_regime'] == 1]
        print(f"Rows in bullish regime: {len(df)}")

        # Define full parameter ranges for each indicator
        rvol_thresholds_full = [round(x * 0.1, 2) for x in range(25, 41)]        # 2.5 to 4.0
        momentum_thresholds_full = [round(x * 0.1, 2) for x in range(1, 21)]     # 0.1 to 2.0
        rsi_thresholds_full = [round(x, 2) for x in range(30, 60, 5)]            # 30 to 55
        macd_hist_thresholds_full = [round(x * 0.01, 3) for x in range(0, 51)]   # 0.00 to 0.50
        atr_thresholds_full = [round(x * 0.01, 3) for x in range(1, 11)]         # 0.01 to 0.10
        obv_slope_thresholds_full = [round(x * 0.1, 2) for x in range(0, 11)]    # 0.0 to 1.0
        ema_diff_thresholds_full = [round(x * 0.1, 2) for x in range(0, 21)]     # 0.0 to 2.0

        sample_size = 50  # total combos to test

        combos = set()
        while len(combos) < sample_size:
            combo = (
                random.choice(rvol_thresholds_full),
                random.choice(momentum_thresholds_full),
                random.choice(rsi_thresholds_full),
                random.choice(macd_hist_thresholds_full),
                random.choice(atr_thresholds_full),
                random.choice(obv_slope_thresholds_full),
                random.choice(ema_diff_thresholds_full)
            )
            combos.add(combo)

        best_score = 0
        best_params = None

        for count, (rvol_th, mom_th, rsi_th, macd_th, atr_th, obv_th, ema_th) in enumerate(combos, 1):
            print(f"Testing combo {count}/{sample_size}:")
            print(f"  RVOL>{rvol_th}, MOM>{mom_th}, RSI>{rsi_th}, MACD_Hist>{macd_th}, ATR>{atr_th}, OBV_Slope>{obv_th}, EMA_Diff>{ema_th}")

            signals = df[
                (df['vol_spike'] > rvol_th) &
                (df['momentum'] > mom_th) &
                (df['rsi_14'] > rsi_th) &
                (df['macd_hist'] > macd_th) &
                (df['atr_14'] > atr_th) &
                (df['obv_slope'] > obv_th) &
                (df['ema_diff'] > ema_th)
            ]

            if len(signals) < 10:
                print(f"  Skipping combo due to too few signals: {len(signals)}")
                continue

            win_rate, trades = self.backtest(df, signals, tp_pct, sl_pct, window)
            print(f"  Win rate: {win_rate:.2%} over {trades} trades")

            if win_rate > best_score:
                best_score = win_rate
                best_params = (rvol_th, mom_th, rsi_th, macd_th, atr_th, obv_th, ema_th)

        if best_params:
            print(f"Best combo found in bullish regime:")
            print(f"  RVOL > {best_params[0]}, MOM > {best_params[1]}, RSI > {best_params[2]}, MACD_Hist > {best_params[3]}, ATR > {best_params[4]}, OBV_Slope > {best_params[5]}, EMA_Diff > {best_params[6]}")
            print(f"With win rate: {best_score:.2%}")
        else:
            print("No valid combos found with enough signals in bullish regime.")

    def load_data(self, coin):
        start_date = datetime(2019, 1, 1, tzinfo=timezone.utc)
        end_date = datetime.now(timezone.utc)

        queryset = CoinAPIPrice.objects.filter(
            coin=coin,
            timestamp__gte=start_date,
            timestamp__lte=end_date
        ).order_by('timestamp')

        df = pd.DataFrame(list(queryset.values('timestamp','open','high','low','close','volume')))
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        df = df.set_index('timestamp').sort_index()
        return df

    def add_indicators(self, df):
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['vol_spike'] = df['volume'] / df['volume_ma_20']
        df['momentum'] = df['close'] - df['close'].shift(5)

        df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)

        macd = ta.trend.MACD(df['close'])
        df['macd_hist'] = macd.macd_diff()

        df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['obv_slope'] = df['obv'].diff(5)

        df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['ema_diff'] = df['ema_9'] - df['ema_21']

        df = df.dropna()
        return df

    def add_market_regime(self, df):
        # Define regime: Bullish if volatility low AND close above 200 MA
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df['ma_200'] = df['close'].rolling(200).mean()
        df['bull_regime'] = ((df['volatility'] < df['volatility'].quantile(0.5)) & (df['close'] > df['ma_200'])).astype(int)
        return df.dropna()

    def generate_labels(self, df, tp=0.01, sl=0.10, window=288):
        df = df.copy()
        df['label'] = 0
        high_rolling = df['high'].rolling(window).max().shift(-window+1)
        low_rolling = df['low'].rolling(window).min().shift(-window+1)
        close = df['close']

        tp_hit = high_rolling >= close * (1 + tp)
        sl_hit = low_rolling <= close * (1 - sl)

        df.loc[tp_hit & (~sl_hit), 'label'] = 1
        return df.dropna()

    def backtest(self, df, signals, tp_pct, sl_pct, window):
        high_max_future = df['high'].rolling(window).max().shift(-window + 1)
        low_min_future = df['low'].rolling(window).min().shift(-window + 1)

        wins = 0
        losses = 0

        for ts in signals.index:
            if ts not in df.index:
                continue

            entry_price = df.at[ts, 'close']
            max_high = high_max_future.get(ts, None)
            min_low = low_min_future.get(ts, None)

            if pd.isna(max_high) or pd.isna(min_low):
                continue

            tp_hit = max_high >= entry_price * (1 + tp_pct)
            sl_hit = min_low <= entry_price * (1 - sl_pct)

            if tp_hit and not sl_hit:
                wins += 1
            elif sl_hit and not tp_hit:
                losses += 1

        total = wins + losses
        win_rate = wins / total if total > 0 else 0
        return win_rate, total
