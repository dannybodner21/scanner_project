import pandas as pd
import ta
from datetime import datetime
from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice

class Command(BaseCommand):
    help = 'Find best volume spike threshold for long trades'

    def handle(self, *args, **options):
        coin = 'BTCUSDT'
        df = self.load_data(coin)
        self.stdout.write(self.style.SUCCESS(f"Loaded {len(df)} rows."))

        df = self.add_indicators(df)
        self.stdout.write(self.style.SUCCESS("Indicators added."))

        best_threshold = None
        best_win_rate = 0
        best_num_trades = 0

        tp_pct = 0.04
        sl_pct = 0.02
        window = 288  # 24h

        for vol_th in [round(x * 0.1, 2) for x in range(10, 31)]:  # 1.0 to 3.0 step 0.1
            signals = df[df['vol_spike'] > vol_th]
            if signals.empty:
                continue
            win_rate, num_trades = self.backtest(df, signals, tp_pct, sl_pct, window)
            if num_trades < 10:
                continue
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_threshold = vol_th
                best_num_trades = num_trades

        self.stdout.write(self.style.SUCCESS(
            f"Best volume spike threshold: {best_threshold}, Win rate: {best_win_rate:.2%}, Trades: {best_num_trades}"
        ))

    def load_data(self, coin):
        start_date = datetime(2019,1,1)
        end_date = datetime(2025,6,13)

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
        df = df.dropna()
        return df

    def backtest(self, df, signals, tp_pct, sl_pct, window):
        wins = 0
        losses = 0
        for ts in signals.index:
            try:
                idx = df.index.get_loc(ts)
            except KeyError:
                continue
            entry_price = df.iloc[idx]['close']
            future_df = df.iloc[idx:idx+window]
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
            result = None
            for _, row in future_df.iterrows():
                if row['high'] >= tp_price:
                    result = 'win'
                    break
                if row['low'] <= sl_price:
                    result = 'loss'
                    break
            if result == 'win':
                wins += 1
            elif result == 'loss':
                losses += 1
        total = wins + losses
        win_rate = wins / total if total > 0 else 0
        return win_rate, total
