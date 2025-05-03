from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
import pandas as pd

class Command(BaseCommand):
    help = 'Check RickisMetrics completeness between March 22 and May 2'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 5, 3))

        metrics = RickisMetrics.objects.filter(timestamp__gte=start, timestamp__lt=end)
        df = pd.DataFrame(list(metrics.values(
            'coin', 'timestamp', 'rsi', 'macd', 'macd_signal', 'stochastic_k', 'stochastic_d',
            'support_level', 'resistance_level', 'price_slope_1h', 'relative_volume',
            'sma_5', 'sma_20', 'ema_12', 'ema_26', 'stddev_1h', 'atr_1h'
        )))

        print(f"Total entries: {len(df)}")

        # 1. Missing values
        excluded = ['long_result', 'short_result']
        missing = df.drop(columns=excluded, errors='ignore').isnull().sum()

        print("\n🚫 Missing fields:")
        print(missing[missing > 0])

        # 2. Count per coin
        coin_counts = df['coin'].value_counts()
        print("\n📊 Entries per coin:")
        print(coin_counts)

        # 3. Expected vs actual count per coin
        expected_count = int(((end - start).total_seconds() / 300))  # 5-min intervals
        print(f"\nExpected intervals per coin: {expected_count}")

        missing_by_coin = coin_counts[coin_counts < expected_count]
        print("\n⚠️ Coins with missing intervals:")
        print(missing_by_coin)

        print("\n✅ Done checking data completeness.")
