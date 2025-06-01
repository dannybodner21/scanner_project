# scanner/management/commands/export_long_training_data.py

import csv
import random
from datetime import datetime
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from scanner.models import RickisMetrics

GOOD_FIELDS = [
    "price", "high_24h", "low_24h", "open", "close", "change_5m", "change_1h", "change_24h",
    "volume", "avg_volume_1h", "rsi", "macd", "macd_signal", "stochastic_k", "stochastic_d",
    "support_level", "resistance_level", "relative_volume", "sma_5", "sma_20",
    "stddev_1h", "atr_1h", "change_since_high", "change_since_low",
    "fib_distance_0_236", "fib_distance_0_382", "fib_distance_0_5", "fib_distance_0_618", "fib_distance_0_786",
    "adx", "bollinger_upper", "bollinger_middle", "bollinger_lower"
]

ZERO_FIELDS = ["change_5m", "stddev_1h", "atr_1h"]

class Command(BaseCommand):
    help = "Export clean long trade data: all wins + sampled losses with no nulls or zeros"

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 23))
        end = make_aware(datetime(2025, 5, 23))

        print("🚀 Fetching clean long wins...")

        long_wins = RickisMetrics.objects.filter(
            timestamp__gte=start,
            timestamp__lt=end,
            long_result=True
        ).exclude(
            **{f"{field}__isnull": True for field in GOOD_FIELDS}
        ).exclude(
            **{f"{field}": 0 for field in ZERO_FIELDS}
        ).select_related('coin')

        print(f"✅ Found {long_wins.count()} long wins.")

        print("🚀 Fetching clean long losses...")

        long_losses = RickisMetrics.objects.filter(
            timestamp__gte=start,
            timestamp__lt=end,
            long_result=False
        ).exclude(
            **{f"{field}__isnull": True for field in GOOD_FIELDS}
        ).exclude(
            **{f"{field}": 0 for field in ZERO_FIELDS}
        ).select_related('coin')

        print(f"✅ Found {long_losses.count()} long losses before sampling.")

        long_wins_list = list(long_wins)
        long_losses_list = list(long_losses)

        sample_size = len(long_wins_list)
        print(f"🎯 Sampling {sample_size} long losses to balance.")

        sampled_losses = random.sample(long_losses_list, sample_size)

        all_rows = long_wins_list + sampled_losses
        random.shuffle(all_rows)

        print(f"📦 Exporting {len(all_rows)} rows to CSV...")

        with open('long_training_data.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            header = [field for field in GOOD_FIELDS] + ['target']

            writer.writerow(header)

            for entry in all_rows:
                row = [
                    getattr(entry, field) for field in GOOD_FIELDS
                ]
                row.append(1 if entry.long_result else 0)  # 1 = win, 0 = loss
                writer.writerow(row)

        print("✅ Export completed: long_training_data.csv")
