# scanner/management/commands/long_data.py

import csv
import random
from datetime import datetime
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from scanner.models import RickisMetrics

FIELDS = [
    "price", "high_24h", "low_24h", "open", "close",
    "change_5m", "change_1h", "change_24h", "volume", "avg_volume_1h",
    "rsi", "macd", "macd_signal", "stochastic_k", "stochastic_d",
    "support_level", "resistance_level", "relative_volume",
    "sma_5", "sma_20", "stddev_1h", "atr_1h", "change_since_high",
    "change_since_low", "fib_distance_0_236", "fib_distance_0_382",
    "fib_distance_0_5", "fib_distance_0_618", "fib_distance_0_786",
    "adx", "bollinger_upper", "bollinger_middle", "bollinger_lower",
    "long_result"
]

class Command(BaseCommand):
    help = "Export clean long wins and random long losses to CSV."

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 23))
        end = make_aware(datetime(2025, 5, 23))

        print("🚀 Fetching clean long wins...")
        long_wins = RickisMetrics.objects.filter(
            timestamp__gte=start,
            timestamp__lt=end,
            long_result=True
        ).exclude(
            price__isnull=True,
            high_24h__isnull=True,
            low_24h__isnull=True,
            open__isnull=True,
            close__isnull=True,
            change_1h__isnull=True,
            change_24h__isnull=True,
            volume__isnull=True,
            avg_volume_1h__isnull=True,
            rsi__isnull=True,
            macd__isnull=True,
            macd_signal__isnull=True,
            stochastic_k__isnull=True,
            stochastic_d__isnull=True,
            support_level__isnull=True,
            resistance_level__isnull=True,
            relative_volume__isnull=True,
            sma_5__isnull=True,
            sma_20__isnull=True,
            stddev_1h__isnull=True,
            atr_1h__isnull=True,
            change_since_high__isnull=True,
            change_since_low__isnull=True,
            fib_distance_0_236__isnull=True,
            fib_distance_0_382__isnull=True,
            fib_distance_0_5__isnull=True,
            fib_distance_0_618__isnull=True,
            fib_distance_0_786__isnull=True,
            adx__isnull=True,
            bollinger_upper__isnull=True,
            bollinger_middle__isnull=True,
            bollinger_lower__isnull=True,
        ).exclude(
            change_5m=0,
            stddev_1h=0,
            atr_1h=0,
        )

        long_win_count = long_wins.count()
        print(f"✅ Found {long_win_count} clean long wins.")

        print("🚀 Fetching clean long losses...")
        long_losses = RickisMetrics.objects.filter(
            timestamp__gte=start,
            timestamp__lt=end,
            long_result=False
        ).exclude(
            price__isnull=True,
            high_24h__isnull=True,
            low_24h__isnull=True,
            open__isnull=True,
            close__isnull=True,
            change_1h__isnull=True,
            change_24h__isnull=True,
            volume__isnull=True,
            avg_volume_1h__isnull=True,
            rsi__isnull=True,
            macd__isnull=True,
            macd_signal__isnull=True,
            stochastic_k__isnull=True,
            stochastic_d__isnull=True,
            support_level__isnull=True,
            resistance_level__isnull=True,
            relative_volume__isnull=True,
            sma_5__isnull=True,
            sma_20__isnull=True,
            stddev_1h__isnull=True,
            atr_1h__isnull=True,
            change_since_high__isnull=True,
            change_since_low__isnull=True,
            fib_distance_0_236__isnull=True,
            fib_distance_0_382__isnull=True,
            fib_distance_0_5__isnull=True,
            fib_distance_0_618__isnull=True,
            fib_distance_0_786__isnull=True,
            adx__isnull=True,
            bollinger_upper__isnull=True,
            bollinger_middle__isnull=True,
            bollinger_lower__isnull=True,
        ).exclude(
            change_5m=0,
            stddev_1h=0,
            atr_1h=0,
        )

        long_loss_count = long_losses.count()
        print(f"✅ Found {long_loss_count} clean long losses before sampling.")

        sample_size = long_win_count
        print(f"🎯 Sampling {sample_size} long losses...")
        loss_ids = list(long_losses.values_list('id', flat=True))
        sampled_loss_ids = random.sample(loss_ids, sample_size)

        sampled_losses = RickisMetrics.objects.filter(id__in=sampled_loss_ids)

        # Open CSV file for writing
        with open("long_training_data.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(FIELDS)

            print("💾 Writing long wins...")
            for entry in long_wins.iterator(chunk_size=1000):
                writer.writerow([getattr(entry, field) for field in FIELDS])

            print("💾 Writing sampled long losses...")
            for entry in sampled_losses.iterator(chunk_size=1000):
                writer.writerow([getattr(entry, field) for field in FIELDS])

        print("\n✅ CSV export complete: long_training_data.csv")
