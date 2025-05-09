# File: recompute_metrics_apr20_apr24.py
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="ta.trend")

from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
from scanner.models import RickisMetrics
from scanner.helpers import (
    calculate_fib_distances,
    calculate_change_since_low,
    calculate_change_since_high
)

class Command(BaseCommand):
    help = (
        "Fast recompute of OHLCV metrics (fib distances, change since low/high) "
        "for RickisMetrics between April 20 and April 24, 2025 inclusive."
    )

    BATCH_SIZE = 250

    def handle(self, *args, **options):
        start = make_aware(datetime(2025, 4, 4))
        end = make_aware(datetime(2025, 4, 7))

        qs = (
            RickisMetrics.objects
            .filter(timestamp__gte=start, timestamp__lt=end)
            .select_related('coin')
            .only(
                "coin", "timestamp", "price", "high_24h", "low_24h",
                "fib_distance_0_236", "fib_distance_0_382", "fib_distance_0_5",
                "fib_distance_0_618", "fib_distance_0_786",
                "change_since_low", "change_since_high"
            )
            .order_by('timestamp')
        )

        total = qs.count()
        processed = 0
        self.stdout.write(f"🔄 Recomputing {total} records (Apr 20–Apr 24) in batches of {self.BATCH_SIZE}...")

        to_update = []
        fields = [
            'fib_distance_0_236', 'fib_distance_0_382', 'fib_distance_0_5',
            'fib_distance_0_618', 'fib_distance_0_786',
            'change_since_low', 'change_since_high'
        ]

        for rm in qs.iterator():
            fibs = calculate_fib_distances(rm.high_24h, rm.low_24h, rm.price) or {}
            rm.fib_distance_0_236 = fibs.get('fib_distance_0_236')
            rm.fib_distance_0_382 = fibs.get('fib_distance_0_382')
            rm.fib_distance_0_5   = fibs.get('fib_distance_0_5')
            rm.fib_distance_0_618 = fibs.get('fib_distance_0_618')
            rm.fib_distance_0_786 = fibs.get('fib_distance_0_786')

            rm.change_since_low  = calculate_change_since_low(rm.price, rm.low_24h)
            rm.change_since_high = calculate_change_since_high(rm.price, rm.high_24h)

            to_update.append(rm)
            processed += 1

            if len(to_update) >= self.BATCH_SIZE:
                RickisMetrics.objects.bulk_update(to_update, fields)
                self.stdout.write(f"✅ {processed}/{total} records processed...")
                to_update.clear()

        if to_update:
            RickisMetrics.objects.bulk_update(to_update, fields)
            self.stdout.write(f"✅ {processed}/{total} records processed (final batch).")

        self.stdout.write("🎉 Completed Apr 20–Apr 24 recomputation.")
