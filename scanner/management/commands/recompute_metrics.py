# File: recompute_metrics_mar29_apr14.py

import warnings
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module="ta.trend"
)

from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
from scanner.models import RickisMetrics
from scanner.helpers import (
    calculate_fib_distances,
    calculate_change_since_low,
    calculate_change_since_high,
    calculate_price_slope_1h
)

class Command(BaseCommand):
    help = (
        "Fast recompute of core OHLCV-derived metrics (fib distances, change since low/high, price slope) "
        "for RickisMetrics between March 29 and April 14, 2025 inclusive."
    )

    BATCH_SIZE = 500

    def handle(self, *args, **options):
        start = make_aware(datetime(2025, 4, 4))
        end = make_aware(datetime(2025, 4, 9)) + timedelta(days=1)

        qs = (
            RickisMetrics.objects
            .filter(timestamp__gte=start, timestamp__lt=end)
            .select_related('coin')
            .order_by('timestamp')
        )

        total = qs.count()
        processed = 0
        self.stdout.write(f"🔄 Recomputing {total} records (Mar 29–Apr 14) in batches of {self.BATCH_SIZE}...")

        to_update = []
        fields = [
            'fib_distance_0_236', 'fib_distance_0_382', 'fib_distance_0_5',
            'fib_distance_0_618', 'fib_distance_0_786',
            'change_since_low', 'change_since_high',
            'price_slope_1h'
        ]

        for rm in qs.iterator():
            # Fibonacci Distances
            fibs = calculate_fib_distances(rm.high_24h, rm.low_24h, rm.price) or {}
            rm.fib_distance_0_236 = fibs.get('fib_distance_0_236')
            rm.fib_distance_0_382 = fibs.get('fib_distance_0_382')
            rm.fib_distance_0_5   = fibs.get('fib_distance_0_5')
            rm.fib_distance_0_618 = fibs.get('fib_distance_0_618')
            rm.fib_distance_0_786 = fibs.get('fib_distance_0_786')

            # Change Since Low/High
            rm.change_since_low  = calculate_change_since_low(rm.price, rm.low_24h)
            rm.change_since_high = calculate_change_since_high(rm.price, rm.high_24h)

            # Price Slope 1h
            rm.price_slope_1h = calculate_price_slope_1h(rm.coin, rm.timestamp)

            # Add to batch
            to_update.append(rm)
            processed += 1

            if len(to_update) >= self.BATCH_SIZE:
                RickisMetrics.objects.bulk_update(to_update, fields)
                self.stdout.write(f"✅ {processed}/{total} records processed...")
                to_update.clear()

        if to_update:
            RickisMetrics.objects.bulk_update(to_update, fields)
            self.stdout.write(f"✅ {processed}/{total} records processed (final batch).")

        self.stdout.write("🎉 Completed Mar 29–Apr 14 recomputation.")
