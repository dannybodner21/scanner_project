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
    calculate_bollinger_bands,
    calculate_adx,
    calculate_change_since_low,
    calculate_change_since_high,
    calculate_price_slope_1h
)

class Command(BaseCommand):
    help = (
        "Recompute OHLCV-derived metrics (fib distances, Bollinger bands, ADX, "
        "change since low/high, 1h price slope) for RickisMetrics between "
        "March 29 and April 14, 2025 inclusive."
    )

    BATCH_SIZE = 500  # Safer batch size

    def handle(self, *args, **options):
        start = make_aware(datetime(2025, 3, 29))
        end = make_aware(datetime(2025, 4, 1)) + timedelta(days=1)

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
            'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
            'adx', 'change_since_low', 'change_since_high',
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

            # Bollinger Bands
            bb_result = calculate_bollinger_bands(rm.coin, rm.timestamp)
            if bb_result:
                upper, middle, lower = bb_result
                rm.bollinger_upper = upper
                rm.bollinger_middle = middle
                rm.bollinger_lower = lower
            else:
                rm.bollinger_upper = None
                rm.bollinger_middle = None
                rm.bollinger_lower = None

            # ADX
            adx = calculate_adx(rm.coin, rm.timestamp)
            rm.adx = adx if adx is not None else None

            # Change Since Low/High
            rm.change_since_low = calculate_change_since_low(rm.price, rm.low_24h)
            rm.change_since_high = calculate_change_since_high(rm.price, rm.high_24h)

            # Price Slope 1h
            slope = calculate_price_slope_1h(rm.coin, rm.timestamp)
            rm.price_slope_1h = slope if slope is not None else None

            # Add to batch
            to_update.append(rm)
            processed += 1

            if len(to_update) >= self.BATCH_SIZE:
                RickisMetrics.objects.bulk_update(to_update, fields)
                self.stdout.write(f"✅ {processed}/{total} records processed...")
                to_update.clear()

        # Final batch
        if to_update:
            RickisMetrics.objects.bulk_update(to_update, fields)
            self.stdout.write(f"✅ {processed}/{total} records processed (final batch).")

        self.stdout.write("🎉 Completed Mar 29–Apr 14 recomputation.")
