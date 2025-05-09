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
        "Recompute OHLCV-derived metrics in bulk for RickisMetrics between "
        "March 22 and May 2, 2025, excluding EMA and volume_mc_ratio fields."
    )

    BATCH_SIZE = 2000

    def handle(self, *args, **options):
        # Define date window (inclusive of May 2)
        start = make_aware(datetime(2025, 3, 29))
        end = make_aware(datetime(2025, 5, 2)) + timedelta(days=1)

        qs = (
            RickisMetrics.objects
            .filter(timestamp__gte=start, timestamp__lt=end)
            .select_related('coin')
            .order_by('timestamp')
        )

        total = qs.count()
        self.stdout.write(
            f"🔄 Recomputing {total} records in batches of {self.BATCH_SIZE}..."
        )

        to_update = []
        fields = [
            'fib_distance_0_236', 'fib_distance_0_382', 'fib_distance_0_5',
            'fib_distance_0_618', 'fib_distance_0_786',
            'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
            'adx', 'change_since_low', 'change_since_high',
            'price_slope_1h'
        ]

        for i, rm in enumerate(qs.iterator(), start=1):
            # Fibonacci distances
            fibs = calculate_fib_distances(rm.high_24h, rm.low_24h, rm.price)
            rm.fib_distance_0_236 = fibs.get('fib_distance_0_236')
            rm.fib_distance_0_382 = fibs.get('fib_distance_0_382')
            rm.fib_distance_0_5   = fibs.get('fib_distance_0_5')
            rm.fib_distance_0_618 = fibs.get('fib_distance_0_618')
            rm.fib_distance_0_786 = fibs.get('fib_distance_0_786')

            # Bollinger Bands
            upper, middle, lower = calculate_bollinger_bands(rm.coin, rm.timestamp)
            rm.bollinger_upper  = upper
            rm.bollinger_middle = middle
            rm.bollinger_lower  = lower

            # ADX
            try:
                rm.adx = calculate_adx(rm.coin, rm.timestamp)
            except Exception:
                rm.adx = None

            # Change since low/high
            rm.change_since_low  = calculate_change_since_low(rm.price, rm.low_24h)
            rm.change_since_high = calculate_change_since_high(rm.price, rm.high_24h)

            # 1h price slope
            rm.price_slope_1h = calculate_price_slope_1h(rm.coin, rm.timestamp)

            to_update.append(rm)

            # Bulk update in batches
            if len(to_update) >= self.BATCH_SIZE:
                RickisMetrics.objects.bulk_update(to_update, fields)
                to_update.clear()

        # Final bulk update of any remaining records
        if to_update:
            RickisMetrics.objects.bulk_update(to_update, fields)

        self.stdout.write("🎉 All metrics recomputed successfully.")
