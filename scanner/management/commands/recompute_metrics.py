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
    calculate_price_slope_1h,
    calculate_ema
)

class Command(BaseCommand):
    help = (
        "Recompute OHLCV-derived metrics (fib distances, Bollinger bands, ADX, "
        "change since low/high, 1h price slope, EMA) for RickisMetrics between "
        "March 22 and May 2, 2025 inclusive."
    )

    def handle(self, *args, **options):
        # Inclusive date window
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 5, 2)) + timedelta(days=1)

        qs = RickisMetrics.objects.filter(
            timestamp__gte=start,
            timestamp__lt=end
        ).order_by('timestamp')

        total = qs.count()
        self.stdout.write(f"🔄 Recomputing {total} records from {start.date()} to {end.date() - timedelta(days=1)}...")

        for rm in qs.iterator():
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
            rm.adx = calculate_adx(rm.coin, rm.timestamp)

            # Change since low/high
            rm.change_since_low  = calculate_change_since_low(rm.price, rm.low_24h)
            rm.change_since_high = calculate_change_since_high(rm.price, rm.high_24h)

            # 1h price slope
            rm.price_slope_1h = calculate_price_slope_1h(rm.coin, rm.timestamp)

            # EMA (using 12-period as example)
            rm.ema = calculate_ema(rm.coin, rm.timestamp, window=12)

            rm.save()

        self.stdout.write("✅ All metrics recomputed successfully.")
