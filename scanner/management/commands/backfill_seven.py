from datetime import datetime
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from scanner.helpers import calculate_change_since_high, calculate_change_since_low, calculate_fib_distances
from django.db import close_old_connections

# nohup python manage.py backfill_seven > output.log 2>&1 &
# tail -f output.log

class Command(BaseCommand):
    help = 'Recalculate change since high/low and fib distances for RickisMetrics from March 23 to May 23, 2025.'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 23))
        end = make_aware(datetime(2025, 5, 23))

        metrics = RickisMetrics.objects.filter(
            timestamp__gte=start,
            timestamp__lt=end
        ).select_related('coin')

        count = 0

        for metric in metrics:
            updated = False
            close_old_connections()

            price = metric.price
            high = metric.high_24h
            low = metric.low_24h

            # Skip if price, high, low are missing
            if price is None or high is None or low is None:
                continue

            try:
                # change_since_high
                new_csh = calculate_change_since_high(price, high)
                if new_csh is not None and (metric.change_since_high is None or abs(metric.change_since_high - new_csh) > 0.0001):
                    metric.change_since_high = new_csh
                    updated = True

                # change_since_low
                new_csl = calculate_change_since_low(price, low)
                if new_csl is not None and (metric.change_since_low is None or abs(metric.change_since_low - new_csl) > 0.0001):
                    metric.change_since_low = new_csl
                    updated = True

                # Fibonacci distances
                fib_distances = calculate_fib_distances(high, low, price)

                if fib_distances:
                    for key, value in fib_distances.items():
                        current_val = getattr(metric, key, None)
                        if value is not None and (current_val is None or abs(current_val - value) > 0.0001):
                            setattr(metric, key, value)
                            updated = True

                if updated:
                    metric.save()
                    count += 1
                    if count % 100 == 0:
                        print(f"✅ Updated {count} metrics")

            except Exception as e:
                print(f"❌ Error at {metric.coin.symbol} {metric.timestamp}: {e}")

        print(f"🎯 Done. Total updated: {count}")
