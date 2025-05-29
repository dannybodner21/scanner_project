from datetime import datetime
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from scanner.helpers import (
    calculate_stochastic, calculate_atr_1h, calculate_bollinger_bands,
    calculate_adx, calculate_support_resistance, calculate_relative_volume,
    calculate_sma, calculate_stddev_1h, calculate_obv,
    calculate_change_since_high, calculate_change_since_low, calculate_fib_distances
)

# nohup python manage.py backfill_six_calc > output.log 2>&1 &
# tail -f output.log

# ps aux | grep backfill_six_calc

class Command(BaseCommand):
    help = 'Recalculate fixed metrics for RickisMetrics from March 23 to April 1, 2025'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 23))
        end = make_aware(datetime(2025, 4, 23))

        metrics = RickisMetrics.objects.filter(timestamp__gte=start, timestamp__lt=end).select_related("coin")
        count = 0

        for metric in metrics:
            coin = metric.coin
            timestamp = metric.timestamp
            updated = False

            try:


                if metric.obv in [None, 0]:
                    obv = calculate_obv(coin, timestamp)
                    if obv is not None:
                        metric.obv = obv
                        updated = True

                # 🆕 Change Since High / Low
                if metric.change_since_high in [None, 0] and metric.price and metric.high_24h:
                    change_high = calculate_change_since_high(metric.price, metric.high_24h)
                    if change_high is not None:
                        metric.change_since_high = change_high
                        updated = True

                if metric.change_since_low in [None, 0] and metric.price and metric.low_24h:
                    change_low = calculate_change_since_low(metric.price, metric.low_24h)
                    if change_low is not None:
                        metric.change_since_low = change_low
                        updated = True

                # 🆕 Fibonacci Distances
                if any(getattr(metric, f"fib_distance_{level}") in [None, 0] for level in ['0_236', '0_382', '0_5', '0_618', '0_786']):
                    fibs = calculate_fib_distances(metric.high_24h, metric.low_24h, metric.price)
                    if fibs:
                        for key, val in fibs.items():
                            setattr(metric, key, val)
                        updated = True

                if updated:
                    metric.save()
                    count += 1
                    print(f"✅ Updated {count} metrics")

            except Exception as e:
                print(f"❌ Error at {coin.symbol} {timestamp}: {e}")

        print(f"✅ Done. Updated {count} metrics")
