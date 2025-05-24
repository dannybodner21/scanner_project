from datetime import datetime
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from scanner.helpers import (
    calculate_stochastic, calculate_atr_1h, calculate_bollinger_bands,
    calculate_adx, calculate_support_resistance, calculate_relative_volume,
    calculate_sma, calculate_stddev_1h, calculate_obv
)

# nohup python manage.py backfill_five_calc > output.log 2>&1 &
# tail -f output.log

class Command(BaseCommand):
    help = 'Recalculate fixed metrics for RickisMetrics from May 9 to May 23, 2025'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 5, 23))

        metrics = RickisMetrics.objects.filter(timestamp__gte=start, timestamp__lt=end).select_related("coin")
        count = 0

        for metric in metrics:
            coin = metric.coin
            timestamp = metric.timestamp
            updated = False

            try:
                if metric.stochastic_k in [None, 0] or metric.stochastic_d in [None, 0]:
                    k, d = calculate_stochastic(coin, timestamp)
                    if k is not None:
                        metric.stochastic_k = k
                        updated = True
                    if d is not None:
                        metric.stochastic_d = d
                        updated = True

                if metric.atr_1h in [None, 0]:
                    atr = calculate_atr_1h(coin, timestamp)
                    if atr is not None:
                        metric.atr_1h = atr
                        updated = True

                if metric.bollinger_upper in [None, 0] or metric.bollinger_middle in [None, 0] or metric.bollinger_lower in [None, 0]:
                    upper, middle, lower = calculate_bollinger_bands(coin, timestamp)
                    if upper is not None:
                        metric.bollinger_upper = upper
                        updated = True
                    if middle is not None:
                        metric.bollinger_middle = middle
                        updated = True
                    if lower is not None:
                        metric.bollinger_lower = lower
                        updated = True

                if metric.adx in [None, 0]:
                    adx = calculate_adx(coin, timestamp)
                    if adx is not None:
                        metric.adx = adx
                        updated = True

                if updated:
                    metric.save()
                    count += 1
                    print(f"✅ Updated {count} metrics")

            except Exception as e:
                print(f"❌ Error at {coin.symbol} {timestamp}: {e}")

        print(f"✅ Done. Updated {count} metrics")
