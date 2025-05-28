from datetime import datetime
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from scanner.helpers import (
    calculate_stochastic, calculate_atr_1h, calculate_bollinger_bands,
    calculate_adx, calculate_support_resistance, calculate_relative_volume,
    calculate_sma, calculate_stddev_1h, calculate_obv
)

# nohup python manage.py backfill_six_calc > output.log 2>&1 &
# tail -f output.log

# ps aux | grep backfill_six_calc

class Command(BaseCommand):
    help = 'Recalculate fixed metrics for RickisMetrics from March 23 to April 23, 2025'

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



                if metric.atr_1h in [None, 0]:
                    atr = calculate_atr_1h(coin, timestamp)
                    if atr is not None:
                        metric.atr_1h = atr
                        updated = True

                if metric.sma_5 in [None, 0]:
                    sma_5 = calculate_sma(coin, timestamp, window=5)
                    if sma_5 is not None:
                        metric.sma_5 = sma_5
                        updated = True

                if metric.sma_20 in [None, 0]:
                    sma_20 = calculate_sma(coin, timestamp, window=20)
                    if sma_20 is not None:
                        metric.sma_20 = sma_20
                        updated = True

                if metric.obv in [None, 0]:
                    obv = calculate_obv(coin, timestamp)
                    if obv is not None:
                        metric.obv = obv
                        updated = True

                if updated:
                    metric.save()
                    count += 1
                    print(f"✅ Updated {count} metrics")

            except Exception as e:
                print(f"❌ Error at {coin.symbol} {timestamp}: {e}")

        print(f"✅ Done. Updated {count} metrics")
