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

                if metric.support_level in [None, 0] or metric.resistance_level in [None, 0]:
                    support, resistance = calculate_support_resistance(coin, timestamp)
                    if support is not None:
                        metric.support_level = support
                        updated = True
                    if resistance is not None:
                        metric.resistance_level = resistance
                        updated = True

                if metric.relative_volume in [None, 0]:
                    rvol = calculate_relative_volume(coin, timestamp)
                    if rvol is not None:
                        metric.relative_volume = rvol
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

                if metric.stddev_1h in [None, 0]:
                    std = calculate_stddev_1h(coin, timestamp)
                    if std is not None:
                        metric.stddev_1h = std
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
