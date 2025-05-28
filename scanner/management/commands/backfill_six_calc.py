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
        end = make_aware(datetime(2025, 4, 1))

        metrics = RickisMetrics.objects.filter(timestamp__gte=start, timestamp__lt=end).select_related("coin")
        count = 0

        for metric in metrics:
            coin = metric.coin
            timestamp = metric.timestamp
            updated = False

            try:


                if metric.support_level in [None, 0] or metric.resistance_level in [None, 0]:
                    support, resistance = calculate_support_resistance(coin, timestamp)
                    if support is not None:
                        metric.support_level = support
                        updated = True
                    if resistance is not None:
                        metric.resistance_level = resistance
                        updated = True



                if updated:
                    metric.save()
                    count += 1
                    print(f"✅ Updated {count} metrics")

            except Exception as e:
                print(f"❌ Error at {coin.symbol} {timestamp}: {e}")

        print(f"✅ Done. Updated {count} metrics")
