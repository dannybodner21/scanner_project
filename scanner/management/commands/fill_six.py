from datetime import datetime
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from scanner.helpers import (
    calculate_rsi, calculate_macd, calculate_stochastic,
    calculate_support_resistance, calculate_sma, calculate_stddev_1h,
    calculate_atr_1h, calculate_price_change_five_min,
    calculate_change_since_high, calculate_change_since_low,
    calculate_obv
)

class Command(BaseCommand):
    help = 'Recalculate missing (zero) metrics for RickisMetrics between April 20 and May 12'

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

                if metric.stochastic_k == 0 or metric.stochastic_d == 0 or metric.stochastic_k is None or metric.stochastic_d is None:
                    k, d = calculate_stochastic(coin, timestamp)
                    print(f"in stochastic: {k} {d} - {coin.symbol} - {timestamp}")
                    if k is not None and d is not None:
                        metric.stochastic_k = k
                        metric.stochastic_d = d
                        updated = True
                    else:
                        print(f"stochastic returned NONE: {coin.symbol} at {timestamp}")

                if updated:
                    count += 1
                    metric.save()
                    #print(f"✅ Updated {coin.symbol} at {timestamp}")
                    #print(f"updated: {count}")

            except Exception as e:
                print(f"❌ Error at {coin.symbol} {timestamp}: {e}")

        print("✅ Selective metric recalculation complete.")
