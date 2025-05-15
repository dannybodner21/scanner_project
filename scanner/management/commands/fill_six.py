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

                if metric.atr_1h == 0 or metric.atr_1h is None:
                    atr = calculate_atr_1h(coin, timestamp)
                    print(f"in atr: {atr} - {coin.symbol} - {timestamp}")
                    if atr is not None:
                        metric.atr_1h = atr
                        updated = True
                    else:
                        print(f"atr1h returned NONE: {coin.symbol} at {timestamp}")

                if metric.change_5m == 0 or metric.change_5m is None:
                    change5m = calculate_price_change_five_min(coin, timestamp)
                    print(f"in change5m: {change5m} - {coin.symbol} - {timestamp}")
                    if change5m is not None:
                        metric.change_5m = change5m
                        updated = True
                    else:
                        print(f"change5 returned NONE: {coin.symbol} at {timestamp}")

                if updated:
                    count += 1
                    metric.save()
                    #print(f"✅ Updated {coin.symbol} at {timestamp}")
                    #print(f"updated: {count}")

            except Exception as e:
                print(f"❌ Error at {coin.symbol} {timestamp}: {e}")

        print("✅ Selective metric recalculation complete.")
