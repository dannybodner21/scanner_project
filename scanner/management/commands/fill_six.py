from datetime import datetime
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from scanner.helpers import (
    calculate_rsi, calculate_macd, calculate_stochastic,
    calculate_support_resistance, calculate_sma, calculate_stddev_1h,
    calculate_atr_1h, calculate_price_change_five_min,
    calculate_change_since_high, calculate_change_since_low,
    calculate_obv, get_recent_volumes, get_recent_prices,
    round_to_five_minutes, calculate_avg_volume_1h, calculate_relative_volume,
    calculate_stddev_1h, calculate_fib_distances,
)

class Command(BaseCommand):
    help = 'Recalculate missing (zero) metrics for RickisMetrics between April 20 and May 12'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 4, 25))

        metrics = RickisMetrics.objects.filter(timestamp__gte=start, timestamp__lt=end).select_related("coin")
        count = 0
        for metric in metrics:
            coin = metric.coin
            timestamp = metric.timestamp
            updated = False

            try:

                if metric.stddev_1h == 0 or metric.stddev_1h is None:
                    stddev = calculate_stddev_1h(coin, timestamp)
                    print(f"in stddev: {stddev} - {coin.symbol} - {timestamp}")
                    if stddev is not None and stddev != 0:
                        metric.stddev_1h = stddev
                        updated = True
                    else:
                        print(f"stddev returned NONE or zero: {stddev} : {coin.symbol} at {timestamp}")

                if metric.rsi == 0 or metric.rsi is None:
                    rsi = calculate_rsi(coin, timestamp)
                    print(f"in rsi: {rsi} - {coin.symbol} - {timestamp}")
                    if rsi is not None:
                        metric.rsi = rsi
                        updated = True
                    else:
                        print(f"rsi returned NONE: {rsi} : {coin.symbol} at {timestamp}")

                if (metric.fib_distance_0_236 == 0 or metric.fib_distance_0_236 is None) or
                    (metric.fib_distance_0_382 == 0 or metric.fib_distance_0_382 is None) or
                    (metric.fib_distance_0_5 == 0 or metric.fib_distance_0_5 is None) or
                    (metric.fib_distance_0_618 == 0 or metric.fib_distance_0_618 is None) or
                    (metric.fib_distance_0_786 == 0 or metric.fib_distance_0_786 is None):

                    fib_distances = calculate_fib_distances(metric.high_24h, metric.low_24h, metric.price)

                    if fib_distances:
                        fib_distance_0_236 = fib_distances.get("fib_distance_0_236")
                        fib_distance_0_382 = fib_distances.get("fib_distance_0_382")
                        fib_distance_0_5 = fib_distances.get("fib_distance_0_5")
                        fib_distance_0_618 = fib_distances.get("fib_distance_0_618")
                        fib_distance_0_786 = fib_distances.get("fib_distance_0_786")

                        print(f"in fib: {fib_distance_0_236} - {coin.symbol} - {timestamp}")
                        if fib_distance_0_236 is not None:
                            metric.fib_distance_0_236 = fib_distance_0_236
                            updated = True
                        else:
                            print(f"fib returned NONE: {fib_distance_0_236} : {coin.symbol} at {timestamp}")
                        if fib_distance_0_382 is not None:
                            metric.fib_distance_0_382 = fib_distance_0_382
                            updated = True

                        if fib_distance_0_5 is not None:
                            metric.fib_distance_0_5 = fib_distance_0_5
                            updated = True
                        if fib_distance_0_618 is not None:
                            metric.fib_distance_0_618 = fib_distance_0_618
                            updated = True
                        if fib_distance_0_786 is not None:
                            metric.fib_distance_0_786 = fib_distance_0_786
                            updated = True
                            
                    else:
                        print(f"FUCK UP IN FIB! -------------------------")



                if updated:
                    count += 1
                    metric.save()
                    #print(f"✅ Updated {coin.symbol} at {timestamp}")
                    #print(f"updated: {count}")

            except Exception as e:
                print(f"❌ Error at {coin.symbol} {timestamp}: {e}")

        print("✅ Selective metric recalculation complete.")
