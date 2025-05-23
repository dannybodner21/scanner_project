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
                if metric.rsi == 0 or metric.rsi is None:
                    rsi = calculate_rsi(coin, timestamp)
                    print(f"in rsi: {rsi} - {coin.symbol} - {timestamp}")
                    if rsi is not None:
                        metric.rsi = rsi
                        updated = True
                    else:
                        print(f"rsi returned NONE: {coin.symbol} at {timestamp}")

                if metric.macd == 0 or metric.macd_signal == 0 or metric.macd is None or metric.macd_signal is None:
                    macd, signal = calculate_macd(coin, timestamp)
                    print(f"in macd: {macd} {signal} - {coin.symbol} - {timestamp}")
                    if macd is not None and signal is not None:
                        metric.macd = macd
                        metric.macd_signal = signal
                        updated = True
                    else:
                        print(f"macd returned NONE: {coin.symbol} at {timestamp}")

                if metric.stochastic_k == 0 or metric.stochastic_d == 0 or metric.stochastic_k is None or metric.stochastic_d is None:
                    k, d = calculate_stochastic(coin, timestamp)
                    print(f"in stochastic: {k} {d} - {coin.symbol} - {timestamp}")
                    if k is not None and d is not None:
                        metric.stochastic_k = k
                        metric.stochastic_d = d
                        updated = True
                    else:
                        print(f"stochastic returned NONE: {coin.symbol} at {timestamp}")

                if metric.support_level == 0 or metric.resistance_level == 0 or metric.support_level is None or metric.resistance_level is None:
                    support, resistance = calculate_support_resistance(coin, timestamp)
                    print(f"in support/resistance: {support} {resistance} - {coin.symbol} - {timestamp}")
                    if support is not None and resistance is not None:
                        metric.support_level = support
                        metric.resistance_level = resistance
                        updated = True
                    else:
                        print(f"support returned NONE: {coin.symbol} at {timestamp}")

                if metric.sma_5 == 0 or metric.sma_5 is None:
                    sma5 = calculate_sma(coin, timestamp, 5)
                    print(f"in sma5: {sma5} - {coin.symbol} - {timestamp}")
                    if sma5 is not None:
                        metric.sma_5 = sma5
                        updated = True
                    else:
                        print(f"sma5 returned NONE: {coin.symbol} at {timestamp}")

                if metric.sma_20 == 0 or metric.sma_20 is None:
                    sma20 = calculate_sma(coin, timestamp, 20)
                    print(f"in sma20: {sma20} - {coin.symbol} - {timestamp}")
                    if sma20 is not None:
                        metric.sma_20 = sma20
                        updated = True
                    else:
                        print(f"sma20 returned NONE: {coin.symbol} at {timestamp}")

                if metric.stddev_1h == 0 or metric.stddev_1h is None:
                    stddev = calculate_stddev_1h(coin, timestamp)
                    print(f"in stddev: {stddev} - {coin.symbol} - {timestamp}")
                    if stddev is not None:
                        metric.stddev_1h = stddev
                        updated = True
                    else:
                        print(f"stddev returned NONE: {coin.symbol} at {timestamp}")

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

                if metric.change_since_high == 0 or metric.change_since_high is None:
                    high_diff = calculate_change_since_high(float(metric.price), float(metric.high_24h))
                    print(f"in high_diff: {high_diff} - {coin.symbol} - {timestamp}")
                    if high_diff is not None:
                        metric.change_since_high = high_diff
                        updated = True
                    else:
                        print(f"change high returned NONE: {coin.symbol} at {timestamp}")

                if metric.change_since_low == 0 or metric.change_since_low is None:
                    low_diff = calculate_change_since_low(float(metric.price), float(metric.low_24h))
                    print(f"in low_diff: {low_diff} - {coin.symbol} - {timestamp}")
                    if low_diff is not None:
                        metric.change_since_low = low_diff
                        updated = True
                    else:
                        print(f"change low returned NONE: {coin.symbol} at {timestamp}")

                if metric.obv == 0 or metric.obv is None:
                    obv = calculate_obv(coin, timestamp)
                    print(f"in obv: {obv} - {coin.symbol} - {timestamp}")
                    if obv is not None:
                        metric.obv = obv
                        updated = True
                    else:
                        print(f"obv returned NONE: {coin.symbol} at {timestamp}")

                if updated:
                    count += 1
                    metric.save()
                    #print(f"✅ Updated {coin.symbol} at {timestamp}")
                    #print(f"updated: {count}")

            except Exception as e:
                print(f"❌ Error at {coin.symbol} {timestamp}: {e}")

        print("✅ Selective metric recalculation complete.")
