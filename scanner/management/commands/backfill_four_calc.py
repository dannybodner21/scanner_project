from datetime import datetime
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from scanner.helpers import (
    calculate_rsi, calculate_macd, calculate_price_change_five_min,
    calculate_avg_volume_1h, calculate_stochastic, calculate_support_resistance,
    calculate_relative_volume, calculate_sma, calculate_stddev_1h,
    calculate_atr_1h, calculate_obv
)

# nohup python manage.py backfill_four_calc > output.log 2>&1 &
# tail -f output.log

class Command(BaseCommand):
    help = 'Recalculate missing metrics for RickisMetrics from May 9 to May 23, 2025'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 5, 9))
        end = make_aware(datetime(2025, 5, 23))

        metrics = RickisMetrics.objects.filter(timestamp__gte=start, timestamp__lt=end).select_related("coin")
        count = 0

        for metric in metrics:
            coin = metric.coin
            timestamp = metric.timestamp
            updated = False

            try:
                if metric.change_5m in [None, 0]:
                    change_5m = calculate_price_change_five_min(coin, timestamp)
                    if change_5m is not None:
                        metric.change_5m = change_5m
                        updated = True

                if metric.avg_volume_1h in [None, 0]:
                    avg_volume = calculate_avg_volume_1h(coin, timestamp)
                    if avg_volume is not None:
                        metric.avg_volume_1h = avg_volume
                        updated = True

                if metric.rsi in [None, 0]:
                    rsi = calculate_rsi(coin, timestamp)
                    if rsi is not None:
                        metric.rsi = rsi
                        updated = True

                if metric.macd in [None, 0] or metric.macd_signal in [None, 0]:
                    macd, signal = calculate_macd(coin, timestamp)
                    if macd is not None:
                        metric.macd = macd
                        updated = True
                    if signal is not None:
                        metric.macd_signal = signal
                        updated = True

                if metric.stochastic_k in [None, 0] or metric.stochastic_d in [None, 0]:
                    k, d = calculate_stochastic(coin, timestamp)
                    if k is not None:
                        metric.stochastic_k = k
                        updated = True
                    if d is not None:
                        metric.stochastic_d = d
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
                    rel_vol = calculate_relative_volume(coin, timestamp)
                    if rel_vol is not None:
                        metric.relative_volume = rel_vol
                        updated = True

                if metric.sma_5 in [None, 0]:
                    sma5 = calculate_sma(coin, timestamp, 5)
                    if sma5 is not None:
                        metric.sma_5 = sma5
                        updated = True

                if metric.sma_20 in [None, 0]:
                    sma20 = calculate_sma(coin, timestamp, 20)
                    if sma20 is not None:
                        metric.sma_20 = sma20
                        updated = True

                if metric.stddev_1h in [None, 0]:
                    stddev = calculate_stddev_1h(coin, timestamp)
                    if stddev is not None:
                        metric.stddev_1h = stddev
                        updated = True

                if metric.atr_1h in [None, 0]:
                    atr = calculate_atr_1h(coin, timestamp)
                    if atr is not None:
                        metric.atr_1h = atr
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

        print(f"✅ Updated {count} metrics")
