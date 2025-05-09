from datetime import datetime
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from scanner.helpers import (
    calculate_rsi,
    calculate_macd,
    calculate_stochastic,
    calculate_support_resistance,
    calculate_price_slope_1h,
    calculate_relative_volume,
    calculate_sma,
    calculate_ema,
    calculate_stddev_1h,
    calculate_atr_1h,
    calculate_change_since_high,
    calculate_change_since_low,
    calculate_fib_distances,
    calculate_obv,
    calculate_bollinger_bands,
)

    # nohup python manage.py m2_trade_outcomes > backfill.log 2>&1 &
    # tail -f backfill.log


class Command(BaseCommand):

    help = "Backfill calculated metrics"

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 5, 3))

        # get all the metrics
        metrics = Metrics.objects.filter(
            timestamp__gte=start,
            timestamp__lt=end
        ).select_related("coin").order_by("timestamp")
        total = metrics.count()
        batch = []
        updated = 0
        # loop through metrics
        for metric in enumerate(metrics, 1):
            try:
                if not metric.price or not metric.coin:
                    continue
                modified = False
                # if the value is missing -> do the calculations
                def set_if_missing(field, value):
                    nonlocal modified
                    if getattr(metric, field) in [None, 0]:
                        setattr(metric, field, value)
                        modified = True

                set_if_missing("rsi", calculate_rsi(metric.coin, metric.timestamp))
                macd, macd_signal = calculate_macd(metric.coin, metric.timestamp)
                set_if_missing("macd", macd)
                set_if_missing("macd_signal", macd_signal)
                k, d = calculate_stochastic(metric.coin, metric.timestamp)
                set_if_missing("stochastic_k", k)
                set_if_missing("stochastic_d", d)
                support, resistance = calculate_support_resistance(metric.coin, metric.timestamp)
                set_if_missing("support_level", support)
                set_if_missing("resistance_level", resistance)
                set_if_missing("relative_volume", calculate_relative_volume(metric.coin, metric.timestamp))
                set_if_missing("sma_5", calculate_sma(metric.coin, metric.timestamp, 5))
                set_if_missing("sma_20", calculate_sma(metric.coin, metric.timestamp, 20))
                set_if_missing("stddev_1h", calculate_stddev_1h(metric.coin, metric.timestamp))
                set_if_missing("atr_1h", calculate_atr_1h(metric.coin, metric.timestamp))
                set_if_missing("change_since_high", calculate_change_since_high(metric.price, metric.high_24h))
                set_if_missing("change_since_low", calculate_change_since_low(metric.price, metric.low_24h))
                set_if_missing("obv", calculate_obv(metric.coin, metric.timestamp))
                fibs = calculate_fib_distances(metric.high_24h, metric.low_24h, metric.price)
                set_if_missing("fib_distance_0_236", fibs.get("fib_0_236"))
                set_if_missing("fib_distance_0_382", fibs.get("fib_0_382"))
                set_if_missing("fib_distance_0_5",   fibs.get("fib_0_5"))
                set_if_missing("fib_distance_0_618", fibs.get("fib_0_618"))
                set_if_missing("fib_distance_0_786", fibs.get("fib_0_786"))

                if modified:
                    batch.append(metric)
                    updated += 1

                # update
                if len(batch) >= 100:
                    Metrics.objects.bulk_update(batch, [
                        "rsi", "macd", "macd_signal", "stochastic_k", "stochastic_d",
                        "support_level", "resistance_level", "relative_volume",
                        "sma_5", "sma_20", "stddev_1h", "atr_1h",
                        "change_since_high", "change_since_low",
                        "fib_distance_0_236", "fib_distance_0_382", "fib_distance_0_5",
                        "fib_distance_0_618", "fib_distance_0_786",
                        "obv",
                    ])
                    batch.clear()

            except Exception as e:
                print(f"error: {e}")
        # update
        if batch:
            Metrics.objects.bulk_update(batch, [
                "rsi", "macd", "macd_signal", "stochastic_k", "stochastic_d",
                "support_level", "resistance_level", "relative_volume",
                "sma_5", "sma_20", "stddev_1h", "atr_1h",
                "change_since_high", "change_since_low",
                "fib_distance_0_236", "fib_distance_0_382", "fib_distance_0_5",
                "fib_distance_0_618", "fib_distance_0_786",
                "obv",
            ])
