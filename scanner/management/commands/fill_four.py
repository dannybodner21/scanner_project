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
    help = 'Recalculate key metrics for RickisMetrics entries between April 20 and May 12'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 4, 20))

        queryset = RickisMetrics.objects.filter(timestamp__gte=start, timestamp__lt=end).select_related("coin")
        batch = []

        for metric in queryset:
            try:
                coin = metric.coin
                timestamp = metric.timestamp

                macd, macd_signal = calculate_macd(coin, timestamp)
                stochastic_k, stochastic_d = calculate_stochastic(coin, timestamp)
                support, resistance = calculate_support_resistance(coin, timestamp)

                metric.rsi = calculate_rsi(coin, timestamp)
                metric.macd = macd
                metric.macd_signal = macd_signal
                metric.stochastic_k = stochastic_k
                metric.stochastic_d = stochastic_d
                metric.support_level = support
                metric.resistance_level = resistance
                metric.sma_5 = calculate_sma(coin, timestamp, window=5)
                metric.sma_20 = calculate_sma(coin, timestamp, window=20)
                metric.stddev_1h = calculate_stddev_1h(coin, timestamp)
                metric.atr_1h = calculate_atr_1h(coin, timestamp)
                metric.change_5m = calculate_price_change_five_min(coin)
                metric.change_since_high = calculate_change_since_high(float(metric.price), float(metric.high_24h))
                metric.change_since_low = calculate_change_since_low(float(metric.price), float(metric.low_24h))
                metric.obv = calculate_obv(coin)

                batch.append(metric)

                if len(batch) >= 100:
                    RickisMetrics.objects.bulk_update(batch, [
                        "rsi", "macd", "macd_signal", "stochastic_k", "stochastic_d",
                        "support_level", "resistance_level", "sma_5", "sma_20",
                        "stddev_1h", "atr_1h", "change_5m", "change_since_high",
                        "change_since_low", "obv"
                    ])
                    batch.clear()

            except Exception as e:
                print(f"❌ Error on {metric.coin.symbol} at {metric.timestamp}: {e}")

        if batch:
            RickisMetrics.objects.bulk_update(batch, [
                "rsi", "macd", "macd_signal", "stochastic_k", "stochastic_d",
                "support_level", "resistance_level", "sma_5", "sma_20",
                "stddev_1h", "atr_1h", "change_5m", "change_since_high",
                "change_since_low", "obv"
            ])

        print("✅ Recalculation complete for April 20 to May 12.")
