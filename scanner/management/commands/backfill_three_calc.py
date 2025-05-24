from datetime import datetime
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from scanner.helpers import (
    calculate_rsi, calculate_macd, calculate_price_change_five_min,
    calculate_avg_volume_1h
)

class Command(BaseCommand):
    help = 'Recalculate missing core metrics: change_5m, avg_volume_1h, rsi, macd, macd_signal for RickisMetrics from May 9 to May 23, 2025'

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

                if updated:
                    metric.save()
                    count += 1

            except Exception as e:
                print(f"❌ Error at {coin.symbol} {timestamp}: {e}")

        print(f"✅ Updated {count} metrics")
