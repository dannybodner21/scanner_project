from datetime import datetime, timedelta
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from scanner.helpers import (
    calculate_price_change_five_min,
    calculate_rsi,
    calculate_macd,
    calculate_stochastic,
    calculate_support_resistance,
    calculate_relative_volume,
    calculate_sma,
    calculate_stddev_1h,
    calculate_atr_1h,
    calculate_obv,
)

class Command(BaseCommand):
    help = "Backfill missing indicators (momentum, trend, volume) from March 22 to May 2, 2025."

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 4, 23))
        end = make_aware(datetime(2025, 5, 2)) + timedelta(days=1)

        qs = (
            RickisMetrics.objects
            .filter(timestamp__gte=start, timestamp__lt=end)
            .select_related("coin")
            .order_by("timestamp")
        )

        total = qs.count()
        updated = 0
        batch = []

        print(f"🔍 Checking {total} entries from {start.date()} to {end.date()}...")

        for rm in qs.iterator():
            modified = False

            if rm.change_5m is None:
                rm.change_5m = calculate_price_change_five_min(rm.coin, rm.timestamp)
                modified = True

            if rm.rsi is None:
                rm.rsi = calculate_rsi(rm.coin, rm.timestamp)
                modified = True

            if rm.macd is None or rm.macd_signal is None:
                macd, signal = calculate_macd(rm.coin, rm.timestamp)
                rm.macd = macd
                rm.macd_signal = signal
                modified = True

            if rm.stochastic_k is None or rm.stochastic_d is None:
                k, d = calculate_stochastic(rm.coin, rm.timestamp)
                rm.stochastic_k = k
                rm.stochastic_d = d
                modified = True

            if rm.support_level is None or rm.resistance_level is None:
                support, resistance = calculate_support_resistance(rm.coin, rm.timestamp)
                rm.support_level = support
                rm.resistance_level = resistance
                modified = True

            if rm.relative_volume is None:
                rm.relative_volume = calculate_relative_volume(rm.coin, rm.timestamp)
                modified = True

            if rm.sma_5 is None:
                rm.sma_5 = calculate_sma(rm.coin, rm.timestamp, 5)
                modified = True

            if rm.sma_20 is None:
                rm.sma_20 = calculate_sma(rm.coin, rm.timestamp, 20)
                modified = True

            if rm.stddev_1h is None:
                rm.stddev_1h = calculate_stddev_1h(rm.coin, rm.timestamp)
                modified = True

            if rm.atr_1h is None:
                rm.atr_1h = calculate_atr_1h(rm.coin, rm.timestamp)
                modified = True

            if rm.obv is None:
                rm.obv = calculate_obv(rm.coin, rm.timestamp)
                modified = True

            if modified:
                batch.append(rm)
                updated += 1

            if len(batch) >= 500:
                RickisMetrics.objects.bulk_update(batch, [
                    'change_5m', 'rsi', 'macd', 'macd_signal',
                    'stochastic_k', 'stochastic_d',
                    'support_level', 'resistance_level', 'relative_volume',
                    'sma_5', 'sma_20', 'stddev_1h', 'atr_1h', 'obv'
                ])
                print(f"✅ {updated} updated so far at {rm.timestamp}")
                batch.clear()

        if batch:
            RickisMetrics.objects.bulk_update(batch, [
                'change_5m', 'rsi', 'macd', 'macd_signal',
                'stochastic_k', 'stochastic_d',
                'support_level', 'resistance_level', 'relative_volume',
                'sma_5', 'sma_20', 'stddev_1h', 'atr_1h', 'obv'
            ])
            print("✅ Final batch saved.")

        print(f"🎯 Done. {updated} entries updated.")
