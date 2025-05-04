from datetime import datetime
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
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
    calculate_stddev,
    calculate_atr,
    calculate_change_since_high,
    calculate_change_since_low,
    calculate_volume_mc_ratio,
    calculate_fib_distances,
    calculate_obv,
    calculate_adx,
    calculate_bollinger_bands,
)

class Command(BaseCommand):
    help = "Backfill all calculated indicators on RickisMetrics entries"

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 5, 3))
        queryset = RickisMetrics.objects.filter(timestamp__gte=start, timestamp__lt=end).select_related('coin').order_by('coin', 'timestamp')

        total = queryset.count()
        print(f"📊 Processing {total} entries...")

        batch = []
        for i, metric in enumerate(queryset, start=1):
            try:
                if not metric.price:
                    continue

                calculate_rsi(metric)
                calculate_macd(metric)
                calculate_stochastic(metric)
                calculate_support_resistance(metric)
                calculate_price_slope_1h(metric)
                calculate_relative_volume(metric)
                calculate_sma(metric)
                calculate_ema(metric)
                calculate_stddev(metric)
                calculate_atr(metric)
                calculate_change_since_high(metric)
                calculate_change_since_low(metric)
                calculate_volume_mc_ratio(metric)
                calculate_fib_distances(metric)
                calculate_obv(metric)
                calculate_adx(metric)
                calculate_bollinger_bands(metric)

                batch.append(metric)

                if i % 100 == 0:
                    RickisMetrics.objects.bulk_update(batch, [
                        "rsi", "macd", "macd_signal", "stochastic_k", "stochastic_d",
                        "support_level", "resistance_level", "price_slope_1h", "relative_volume",
                        "sma_5", "sma_20", "ema_12", "ema_26", "stddev_1h", "atr_1h",
                        "change_since_high", "change_since_low", "volume_mc_ratio",
                        "fib_0_236", "fib_0_382", "fib_0_5", "fib_0_618", "fib_0_786",
                        "obv", "adx", "bollinger_upper", "bollinger_middle", "bollinger_lower"
                    ])
                    print(f"✅ {i}/{total} updated")
                    batch.clear()

            except Exception as e:
                print(f"❌ Error on {metric.coin.symbol} @ {metric.timestamp}: {e}")

        if batch:
            RickisMetrics.objects.bulk_update(batch, [
                "rsi", "macd", "macd_signal", "stochastic_k", "stochastic_d",
                "support_level", "resistance_level", "price_slope_1h", "relative_volume",
                "sma_5", "sma_20", "ema_12", "ema_26", "stddev_1h", "atr_1h",
                "change_since_high", "change_since_low", "volume_mc_ratio",
                "fib_0_236", "fib_0_382", "fib_0_5", "fib_0_618", "fib_0_786",
                "obv", "adx", "bollinger_upper", "bollinger_middle", "bollinger_lower"
            ])
            print(f"✅ Final batch of {len(batch)} saved.")

        print("🏁 All entries processed.")
