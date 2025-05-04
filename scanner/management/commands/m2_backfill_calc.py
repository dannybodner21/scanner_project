from datetime import datetime
from django.utils.timezone import make_aware
from scanner.models import RickisMetrics
from django.core.management.base import BaseCommand
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
    calculate_adx,
    calculate_bollinger_bands,
)

class Command(BaseCommand):
    help = "Backfill all derived metrics on RickisMetrics"

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 4, 13))

        metrics = RickisMetrics.objects.filter(
            timestamp__gte=start,
            timestamp__lt=end
        ).select_related("coin").order_by("timestamp")

        total = metrics.count()
        print(f"📊 Processing {total} entries...")

        batch = []
        for i, rm in enumerate(metrics, 1):
            try:
                if not rm.price or not rm.coin:
                    continue

                rm.rsi = calculate_rsi(rm.coin, rm.timestamp)
                rm.macd, rm.macd_signal = calculate_macd(rm.coin, rm.timestamp)
                rm.stochastic_k, rm.stochastic_d = calculate_stochastic(rm.coin, rm.timestamp)
                rm.support_level, rm.resistance_level = calculate_support_resistance(rm.coin, rm.timestamp)
                rm.price_slope_1h = calculate_price_slope_1h(rm.coin, rm.timestamp)
                rm.relative_volume = calculate_relative_volume(rm.coin, rm.timestamp)
                rm.sma_5 = calculate_sma(rm.coin, rm.timestamp, 5)
                rm.sma_20 = calculate_sma(rm.coin, rm.timestamp, 20)
                rm.ema_12 = calculate_ema(rm.coin, rm.timestamp, 12)
                rm.ema_26 = calculate_ema(rm.coin, rm.timestamp, 26)
                rm.stddev_1h = calculate_stddev_1h(rm.coin, rm.timestamp)
                rm.atr_1h = calculate_atr_1h(rm.coin, rm.timestamp)
                rm.change_since_high = calculate_change_since_high(rm.price, rm.high_24h)
                rm.change_since_low = calculate_change_since_low(rm.price, rm.low_24h)
                fibs = calculate_fib_distances(rm.high_24h, rm.low_24h, rm.price)
                rm.fib_distance_0_236 = fibs.get("fib_0_236")
                rm.fib_distance_0_382 = fibs.get("fib_0_382")
                rm.fib_distance_0_5 = fibs.get("fib_0_5")
                rm.fib_distance_0_618 = fibs.get("fib_0_618")
                rm.fib_distance_0_786 = fibs.get("fib_0_786")
                rm.obv = calculate_obv(rm.coin, rm.timestamp)
                rm.adx = calculate_adx(rm.coin, rm.timestamp)
                rm.bollinger_upper, rm.bollinger_middle, rm.bollinger_lower = calculate_bollinger_bands(rm.coin, rm.timestamp)
                batch.append(rm)

                print("working")

                if len(batch) >= 100:
                    RickisMetrics.objects.bulk_update(batch, [
                        "rsi", "macd", "macd_signal", "stochastic_k", "stochastic_d",
                        "support_level", "resistance_level", "price_slope_1h", "relative_volume",
                        "sma_5", "sma_20", "ema_12", "ema_26", "stddev_1h", "atr_1h",
                        "change_since_high", "change_since_low",
                        "fib_0_236", "fib_0_382", "fib_0_5", "fib_0_618", "fib_0_786",
                        "obv", "adx", "bollinger_upper", "bollinger_middle", "bollinger_lower"
                    ])
                    print(f"✅ {i}/{total} updated")
                    batch.clear()

            except Exception as e:
                print(f"❌ Error for {rm.coin.symbol} @ {rm.timestamp}: {e}")

        if batch:
            RickisMetrics.objects.bulk_update(batch, [
                "rsi", "macd", "macd_signal", "stochastic_k", "stochastic_d",
                "support_level", "resistance_level", "price_slope_1h", "relative_volume",
                "sma_5", "sma_20", "ema_12", "ema_26", "stddev_1h", "atr_1h",
                "change_since_high", "change_since_low",
                "fib_0_236", "fib_0_382", "fib_0_5", "fib_0_618", "fib_0_786",
                "obv", "adx", "bollinger_upper", "bollinger_middle", "bollinger_lower"
            ])
            print("✅ Final batch saved.")

        print("🏁 All metrics processed.")
