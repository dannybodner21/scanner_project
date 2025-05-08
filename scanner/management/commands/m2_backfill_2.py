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

class Command(BaseCommand):

    help = "Backfill all missing derived metrics in RickisMetrics"

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 4, 2))
        end = make_aware(datetime(2025, 5, 3))

        metrics = RickisMetrics.objects.filter(
            timestamp__gte=start,
            timestamp__lt=end
        ).select_related("coin").order_by("timestamp")

        total = metrics.count()
        print(f"📊 Processing {total} entries...")

        batch = []
        updated = 0

        for i, rm in enumerate(metrics, 1):
            try:
                if not rm.price or not rm.coin:
                    continue

                modified = False

                def set_if_missing(field, value):
                    nonlocal modified
                    if getattr(rm, field) is None and value is not None:
                        setattr(rm, field, value)
                        modified = True

                set_if_missing("rsi", calculate_rsi(rm.coin, rm.timestamp))
                macd, macd_signal = calculate_macd(rm.coin, rm.timestamp)
                set_if_missing("macd", macd)
                set_if_missing("macd_signal", macd_signal)

                k, d = calculate_stochastic(rm.coin, rm.timestamp)
                set_if_missing("stochastic_k", k)
                set_if_missing("stochastic_d", d)

                support, resistance = calculate_support_resistance(rm.coin, rm.timestamp)
                set_if_missing("support_level", support)
                set_if_missing("resistance_level", resistance)

                set_if_missing("price_slope_1h", calculate_price_slope_1h(rm.coin, rm.timestamp))
                set_if_missing("relative_volume", calculate_relative_volume(rm.coin, rm.timestamp))
                set_if_missing("sma_5", calculate_sma(rm.coin, rm.timestamp, 5))
                set_if_missing("sma_20", calculate_sma(rm.coin, rm.timestamp, 20))
                set_if_missing("ema_12", calculate_ema(rm.coin, rm.timestamp, 12))
                set_if_missing("ema_26", calculate_ema(rm.coin, rm.timestamp, 26))
                set_if_missing("stddev_1h", calculate_stddev_1h(rm.coin, rm.timestamp))
                set_if_missing("atr_1h", calculate_atr_1h(rm.coin, rm.timestamp))
                set_if_missing("change_since_high", calculate_change_since_high(rm.price, rm.high_24h))
                set_if_missing("change_since_low", calculate_change_since_low(rm.price, rm.low_24h))
                set_if_missing("obv", calculate_obv(rm.coin, rm.timestamp))

                fibs = calculate_fib_distances(rm.high_24h, rm.low_24h, rm.price)
                set_if_missing("fib_distance_0_236", fibs.get("fib_distance_0_236"))
                set_if_missing("fib_distance_0_382", fibs.get("fib_distance_0_382"))
                set_if_missing("fib_distance_0_5", fibs.get("fib_distance_0_5"))
                set_if_missing("fib_distance_0_618", fibs.get("fib_distance_0_618"))
                set_if_missing("fib_distance_0_786", fibs.get("fib_distance_0_786"))

                upper, middle, lower = calculate_bollinger_bands(rm.coin, rm.timestamp)
                set_if_missing("bollinger_upper", upper)
                set_if_missing("bollinger_middle", middle)
                set_if_missing("bollinger_lower", lower)

                if modified:
                    batch.append(rm)
                    updated += 1

                if len(batch) >= 100:
                    RickisMetrics.objects.bulk_update(batch, [
                        "rsi", "macd", "macd_signal", "stochastic_k", "stochastic_d",
                        "support_level", "resistance_level", "price_slope_1h", "relative_volume",
                        "sma_5", "sma_20", "ema_12", "ema_26", "stddev_1h", "atr_1h",
                        "change_since_high", "change_since_low",
                        "fib_distance_0_236", "fib_distance_0_382", "fib_distance_0_5",
                        "fib_distance_0_618", "fib_distance_0_786",
                        "obv", "bollinger_upper", "bollinger_middle", "bollinger_lower"
                    ])
                    print(f"✅ {i}/{total} entries updated")
                    batch.clear()

            except Exception as e:
                print(f"❌ Error for {rm.coin.symbol} @ {rm.timestamp}: {e}")

        if batch:
            RickisMetrics.objects.bulk_update(batch, [
                "rsi", "macd", "macd_signal", "stochastic_k", "stochastic_d",
                "support_level", "resistance_level", "price_slope_1h", "relative_volume",
                "sma_5", "sma_20", "ema_12", "ema_26", "stddev_1h", "atr_1h",
                "change_since_high", "change_since_low",
                "fib_distance_0_236", "fib_distance_0_382", "fib_distance_0_5",
                "fib_distance_0_618", "fib_distance_0_786",
                "obv", "bollinger_upper", "bollinger_middle", "bollinger_lower"
            ])
            print(f"✅ Final batch saved.")

        print(f"🏁 Done. {updated} entries updated.")
