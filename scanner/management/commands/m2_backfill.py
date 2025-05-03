from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from scanner.helpers import (
    calculate_price_change_five_min,
    calculate_rsi,
    calculate_macd,
    calculate_macd_signal,
    calculate_stochastic_kd,
    calculate_relative_volume,
    calculate_price_slope,
    calculate_ema,
    calculate_sma,
    calculate_stddev_1h,
    calculate_atr_1h,
    calculate_change_since_high,
    calculate_change_since_low,
    calculate_volume_to_market_cap,
)
from django.utils.timezone import make_aware
from datetime import datetime

class Command(BaseCommand):
    help = "Backfill missing metric fields in RickisMetrics from March 22 to May 2"

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 5, 3))

        qs = RickisMetrics.objects.filter(timestamp__gte=start, timestamp__lt=end)
        total = qs.count()
        print(f"🔁 Checking {total} entries...")

        updated = 0
        for i, entry in enumerate(qs.iterator()):
            dirty = False  # Flag to avoid unnecessary saves

            if entry.change_5m is None:
                entry.change_5m = calculate_price_change_five_min(entry.coin, entry.timestamp)
                dirty = True

            if entry.rsi is None:
                entry.rsi = calculate_rsi(entry.coin, entry.timestamp)
                dirty = True

            if entry.macd is None or entry.macd_signal is None:
                macd, signal = calculate_macd(entry.coin, entry.timestamp)
                entry.macd = macd
                entry.macd_signal = signal
                dirty = True

            if entry.stochastic_k is None or entry.stochastic_d is None:
                k, d = calculate_stochastic_kd(entry.coin, entry.timestamp)
                entry.stochastic_k = k
                entry.stochastic_d = d
                dirty = True

            if entry.relative_volume is None:
                entry.relative_volume = calculate_relative_volume(entry.coin, entry.timestamp)
                dirty = True

            if entry.price_slope_1h is None:
                entry.price_slope_1h = calculate_price_slope(entry.coin, entry.timestamp)
                dirty = True

            if entry.ema_12 is None:
                entry.ema_12 = calculate_ema(entry.coin, entry.timestamp, period=12)
                dirty = True

            if entry.ema_26 is None:
                entry.ema_26 = calculate_ema(entry.coin, entry.timestamp, period=26)
                dirty = True

            if entry.sma_5 is None:
                entry.sma_5 = calculate_sma(entry.coin, entry.timestamp, period=5)
                dirty = True

            if entry.sma_20 is None:
                entry.sma_20 = calculate_sma(entry.coin, entry.timestamp, period=20)
                dirty = True

            if entry.stddev_1h is None:
                entry.stddev_1h = calculate_stddev_1h(entry.coin, entry.timestamp)
                dirty = True

            if entry.atr_1h is None:
                entry.atr_1h = calculate_atr_1h(entry.coin, entry.timestamp)
                dirty = True

            # Optional additional metrics
            if hasattr(entry, "change_since_high") and entry.change_since_high is None:
                entry.change_since_high = calculate_change_since_high(entry.price, entry.high_24h)
                dirty = True

            if hasattr(entry, "change_since_low") and entry.change_since_low is None:
                entry.change_since_low = calculate_change_since_low(entry.price, entry.low_24h)
                dirty = True

            if hasattr(entry, "volume_mc_ratio") and entry.volume_mc_ratio is None:
                # You'll need to get market cap from ShortIntervalData or Coin model
                market_cap = getattr(entry.coin, "market_cap", None)
                entry.volume_mc_ratio = calculate_volume_to_market_cap(entry.volume, market_cap)
                dirty = True

            if dirty:
                entry.save()
                updated += 1

            if i % 500 == 0:
                print(f"{i}/{total} checked — {updated} updated")

        print(f"\n✅ Backfill complete. {updated} entries updated.")
