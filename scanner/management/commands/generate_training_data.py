from django.core.management.base import BaseCommand
from django.utils.timezone import now, timedelta
from scanner.models import Metrics, SuccessfulMove, BacktestResult
from decimal import Decimal

class Command(BaseCommand):
    help = "Generate training data by labeling past Metrics as successes or failures"

    def handle(self, *args, **kwargs):
        cutoff = now() - timedelta(days=7)
        metrics = Metrics.objects.filter(timestamp__gte=cutoff).order_by("coin", "timestamp")
        metrics_by_coin = {}

        # Group by coin
        for m in metrics:
            if not m.last_price:
                continue
            metrics_by_coin.setdefault(m.coin_id, []).append(m)

        total = 0
        successes = 0
        failures = 0
        skipped = 0

        for coin_id, entries in metrics_by_coin.items():
            for i in range(len(entries) - 48):  # ~4 hour lookahead
                current = entries[i]
                if None in (
                    current.price_change_5min,
                    current.price_change_10min,
                    current.price_change_1hr,
                    current.five_min_relative_volume,
                    current.rolling_relative_volume,
                    current.twenty_min_relative_volume,
                    current.price_change_24hr,
                    current.price_change_7d,
                    current.volume_24h,
                ):
                    skipped += 1
                    continue

                entry_price = Decimal(current.last_price)
                tp = entry_price * Decimal("1.03")  # 3% take profit
                sl = entry_price * Decimal("0.98")  # 2% stop loss

                future_entries = entries[i + 1 : i + 49]
                prices = [Decimal(f.last_price) for f in future_entries if f.last_price]

                if not prices:
                    continue

                hit_tp = any(p >= tp for p in prices)
                hit_sl = any(p <= sl for p in prices)

                if hit_tp and not hit_sl:
                    success = True
                    successes += 1
                elif hit_sl and not hit_tp:
                    success = False
                    failures += 1
                else:
                    continue

                BacktestResult.objects.create(
                    coin=current.coin,
                    entry_price=float(entry_price),
                    timestamp=current.timestamp,
                    success=success,
                    entry_metrics=current,
                    confidence=0.0,
                )
                total += 1

        print("🏗️  TRAINING DATA GENERATED")
        print(f"Total entries: {total}")
        print(f"✅ Successes: {successes}")
        print(f"❌ Failures: {failures}")
        print(f"⏭️ Skipped: {skipped}")
