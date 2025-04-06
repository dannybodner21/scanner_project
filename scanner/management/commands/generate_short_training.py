from django.core.management.base import BaseCommand
from django.utils.timezone import now, timedelta
from scanner.models import Metrics, BacktestResult
from decimal import Decimal

class Command(BaseCommand):
    help = "Generate short training data (3% drop before 2% rise = success)"

    def handle(self, *args, **kwargs):
        cutoff = now() - timedelta(days=7)
        metrics = Metrics.objects.filter(timestamp__gte=cutoff).order_by("coin", "timestamp")
        metrics_by_coin = {}

        for m in metrics:
            if not m.last_price:
                continue
            metrics_by_coin.setdefault(m.coin_id, []).append(m)

        total = 0
        successes = 0
        failures = 0
        skipped = 0

        for coin_id, entries in metrics_by_coin.items():
            for i in range(len(entries) - 48):  # 4 hours forward
                current = entries[i]
                if None in (
                    current.price_change_5min,
                    current.price_change_10min,
                    current.price_change_1hr,
                    current.price_change_24hr,
                    current.price_change_7d,
                    current.five_min_relative_volume,
                    current.rolling_relative_volume,
                    current.volume_24h,
                ):
                    skipped += 1
                    continue

                entry_price = Decimal(current.last_price)
                tp = entry_price * Decimal("0.96")  # take profit: -4%
                sl = entry_price * Decimal("1.02")  # stop loss: +2%

                future_entries = entries[i + 1 : i + 49]
                prices = [Decimal(f.last_price) for f in future_entries if f.last_price]

                if not prices:
                    skipped += 1
                    continue

                hit_tp = any(p <= tp for p in prices)
                hit_sl = any(p >= sl for p in prices)

                if hit_tp and not hit_sl:
                    success = True
                    successes += 1
                elif hit_sl and not hit_tp:
                    success = False
                    failures += 1
                else:
                    continue  # discard if both hit or neither hit

                BacktestResult.objects.create(
                    coin=current.coin,
                    entry_price=float(entry_price),
                    exit_price=float(min(prices)) if success else float(max(prices)),
                    timestamp=current.timestamp,
                    success=success,
                    confidence=0.0,
                    entry_metrics=current,
                    trade_type="short",
                )
                total += 1

        print("📊 SHORT TRAINING DATA GENERATED")
        print(f"Total entries: {total}")
        print(f"✅ Successes: {successes}")
        print(f"❌ Failures: {failures}")
        print(f"⏭️ Skipped: {skipped}")
