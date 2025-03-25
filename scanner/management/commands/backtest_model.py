from django.core.management.base import BaseCommand
from django.utils.timezone import now, timedelta
from scanner.models import Metrics, Coin
from scanner.utils import score_metrics
from decimal import Decimal

class Command(BaseCommand):
    help = "Backtest ML model on historical Metrics data"

    def handle(self, *args, **kwargs):
        cutoff = now() - timedelta(days=7)
        metrics = Metrics.objects.filter(timestamp__gte=cutoff).order_by("coin", "timestamp")
        metrics_by_coin = {}

        # Organize metrics per coin
        for m in metrics:
            if not m.last_price:
                continue
            metrics_by_coin.setdefault(m.coin_id, []).append(m)

        total_tested = 0
        wins = 0
        losses = 0
        skipped = 0

        for coin_id, entries in metrics_by_coin.items():
            for i in range(len(entries) - 12):  # leave room for 1hr lookahead
                current = entries[i]
                if None in (current.price_change_5min, current.five_min_relative_volume, current.price_change_1hr, current.market_cap):
                    skipped += 1
                    continue

                metrics_dict = {
                    "price_change_5min": current.price_change_5min,
                    "five_min_relative_volume": current.five_min_relative_volume,
                    "price_change_1hr": current.price_change_1hr,
                    "market_cap": float(current.market_cap)
                }

                confidence = score_metrics(metrics_dict)
                if confidence < 0.8:
                    continue  # Skip low-confidence signals

                entry_price = Decimal(current.last_price)
                tp_price = entry_price * Decimal("1.05")
                sl_price = entry_price * Decimal("0.98")

                future_entries = entries[i+1 : i+13]  # 1 hour lookahead
                prices = [Decimal(f.last_price) for f in future_entries if f.last_price]

                if not prices:
                    skipped += 1
                    continue

                hit_tp = any(p >= tp_price for p in prices)
                hit_sl = any(p <= sl_price for p in prices)

                total_tested += 1

                if hit_tp and not hit_sl:
                    wins += 1
                elif hit_sl and not hit_tp:
                    losses += 1
                else:
                    # no resolution — ignore
                    pass

        print("📊 BACKTEST RESULTS")
        print(f"Tested: {total_tested}")
        print(f"✅ Wins: {wins}")
        print(f"❌ Losses: {losses}")
        if total_tested:
            accuracy = (wins / total_tested) * 100
            print(f"📈 Win Rate: {accuracy:.2f}%")
        else:
            print("⚠️ No trades tested.")
