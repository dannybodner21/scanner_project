from django.core.management.base import BaseCommand
from django.utils.timezone import now, timedelta
from scanner.models import Metrics, BacktestResult
from scanner.utils import score_metrics
from decimal import Decimal
import os
from pathlib import Path
from django.conf import settings


MODEL_DIR = "/workspace/tmp"
MODEL_FILENAME = "ml_model.pkl"
model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)


class Command(BaseCommand):
    help = "Backtest ML model on historical Metrics data"

    def handle(self, *args, **kwargs):
        cutoff = now() - timedelta(days=7)
        metrics = Metrics.objects.filter(timestamp__gte=cutoff).order_by("coin", "timestamp")
        metrics_by_coin = {}

        for m in metrics:
            if not m.last_price:
                continue
            metrics_by_coin.setdefault(m.coin_id, []).append(m)

        total_tested = 0
        wins = 0
        losses = 0
        skipped = 0

        for coin_id, entries in metrics_by_coin.items():
            for i in range(len(entries) - 48):  # ~4 hour lookahead (5min * 48)
                current = entries[i]

                # Check all required fields are present
                if None in (
                    current.price_change_5min,
                    current.price_change_10min,
                    current.price_change_1hr,
                    current.price_change_24hr,
                    current.price_change_7d,
                    current.five_min_relative_volume,
                    current.rolling_relative_volume,
                    current.twenty_min_relative_volume,
                    current.volume_24h,
                ):
                    skipped += 1
                    continue

                metrics_dict = {
                    "price_change_5min": current.price_change_5min,
                    "price_change_10min": current.price_change_10min,
                    "price_change_1hr": current.price_change_1hr,
                    "price_change_24hr": current.price_change_24hr,
                    "price_change_7d": current.price_change_7d,
                    "five_min_relative_volume": current.five_min_relative_volume,
                    "rolling_relative_volume": current.rolling_relative_volume,
                    "twenty_min_relative_volume": current.twenty_min_relative_volume,
                    "volume_24h": current.volume_24h,
                }

                confidence = score_metrics(metrics_dict)
                if confidence < 0.7:
                    print(f"🧠 {current.coin.symbol} at {current.timestamp} → Confidence: {confidence:.4f}")
                    continue

                entry_price = Decimal(current.last_price)
                tp_price = entry_price * Decimal("1.03")  # 3% take profit
                sl_price = entry_price * Decimal("0.98")  # 2% stop loss

                future_entries = entries[i+1 : i+49]  # 4 hours of 5min intervals
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
                # If both hit or neither, don't count it

                BacktestResult.objects.create(
                    coin=current.coin,
                    timestamp=current.timestamp,
                    entry_price=entry_price,
                    exit_price=max(prices) if hit_tp else min(prices) if hit_sl else None,
                    success=bool(hit_tp and not hit_sl),
                    confidence=confidence,
                    entry_metrics=current
                )


        print("📊 BACKTEST RESULTS")
        print(f"Tested: {total_tested}")
        print(f"✅ Wins: {wins}")
        print(f"❌ Losses: {losses}")
        if total_tested:
            accuracy = (wins / total_tested) * 100
            print(f"📈 Win Rate: {accuracy:.2f}%")
        else:
            print("⚠️ No trades tested.")
