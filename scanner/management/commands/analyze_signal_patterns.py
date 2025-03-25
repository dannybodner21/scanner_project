from django.core.management.base import BaseCommand
from scanner.models import FiredSignal
from collections import defaultdict
from decimal import Decimal
import math

class Command(BaseCommand):
    help = "Analyze signal success rates by metric patterns"

    def handle(self, *args, **kwargs):
        signals = FiredSignal.objects.filter(result__in=["win", "loss"])
        bins = defaultdict(lambda: {"wins": 0, "losses": 0, "total": 0})

        for sig in signals:
            m = sig.metrics
            if not m: continue

            required_keys = ["price_change_5min", "five_min_relative_volume", "price_change_1hr", "market_cap"]
            if not all(k in m and m[k] is not None for k in required_keys):
                print(f"❌ Skipping {sig.coin.symbol} — missing one or more required metrics: {m}")
                continue


            # Round metrics into buckets
            bucket = (
                f"5mΔ:{math.floor(m['price_change_5min'])}%",
                f"vol:{round(m['five_min_relative_volume'], 1)}x",
                f"1hΔ:{math.floor(m['price_change_1hr'])}%",
                f"mc:{int(Decimal(m['market_cap']) / 1_000_000)}M"
            )

            key = " | ".join(bucket)

            if sig.result == "win":
                bins[key]["wins"] += 1
            else:
                bins[key]["losses"] += 1

            bins[key]["total"] += 1

        sorted_bins = sorted(bins.items(), key=lambda kv: kv[1]["wins"], reverse=True)

        print("\nTop performing metric buckets:\n")
        for key, stats in sorted_bins:
            total = stats["total"]
            win_rate = (stats["wins"] / total) * 100 if total > 0 else 0
            if total >= 5:  # Only show combos with at least 5 signals
                print(f"{key}  →  {stats['wins']}W / {stats['losses']}L  →  {win_rate:.1f}% win rate")
