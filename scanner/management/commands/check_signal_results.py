from django.core.management.base import BaseCommand
from scanner.models import FiredSignal, ShortIntervalData
from django.utils.timezone import now, timedelta
from decimal import Decimal

class Command(BaseCommand):
    help = "Evaluate all unresolved or unverified FiredSignals and assign win/loss based on 3% TP/SL"

    def handle(self, *args, **kwargs):
        signals = FiredSignal.objects.all().order_by("-fired_at")
        print(f"🧠 Total FiredSignals: {signals.count()}")

        checked = 0
        wins = 0
        losses = 0
        skipped = 0

        for signal in signals:
            if not signal.price_at_fired:
                print(f"⚠️ Skipping {signal.coin.symbol} — No entry price")
                continue

            coin = signal.coin
            fired_at = signal.fired_at
            entry_price = float(signal.price_at_fired)

            prices = ShortIntervalData.objects.filter(
                coin=coin,
                timestamp__gt=fired_at,
                timestamp__lte=fired_at + timedelta(hours=4)
            ).order_by("timestamp")

            if not prices.exists():
                print(f"⚠️ No price data for {signal.coin.symbol} after {fired_at}")
                skipped += 1
                continue

            high = max([float(p.price) for p in prices])
            low = min([float(p.price) for p in prices])

            # 3% thresholds
            tp_long = entry_price * 1.04
            sl_long = entry_price * 0.98
            tp_short = entry_price * 0.96
            sl_short = entry_price * 1.02

            result = "unknown"

            # Try both logics (since signal_type isn't stored yet)
            if signal.signal_type == "long":
                if high >= tp_long and low > sl_long:
                    result = "win"
                elif low <= sl_long and high < tp_long:
                    result = "loss"
            elif signal.signal_type == "short":
                if low <= tp_short and high < sl_short:
                    result = "win"
                elif high >= sl_short and low > tp_short:
                    result = "loss"


            if result != "unknown":
                signal.result = result
                signal.checked_at = now()
                signal.save()

                if result == "win":
                    wins += 1
                else:
                    losses += 1

                checked += 1

        print(f"✅ Checked {checked} signals")
        print(f"🏆 Wins: {wins}")
        print(f"💀 Losses: {losses}")
        print(f"⏭️ Skipped (no data): {skipped}")
