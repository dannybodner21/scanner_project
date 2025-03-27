from django.core.management.base import BaseCommand
from django.utils.timezone import now, timedelta
from scanner.models import FiredSignal, ShortIntervalData

class Command(BaseCommand):
    help = "Check fired signals and update them as win/loss based on price movement"

    def handle(self, *args, **kwargs):
        cutoff = now() - timedelta(hours=4)
        signals = FiredSignal.objects.filter(result="unknown", fired_at__gte=cutoff)

        checked = 0
        wins = 0
        losses = 0

        for signal in signals:
            coin = signal.coin
            fired_at = signal.fired_at
            entry_price = float(signal.price_at_fired)

            # Fetch price data since the signal fired
            prices = ShortIntervalData.objects.filter(
                coin=coin,
                timestamp__gt=fired_at,
                timestamp__lte=fired_at + timedelta(hours=4)
            ).order_by("timestamp")

            if not prices.exists():
                continue

            high = max([float(p.price) for p in prices])
            low = min([float(p.price) for p in prices])

            tp_long = entry_price * 1.04  # +4%
            sl_long = entry_price * 0.98  # -2%

            tp_short = entry_price * 0.96  # -4%
            sl_short = entry_price * 1.02  # +2%

            result = "unknown"

            # LONG logic
            if high >= tp_long and low > sl_long:
                result = "win"
            elif low <= sl_long and high < tp_long:
                result = "loss"
            # SHORT logic
            elif low <= tp_short and high < sl_short:
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
