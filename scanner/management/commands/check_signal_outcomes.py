from django.core.management.base import BaseCommand
from django.utils.timezone import now, timedelta
from scanner.models import FiredSignal, ShortIntervalData

class Command(BaseCommand):
    help = "Checks fired signals and updates their win/loss result"

    def handle(self, *args, **kwargs):
        cutoff = now() - timedelta(hours=1)
        signals = FiredSignal.objects.filter(result="unknown", fired_at__lte=cutoff)

        for sig in signals:
            data = ShortIntervalData.objects.filter(
                coin=sig.coin,
                timestamp__gt=sig.fired_at,
                timestamp__lte=sig.fired_at + timedelta(hours=1)
            ).order_by("timestamp")

            prices = [d.price for d in data if d.price]

            if not prices:
                print(f"No data for {sig.coin.symbol} after {sig.fired_at}")
                continue

            max_price = max(prices)
            min_price = min(prices)

            entry = float(sig.price_at_fired)
            take_profit = entry * (1 + sig.take_profit_pct / 100)
            stop_loss = entry * (1 - sig.stop_loss_pct / 100)

            if max_price >= take_profit:
                sig.result = "win"
            elif min_price <= stop_loss:
                sig.result = "loss"
            else:
                sig.result = "unknown"

            sig.checked_at = now()
            sig.save()
            print(f"{sig.coin.symbol}: {sig.result}")
