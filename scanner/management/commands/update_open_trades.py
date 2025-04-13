from django.core.management.base import BaseCommand
from scanner.models import FiredSignal, Metrics
from django.utils.timezone import now
from decimal import Decimal

class Command(BaseCommand):
    help = "Efficiently check open trades and close them if TP or SL hit"

    def handle(self, *args, **kwargs):
        open_signals = FiredSignal.objects.select_related("coin").filter(result="unknown")
        if not open_signals.exists():
            print("✅ No open signals to check.")
            return

        coin_ids = list(open_signals.values_list("coin_id", flat=True).distinct())

        # Get most recent metric per coin
        latest_by_coin = {}
        for coin_id in coin_ids:
            metric = (
                Metrics.objects.filter(coin_id=coin_id, last_price__isnull=False)
                .order_by("-timestamp")
                .first()
            )
            if metric:
                latest_by_coin[coin_id] = metric

        closed = 0
        for signal in open_signals:
            metric = latest_by_coin.get(signal.coin_id)
            if not metric:
                continue

            current_price = Decimal(metric.last_price)
            entry = Decimal(signal.price_at_fired)

            if signal.signal_type == "long":
                tp = entry * Decimal("1.03")
                sl = entry * Decimal("0.98")
                if current_price >= tp:
                    signal.result = "success"
                elif current_price <= sl:
                    signal.result = "failure"
                else:
                    continue
            elif signal.signal_type == "short":
                tp = entry * Decimal("0.97")
                sl = entry * Decimal("1.02")
                if current_price <= tp:
                    signal.result = "success"
                elif current_price >= sl:
                    signal.result = "failure"
                else:
                    continue
            else:
                continue

            signal.exit_price = current_price
            signal.closed_at = metric.timestamp  # ← accurate time
            signal.save()
            closed += 1
            print(f"✅ Closed {signal.coin.symbol} at {current_price:.4f} ({signal.result})")

        print(f"✅ Closed {closed} signals")
