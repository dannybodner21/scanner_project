from django.core.management.base import BaseCommand
from scanner.models import FiredSignal, Metrics
from django.utils.timezone import now
from decimal import Decimal

class Command(BaseCommand):
    help = "Check open trades and close them if TP or SL hit"

    def handle(self, *args, **kwargs):
        open_signals = FiredSignal.objects.filter(result="unknown")
        closed = 0

        for signal in open_signals:
            recent_metric = Metrics.objects.filter(
                coin=signal.coin,
                timestamp__gte=signal.fired_at
            ).order_by("-timestamp").first()

            if not recent_metric or not recent_metric.last_price:
                continue

            current_price = Decimal(recent_metric.last_price)
            entry = Decimal(signal.price_at_fired)

            tp_price = entry * Decimal("1.03")  # +3%
            sl_price = entry * Decimal("0.98")  # -2%

            if signal.signal_type == "long":
                if current_price >= tp_price:
                    signal.result = "success"
                    signal.exit_price = current_price
                    signal.closed_at = now()
                    signal.save()
                    closed += 1
                elif current_price <= sl_price:
                    signal.result = "failure"
                    signal.exit_price = current_price
                    signal.closed_at = now()
                    signal.save()
                    closed += 1

            elif signal.signal_type == "short":
                if current_price <= entry * Decimal("0.97"):  # -3%
                    signal.result = "success"
                    signal.exit_price = current_price
                    signal.closed_at = now()
                    signal.save()
                    closed += 1
                elif current_price >= entry * Decimal("1.02"):  # +2%
                    signal.result = "failure"
                    signal.exit_price = current_price
                    signal.closed_at = now()
                    signal.save()
                    closed += 1

        print(f"✅ Closed {closed} signals")
