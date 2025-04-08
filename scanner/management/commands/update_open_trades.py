from django.core.management.base import BaseCommand
from scanner.models import FiredSignal, Metrics
from django.utils.timezone import now
from decimal import Decimal

class Command(BaseCommand):
    help = "Efficiently check open trades and close them if TP or SL hit"

    def handle(self, *args, **kwargs):
        open_signals = FiredSignal.objects.select_related("coin").filter(result="unknown")
        closed = 0

        # Create a cache for latest metrics per coin
        coin_ids = list(open_signals.values_list("coin_id", flat=True).distinct())
        latest_metrics = (
            Metrics.objects.filter(coin_id__in=coin_ids)
            .order_by("coin_id", "-timestamp")
        )

        # Map coin_id -> latest metric
        latest_by_coin = {}
        for metric in latest_metrics:
            if metric.coin_id not in latest_by_coin and metric.last_price:
                latest_by_coin[metric.coin_id] = metric

        for signal in open_signals:
            metric = latest_by_coin.get(signal.coin_id)
            if not metric or not metric.last_price:
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
            else:  # short
                tp = entry * Decimal("0.97")
                sl = entry * Decimal("1.02")
                if current_price <= tp:
                    signal.result = "success"
                elif current_price >= sl:
                    signal.result = "failure"
                else:
                    continue

            signal.exit_price = current_price
            signal.closed_at = now()
            signal.save()
            closed += 1

        print(f"✅ Closed {closed} signals")
