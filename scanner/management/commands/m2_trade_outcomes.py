from datetime import datetime, timedelta
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics

class Command(BaseCommand):
    help = "Label Metrics with long_result and short_result."

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 5, 3))

        entries = RickisMetrics.objects.filter(
            timestamp__gte=start, timestamp__lt=end
        ).select_related("coin").order_by("timestamp")
        total = entries.count()
        batch = []

        # get entry price / SL = 2% / TP = 6%
        for index, entry in enumerate(entries, 1):
            entry_price = float(entry.price)
            tp_long = entry_price * 1.06
            sl_long = entry_price * 0.98
            tp_short = entry_price * 0.94
            sl_short = entry_price * 1.02

            # look at future prices for 24 hours
            future_metrics = RickisMetrics.objects.filter(
                coin=entry.coin,
                timestamp__gt=entry.timestamp,
                timestamp__lte=entry.timestamp + timedelta(hours=24)
            ).only("timestamp", "price").order_by("timestamp")

            long_result = None
            short_result = None
            # loop through future metrics and check prices
            # against the TP and SL for long and short trades
            for metric in future_metrics:
                try:
                    price = float(metric.price)

                    if long_result is None:
                        if price >= tp_long:
                            long_result = True
                        elif price <= sl_long:
                            long_result = False

                    if short_result is None:
                        if price <= tp_short:
                            short_result = True
                        elif price >= sl_short:
                            short_result = False

                    if long_result is not None and short_result is not None:
                        break

                except:
                    continue

            # if trade is open after 24 hours, mark it as a losing trade
            if long_result is None:
                long_result = False
            if short_result is None:
                short_result = False

            # save results
            entry.long_result = long_result
            entry.short_result = short_result
            batch.append(entry)

            if len(batch) >= 100:
                RickisMetrics.objects.bulk_update(batch, ["long_result", "short_result"])
                batch.clear()

        if batch:
            RickisMetrics.objects.bulk_update(batch, ["long_result", "short_result"])

        print("done.")
