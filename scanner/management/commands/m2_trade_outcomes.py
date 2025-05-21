from datetime import datetime, timedelta
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics

class Command(BaseCommand):
    help = "Label each RickisMetrics row as a winning or losing long/short trade (TP=10%, SL=2%)."

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 4, 12))

        entries = RickisMetrics.objects.filter(
            timestamp__gte=start, timestamp__lt=end
        ).select_related("coin").order_by("timestamp")

        total = entries.count()
        print(f"📊 Labeling {total} entries...")

        batch = []

        long_wins = 0
        short_wins = 0

        for index, entry in enumerate(entries, 1):
            entry_price = float(entry.price)
            tp_long = entry_price * 1.06
            sl_long = entry_price * 0.98
            tp_short = entry_price * 0.94
            sl_short = entry_price * 1.02

            future_metrics = RickisMetrics.objects.filter(
                coin=entry.coin,
                timestamp__gt=entry.timestamp,
                timestamp__lte=entry.timestamp + timedelta(hours=24)
            ).only("timestamp", "price").order_by("timestamp")

            long_result = None
            short_result = None

            for metric in future_metrics:
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

            # Default to losing if neither hit
            if long_result is None:
                long_result = False
            if short_result is None:
                short_result = False

            if long_result:
                long_wins += 1
            if short_result:
                short_wins += 1

            entry.long_result = long_result
            entry.short_result = short_result
            batch.append(entry)

            if len(batch) >= 100:
                RickisMetrics.objects.bulk_update(batch, ["long_result", "short_result"])
                batch.clear()

            if index % 10000 == 0:
                print(f"✅ Processed {index} of {total}")

        if batch:
            RickisMetrics.objects.bulk_update(batch, ["long_result", "short_result"])

        print("\n🏁 Done labeling trades.")
        print(f"✅ Winning Long Trades:  {long_wins}")
        print(f"✅ Winning Short Trades: {short_wins}")
