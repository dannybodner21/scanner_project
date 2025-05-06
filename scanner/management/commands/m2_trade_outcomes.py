from datetime import datetime, timedelta
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics

class Command(BaseCommand):
    help = "Label RickisMetrics entries with long_result and short_result based on TP/SL rules"

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 4, 18))

        entries = RickisMetrics.objects.filter(timestamp__gte=start, timestamp__lt=end).select_related("coin").order_by("timestamp")
        total = entries.count()
        print(f"📊 Checking {total} RickisMetrics entries...")

        batch = []

        for i, entry in enumerate(entries, 1):
            entry_price = float(entry.price)
            tp_long = entry_price * 1.10
            sl_long = entry_price * 0.98
            tp_short = entry_price * 0.90
            sl_short = entry_price * 1.02

            future_metrics = RickisMetrics.objects.filter(
                coin=entry.coin,
                timestamp__gt=entry.timestamp,
                timestamp__lte=entry.timestamp + timedelta(hours=4)
            ).order_by("timestamp")

            long_result = None
            short_result = None

            for fm in future_metrics:
                try:
                    price = float(fm.price)

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

            entry.long_result = long_result
            entry.short_result = short_result
            batch.append(entry)

            if len(batch) >= 100:
                RickisMetrics.objects.bulk_update(batch, ["long_result", "short_result"])
                print(f"✅ {i}/{total} updated")
                batch.clear()

        if batch:
            RickisMetrics.objects.bulk_update(batch, ["long_result", "short_result"])
            print("✅ Final batch saved.")

        print("🏁 Trade result labeling complete.")
