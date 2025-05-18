from datetime import datetime, timedelta
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics, Coin
from django.db.models import Q

class Command(BaseCommand):
    help = 'Fill all missing change_1h and change_24h using RickisMetrics prices.'

    def handle(self, *args, **kwargs):
        start_date = make_aware(datetime(2025, 4, 13))
        end_date = make_aware(datetime(2025, 5, 13))

        print("📥 Loading all RickisMetrics rows into memory...")
        all_metrics = RickisMetrics.objects.filter(
            timestamp__gte=start_date,
            timestamp__lte=end_date
        ).only('id', 'coin_id', 'timestamp', 'price', 'change_1h', 'change_24h')

        print("🔍 Indexing metrics by (coin_id, timestamp)...")
        metrics_by_key = {
            (m.coin_id, m.timestamp): m for m in all_metrics
        }

        print("🧩 Finding metrics with missing change_1h or change_24h...")
        to_update = []

        for metric in all_metrics:
            if metric.change_1h not in [None, 0] and metric.change_24h not in [None, 0]:
                continue  # already filled

            price_now = metric.price
            if price_now is None:
                continue

            coin_id = metric.coin_id
            ts = metric.timestamp

            # Find exact 1 hour and 24 hour old prices
            price_1h_ago = metrics_by_key.get((coin_id, ts - timedelta(hours=1)))
            price_24h_ago = metrics_by_key.get((coin_id, ts - timedelta(hours=24)))

            changed = False

            if price_1h_ago and price_1h_ago.price and metric.change_1h in [None, 0]:
                try:
                    metric.change_1h = ((price_now - price_1h_ago.price) / price_1h_ago.price) * 100
                    changed = True
                except ZeroDivisionError:
                    pass

            if price_24h_ago and price_24h_ago.price and metric.change_24h in [None, 0]:
                try:
                    metric.change_24h = ((price_now - price_24h_ago.price) / price_24h_ago.price) * 100
                    changed = True
                except ZeroDivisionError:
                    pass

            if changed:
                to_update.append(metric)

        print(f"📋 Found {len(to_update)} metrics to update.")
        for m in to_update[:20]:  # show first 20 for debugging
            print(f"{m.coin_id} at {m.timestamp} — 1h: {m.change_1h:.2f}%, 24h: {m.change_24h:.2f}%")

        if to_update:
            RickisMetrics.objects.bulk_update(to_update, ["change_1h", "change_24h"])
            print(f"✅ Updated {len(to_update)} rows.")
        else:
            print("✅ Nothing to update.")
