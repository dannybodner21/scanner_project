from datetime import datetime, timedelta
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics, Coin
from django.db.models import Q

class Command(BaseCommand):
    help = 'Backfill missing change_1h and change_24h using RickisMetrics prices directly.'

    def handle(self, *args, **kwargs):
        start_date = make_aware(datetime(2025, 5, 1))
        end_date = make_aware(datetime(2025, 5, 13))
        all_coins = Coin.objects.all()

        for coin in all_coins:
            print(f"\n🚀 Processing {coin.symbol}")

            metrics = RickisMetrics.objects.filter(
                coin=coin,
                timestamp__gte=start_date,
                timestamp__lte=end_date
            ).filter(Q(change_1h__in=[None, 0]) | Q(change_24h__in=[None, 0])).order_by("timestamp")

            updated = []

            for metric in metrics:
                ts = metric.timestamp

                try:
                    # Get the price 1 hour ago
                    ts_1h_ago = ts - timedelta(hours=1)
                    price_1h_ago = RickisMetrics.objects.filter(
                        coin=coin, timestamp__lte=ts_1h_ago
                    ).order_by('-timestamp').values_list('price', flat=True).first()

                    # Get the price 24 hours ago
                    ts_24h_ago = ts - timedelta(hours=24)
                    price_24h_ago = RickisMetrics.objects.filter(
                        coin=coin, timestamp__lte=ts_24h_ago
                    ).order_by('-timestamp').values_list('price', flat=True).first()

                    if metric.price and price_1h_ago and metric.change_1h in [None, 0]:
                        metric.change_1h = ((metric.price - price_1h_ago) / price_1h_ago) * 100

                    if metric.price and price_24h_ago and metric.change_24h in [None, 0]:
                        metric.change_24h = ((metric.price - price_24h_ago) / price_24h_ago) * 100

                    updated.append(metric)

                except Exception as e:
                    print(f"❌ Error at {ts} for {coin.symbol}: {e}")
                    continue

            if updated:
                RickisMetrics.objects.bulk_update(updated, ["change_1h", "change_24h"])
                print(f"✅ {coin.symbol}: filled {len(updated)} entries")

        print("\n🎉 All missing change_1h and change_24h values filled from price history.")
