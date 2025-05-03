# to backfill the change_5min variable

from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from scanner.helpers import calculate_price_change_five_min
from django.utils.timezone import make_aware
from datetime import datetime

class Command(BaseCommand):
    help = 'Backfill missing change_5m values in RickisMetrics'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 5, 3))

        entries = RickisMetrics.objects.filter(
            timestamp__gte=start,
            timestamp__lt=end,
            change_5m__isnull=True
        )

        total = entries.count()
        print(f"Backfilling {total} entries...")

        for i, entry in enumerate(entries.iterator()):
            change = calculate_price_change_five_min(entry.coin.symbol, entry.timestamp)
            entry.change_5m = change
            entry.save(update_fields=['change_5m'])

            if i % 100 == 0:
                print(f"{i}/{total} done")

        print("✅ Done backfilling change_5m.")
