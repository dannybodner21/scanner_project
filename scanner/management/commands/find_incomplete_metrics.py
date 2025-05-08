from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from django.db.models import Q

class Command(BaseCommand):
    help = "Find RickisMetrics entries missing price, high, low, open, or close"

    def handle(self, *args, **kwargs):
        missing_qs = RickisMetrics.objects.filter(
            Q(price__isnull=True) |
            Q(high_24h__isnull=True) |
            Q(low_24h__isnull=True) |
            Q(open__isnull=True) |
            Q(close__isnull=True)
        ).select_related("coin").order_by("timestamp")

        total = missing_qs.count()
        print(f"❗ Found {total} incomplete RickisMetrics entries.")

        for rm in missing_qs:
            missing_fields = []
            if rm.price is None: missing_fields.append("price")
            if rm.high_24h is None: missing_fields.append("high_24h")
            if rm.low_24h is None: missing_fields.append("low_24h")
            if rm.open is None: missing_fields.append("open")
            if rm.close is None: missing_fields.append("close")

            print(f"{rm.coin.symbol} @ {rm.timestamp}: Missing {', '.join(missing_fields)}")
