import datetime
from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice

class Command(BaseCommand):
    help = "Check for missing 5-minute gaps in BTCUSDT data"

    def handle(self, *args, **options):
        start_time = datetime.datetime(2025, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        end_time = datetime.datetime(2025, 6, 10, 23, 55, 0, tzinfo=datetime.timezone.utc)

        timestamps = list(CoinAPIPrice.objects.filter(coin='SNXUSDT')
                          .values_list('timestamp', flat=True).order_by('timestamp'))

        expected = start_time
        gaps = []

        for ts in timestamps:
            while expected < ts:
                gaps.append(expected)
                expected += datetime.timedelta(minutes=5)
            expected += datetime.timedelta(minutes=5)

        while expected <= end_time:
            gaps.append(expected)
            expected += datetime.timedelta(minutes=5)

        if not gaps:
            print("✅ No gaps found.")
        else:
            print(f"Found {len(gaps)} gaps:")
            for gap in gaps:
                print(gap)
