import datetime
from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice


# 1. BTCUSDT - good
# 2. ETHUSDT - good
# 3. XRPUSDT - good
# 4. LTCUSDT - good
# 5. SOLUSDT - good
# 6. DOGEUSDT - good
# 7. LINKUSDT - good
# 8. DOTUSDT - good
# 9. SHIBUSDT - good
# 10. ADAUSDT - good
# 11. UNIUSDT - good
# 12. AVAXUSDT - good
# 13. XLMUSDT - good


# 14. HBARUSDT -


class Command(BaseCommand):
    help = "Check for missing 5-minute gaps in BTCUSDT data"

    def handle(self, *args, **options):
        start_time = datetime.datetime(2022, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        end_time = datetime.datetime(2025, 6, 30, 23, 55, 0, tzinfo=datetime.timezone.utc)

        # Load all existing timestamps into a set for fast lookup
        existing_timestamps = set(
            ts.replace(tzinfo=datetime.timezone.utc)
            for ts in CoinAPIPrice.objects.filter(coin='UNIUSDT')
            .values_list('timestamp', flat=True)
        )

        gaps = []
        expected = start_time

        while expected <= end_time:
            if expected not in existing_timestamps:
                gaps.append(expected)
            expected += datetime.timedelta(minutes=5)

        if not gaps:
            print("✅ No gaps found.")
        else:
            print(f"❌ Found {len(gaps)} gaps:")
            for gap in gaps:
                print(gap)
