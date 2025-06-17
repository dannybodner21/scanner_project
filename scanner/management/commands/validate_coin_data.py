import datetime
from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice

class Command(BaseCommand):
    help = "Validate historical CoinAPIPrice data for gaps, flats, and volume issues."

    def add_arguments(self, parser):
        parser.add_argument("coin", type=str)

    def handle(self, *args, **options):
        coin = options['coin'].upper()

        start_time = datetime.datetime(2019, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
        end_time = datetime.datetime(2025, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)

        # Load all timestamps for this coin
        self.stdout.write(f"Loading {coin} data...")
        qs = CoinAPIPrice.objects.filter(
            coin=coin,
            timestamp__gte=start_time,
            timestamp__lt=end_time
        ).values_list("timestamp", "open", "high", "low", "close", "volume")

        data = {ts: (o, h, l, c, v) for ts, o, h, l, c, v in qs}

        gaps = []
        flats = []
        zero_volume = []

        expected = start_time
        interval = datetime.timedelta(minutes=5)
        total_expected = 0

        while expected < end_time:
            total_expected += 1
            row = data.get(expected)

            if row is None:
                gaps.append(expected)
            else:
                o, h, l, c, v = row
                if o == h == l == c:
                    flats.append(expected)
                if v is None or v == 0:
                    zero_volume.append(expected)

            expected += interval

        self.stdout.write(f"✅ Total expected intervals: {total_expected}")
        self.stdout.write(f"✅ Total found: {len(data)}")

        if not gaps and not flats and not zero_volume:
            self.stdout.write("🎯 Data check PASSED: No issues found.")
        else:
            if gaps:
                self.stdout.write(f"❌ Found {len(gaps)} gaps:")
                for ts in gaps:
                    self.stdout.write(str(ts))

            if flats:
                self.stdout.write(f"❌ Found {len(flats)} flat candles:")
                for ts in flats:
                    self.stdout.write(str(ts))

            if zero_volume:
                self.stdout.write(f"❌ Found {len(zero_volume)} zero volume candles:")
                for ts in zero_volume:
                    self.stdout.write(str(ts))

        self.stdout.write("🚀 Data validation complete.")
