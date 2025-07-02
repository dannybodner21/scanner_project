import datetime
from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice

# python manage.py validate_coin_data BTCUSDT - good
'''

'''

# python manage.py validate_coin_data ETHUSDT - good
'''

'''

# python manage.py validate_coin_data XRPUSDT - good
'''

'''

# python manage.py validate_coin_data LTCUSDT - good
'''

'''

# python manage.py validate_coin_data SOLUSDT - not good
'''

("2025-01-01 01:05:00", "2025-01-01 01:05:00"),

'''

# python manage.py validate_coin_data DOGEUSDT - good
'''

'''

# python manage.py validate_coin_data LINKUSDT - good
'''

'''

# python manage.py validate_coin_data DOTUSDT - good
'''

'''

# python manage.py validate_coin_data SHIBUSDT - good
'''

'''

# python manage.py validate_coin_data ADAUSDT - good
'''

'''

# python manage.py validate_coin_data UNIUSDT - not good
'''
❌ Found gaps:
2024-02-04 17:10:00+00:00
2024-02-05 05:35:00+00:00
2024-05-09 02:55:00+00:00

'''

# python manage.py validate_coin_data AVAXUSDT - not good
'''

❌ Found 13 flat candles:
2022-12-25 03:55:00+00:00
2023-04-29 02:40:00+00:00
2023-08-12 04:45:00+00:00
2023-08-12 07:40:00+00:00
2023-08-27 03:55:00+00:00
2023-09-02 14:10:00+00:00
2023-09-05 23:40:00+00:00
2023-09-08 23:45:00+00:00
2023-09-08 23:50:00+00:00
2023-09-09 03:50:00+00:00
2023-09-09 05:25:00+00:00
2023-10-01 01:35:00+00:00
2023-10-15 06:05:00+00:00

'''

# python manage.py validate_coin_data XLMUSDT - fine
'''

❌ Found 1 flat candles:
2024-05-31 22:45:00+00:00


'''



class Command(BaseCommand):
    help = "Validate historical CoinAPIPrice data for gaps, flats, and volume issues."

    def add_arguments(self, parser):
        parser.add_argument("coin", type=str)

    def handle(self, *args, **options):
        coin = options['coin'].upper()

        start_time = datetime.datetime(2022, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
        end_time = datetime.datetime(2025, 6, 30, 23, 55, tzinfo=datetime.timezone.utc)

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
