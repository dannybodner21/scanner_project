import datetime
from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice



# python manage.py check_flat_windows BTCUSDT
# python manage.py check_flat_windows ETHUSDT
# python manage.py check_flat_windows XRPUSDT
# python manage.py check_flat_windows LTCUSDT
# python manage.py check_flat_windows SOLUSDT
# python manage.py check_flat_windows DOGEUSDT
# python manage.py check_flat_windows LINKUSDT
# python manage.py check_flat_windows DOTUSDT
# python manage.py check_flat_windows SHIBUSDT
# python manage.py check_flat_windows ADAUSDT
# python manage.py check_flat_windows UNIUSDT
# python manage.py check_flat_windows AVAXUSDT
# python manage.py check_flat_windows XLMUSDT

# python manage.py check_flat_windows HBARUSDT 183 flat windows


class Command(BaseCommand):
    help = "Detect flat windows (consecutive flat candles) for a coin."

    def add_arguments(self, parser):
        parser.add_argument("coin", type=str)

    def handle(self, *args, **options):
        coin = options['coin'].upper()

        start_time = datetime.datetime(2022, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
        end_time = datetime.datetime(2025, 6, 30, 23, 55, tzinfo=datetime.timezone.utc)

        self.stdout.write(f"Scanning {coin} for flat windows...")

        qs = CoinAPIPrice.objects.filter(
            coin=coin,
            timestamp__gte=start_time,
            timestamp__lt=end_time
        ).order_by('timestamp')

        flat_windows = []
        current_window = []
        interval = datetime.timedelta(minutes=5)

        expected_time = start_time

        for row in qs:
            ts = row.timestamp
            # Catch up if DB is missing rows (handles any gaps safely)
            while expected_time < ts:
                if current_window:
                    if len(current_window) >= 2:
                        flat_windows.append(current_window)
                    current_window = []
                expected_time += interval

            # Evaluate current candle
            if row.open == row.high == row.low == row.close:
                current_window.append(ts)
            else:
                if current_window:
                    if len(current_window) >= 2:
                        flat_windows.append(current_window)
                    current_window = []

            expected_time += interval

        # Handle any final window
        if current_window and len(current_window) >= 2:
            flat_windows.append(current_window)

        # Output results
        if not flat_windows:
            self.stdout.write("🎯 No flat windows found. Data is clean.")
        else:
            self.stdout.write(f"❌ Found {len(flat_windows)} flat windows:")
            for window in flat_windows:
                start = window[0]
                end = window[-1]
                length = len(window)
                self.stdout.write(f"Window: {start} → {end} ({length} candles)")

        self.stdout.write("🚀 Flat window check complete.")
