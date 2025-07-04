

# python manage.py delete_flat_windows LINKUSDT
# python manage.py delete_flat_windows SHIBUSDT
# python manage.py delete_flat_windows ADAUSDT
# python manage.py delete_flat_windows UNIUSDT
# python manage.py delete_flat_windows AVAXUSDT
# python manage.py delete_flat_windows XLMUSDT

# python manage.py delete_flat_windows HBARUSDT 183 flat windows


import datetime
from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice

class Command(BaseCommand):
    help = "Delete all flat candle windows for ETHUSDT."

    def handle(self, *args, **options):
        flat_windows = [
            ("2022-03-15 03:10:00", "2022-03-15 03:10:00"),
            ("2022-05-28 11:50:00", "2022-05-28 11:50:00"),
            ("2022-12-09 09:00:00", "2022-12-09 09:00:00"),
            ("2022-12-09 22:45:00", "2022-12-09 22:45:00"),
            ("2022-12-11 01:20:00", "2022-12-11 01:20:00"),
            ("2022-12-11 05:10:00", "2022-12-11 05:10:00"),
            ("2022-12-11 07:40:00", "2022-12-11 07:40:00"),
            ("2022-12-11 09:35:00", "2022-12-11 09:35:00"),
            ("2022-12-24 18:35:00", "2022-12-24 18:35:00"),
            ("2022-12-24 19:40:00", "2022-12-24 19:40:00"),
            ("2022-12-25 10:55:00", "2022-12-25 10:55:00"),
            ("2023-04-02 04:45:00", "2023-04-02 04:45:00"),
            ("2023-04-08 10:35:00", "2023-04-08 10:35:00"),
            ("2023-05-20 06:10:00", "2023-05-20 06:10:00"),
            ("2023-05-22 23:35:00", "2023-05-22 23:35:00"),
            ("2023-05-23 11:35:00", "2023-05-23 11:35:00"),
            ("2023-05-23 21:50:00", "2023-05-23 21:50:00"),
            ("2023-05-27 08:35:00", "2023-05-27 08:35:00"),
            ("2023-05-29 19:35:00", "2023-05-29 19:35:00"),
            ("2023-06-03 08:20:00", "2023-06-03 08:20:00"),
            ("2023-06-03 20:25:00", "2023-06-03 20:25:00"),
            ("2023-06-03 20:40:00", "2023-06-03 20:40:00"),
            ("2023-06-07 04:30:00", "2023-06-07 04:30:00"),
            ("2023-06-08 01:20:00", "2023-06-08 01:20:00"),
            ("2023-06-08 04:10:00", "2023-06-08 04:10:00"),
            ("2023-06-08 04:40:00", "2023-06-08 04:40:00"),
            ("2023-06-08 05:10:00", "2023-06-08 05:10:00"),
            ("2023-06-08 05:30:00", "2023-06-08 05:30:00"),
            ("2023-06-08 06:45:00", "2023-06-08 06:45:00"),
            ("2023-06-08 06:55:00", "2023-06-08 06:55:00"),
            ("2023-06-08 07:05:00", "2023-06-08 07:05:00"),
            ("2023-06-08 08:00:00", "2023-06-08 08:00:00"),
            ("2023-06-08 19:15:00", "2023-06-08 19:15:00"),
            ("2023-06-08 21:05:00", "2023-06-08 21:05:00"),
            ("2023-06-08 23:00:00", "2023-06-08 23:00:00"),
            ("2023-08-26 03:55:00", "2023-08-26 03:55:00"),
            ("2023-09-16 22:45:00", "2023-09-16 22:45:00"),
            ("2023-09-23 21:15:00", "2023-09-23 21:15:00"),
            ("2023-09-24 11:15:00", "2023-09-24 11:15:00"),
            ("2023-09-26 13:25:00", "2023-09-26 13:25:00"),
            ("2023-10-01 03:45:00", "2023-10-01 03:45:00"),
            ("2023-10-07 07:35:00", "2023-10-07 07:35:00"),
            ("2023-10-08 15:50:00", "2023-10-08 15:50:00"),
            ("2023-10-12 01:50:00", "2023-10-12 01:50:00"),
            ("2023-10-14 05:20:00", "2023-10-14 05:20:00"),
            ("2023-10-14 07:35:00", "2023-10-14 07:35:00"),
            ("2023-10-14 09:55:00", "2023-10-14 09:55:00"),
            ("2023-10-15 07:20:00", "2023-10-15 07:20:00"),
            ("2023-10-19 22:20:00", "2023-10-19 22:20:00"),
            ("2023-10-21 00:55:00", "2023-10-21 00:55:00"),
            ("2024-02-03 17:10:00", "2024-02-03 17:10:00"),
            ("2024-05-12 05:55:00", "2024-05-12 05:55:00"),
            ("2024-05-15 06:55:00", "2024-05-15 06:55:00"),
        ]

        total_deleted = 0
        for start_str, end_str in flat_windows:
            start = datetime.datetime.fromisoformat(start_str).replace(tzinfo=datetime.timezone.utc)
            end = datetime.datetime.fromisoformat(end_str).replace(tzinfo=datetime.timezone.utc)
            deleted, _ = CoinAPIPrice.objects.filter(
                coin="UNIUSDT",
                timestamp__gte=start,
                timestamp__lte=end
            ).delete()
            total_deleted += deleted

        self.stdout.write(self.style.SUCCESS(f"✅ Deleted {total_deleted} flat candles for HBARUSDT."))
