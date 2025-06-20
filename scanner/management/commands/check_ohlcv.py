import datetime
from django.core.management.base import BaseCommand
from scanner.models import Coin, CoinAPIPrice


class Command(BaseCommand):
    help = "Check for CoinAPIPrice entries with missing or zero OHLCV data (hardcoded for ADAUSDT)"

    def handle(self, *args, **kwargs):
        coin_symbol = "ADA"
        start_date = datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc)
        end_date = datetime.datetime(2025, 6, 19, 23, 55, 0, tzinfo=datetime.timezone.utc)

        try:
            coin = Coin.objects.get(symbol=coin_symbol)
        except Coin.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Coin '{coin_symbol}' not found."))
            return

        incomplete_qs = CoinAPIPrice.objects.filter(
            coin=coin,
            timestamp__range=(start_date, end_date)
        ).filter(
            open__isnull=True
        ) | CoinAPIPrice.objects.filter(
            coin=coin,
            timestamp__range=(start_date, end_date),
            open=0
        ) | CoinAPIPrice.objects.filter(
            coin=coin,
            timestamp__range=(start_date, end_date),
            high__isnull=True
        ) | CoinAPIPrice.objects.filter(
            coin=coin,
            timestamp__range=(start_date, end_date),
            high=0
        ) | CoinAPIPrice.objects.filter(
            coin=coin,
            timestamp__range=(start_date, end_date),
            low__isnull=True
        ) | CoinAPIPrice.objects.filter(
            coin=coin,
            timestamp__range=(start_date, end_date),
            low=0
        ) | CoinAPIPrice.objects.filter(
            coin=coin,
            timestamp__range=(start_date, end_date),
            close__isnull=True
        ) | CoinAPIPrice.objects.filter(
            coin=coin,
            timestamp__range=(start_date, end_date),
            close=0
        ) | CoinAPIPrice.objects.filter(
            coin=coin,
            timestamp__range=(start_date, end_date),
            volume__isnull=True
        ) | CoinAPIPrice.objects.filter(
            coin=coin,
            timestamp__range=(start_date, end_date),
            volume=0
        )

        count = incomplete_qs.count()

        if count == 0:
            self.stdout.write(self.style.SUCCESS("✅ No incomplete OHLCV entries found."))
        else:
            self.stdout.write(self.style.WARNING(f"⚠️ Found {count} incomplete OHLCV entries:\n"))
            for candle in incomplete_qs.order_by("timestamp"):
                self.stdout.write(f"- {candle.timestamp.isoformat()}")
