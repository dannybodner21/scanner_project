
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime
from scanner.models import RickisMetrics
import requests
from decimal import Decimal
import time

API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'


url = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/historical'
headers = {"X-CMC_PRO_API_KEY": API_KEY}

class Command(BaseCommand):

    help = 'Backfill high, low, open, close'

    def handle(self, *args, **kwargs):

        # get metrics from given dates
        metrics = Metrics.objects.filter(
            timestamp__gte=make_aware(datetime(2025, 3, 22)),
            timestamp__lt=make_aware(datetime(2025, 5, 3)),
        ).order_by("timestamp")

        # loop through metrics
        for entry in enumerate(metrics, start=1):

            date_str = entry.timestamp.date().isoformat()
            symbol = entry.coin.symbol

            # fetch ohlcv data and save
            try:
                data = self.fetch_ohlcv(symbol, date_str)
                quote = data["quote"]["USD"]
                entry.open = Decimal(str(quote["open"]))
                entry.high_24h = Decimal(str(quote["high"]))
                entry.low_24h = Decimal(str(quote["low"]))
                entry.close = Decimal(str(quote["close"]))
                entry.save()

            except Exception as e:
                print(f"error: {e}")

            time.sleep(2)

    # functin to get ohlcv data from Coinmarketcap
    def fetch_ohlcv(self, symbol, date_str):
        params = {
            "symbol": symbol,
            "time_period": "daily",
            "time_start": date_str,
            "time_end": date_str,
        }
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        quotes = response.json().get("data", {}).get("quotes")
        return quotes[0] if quotes else None
