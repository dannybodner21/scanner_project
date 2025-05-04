
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime
from scanner.models import RickisMetrics
import requests
from decimal import Decimal
import time

API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
CMC_OHLCV_URL = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/historical'
HEADERS = {"X-CMC_PRO_API_KEY": API_KEY}

class Command(BaseCommand):
    help = 'Backfill high_24h and low_24h for RickisMetrics using daily OHLCV data'

    def handle(self, *args, **kwargs):
        metrics = RickisMetrics.objects.filter(
            timestamp__gte=make_aware(datetime(2025, 3, 22)),
            timestamp__lt=make_aware(datetime(2025, 4, 13)),
        ).order_by("timestamp")

        total = metrics.count()
        print(f"📊 Processing {total} entries...")

        for idx, entry in enumerate(metrics, start=1):
            date_str = entry.timestamp.date().isoformat()
            symbol = entry.coin.symbol

            try:
                data = self.fetch_ohlcv(symbol, date_str)
                if not data:
                    raise ValueError("No OHLCV data returned")

                quote = data["quote"]["USD"]
                entry.high_24h = Decimal(str(quote["high"]))
                entry.low_24h = Decimal(str(quote["low"]))
                entry.save()
                print(f"✅ [{idx}/{total}] {symbol} @ {date_str}")
            except Exception as e:
                print(f"❌ [{idx}/{total}] Error for {symbol} @ {date_str}: {e}")

            time.sleep(1.1)  # Respect rate limits

    def fetch_ohlcv(self, symbol, date_str):
        params = {
            "symbol": symbol,
            "time_period": "daily",
            "time_start": date_str,
            "time_end": date_str,
        }
        response = requests.get(CMC_OHLCV_URL, headers=HEADERS, params=params)
        response.raise_for_status()
        quotes = response.json().get("data", {}).get("quotes")
        return quotes[0] if quotes else None
