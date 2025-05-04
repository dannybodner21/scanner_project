
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
import requests
import time
from decimal import Decimal

CMC_API_KEY = "7dd5dd98-35d0-475d-9338-407631033cd9"
HEADERS = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
BASE_URL = "https://pro-api.coinmarketcap.com/v2"

class Command(BaseCommand):
    help = "Backfill high_24h and low_24h from CMC hourly OHLCV"

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 4, 13))

        queryset = RickisMetrics.objects.filter(
            timestamp__gte=start,
            timestamp__lt=end
        ).select_related("coin").order_by("timestamp")

        self.stdout.write(f"📊 Processing {queryset.count()} entries...")

        for i, metric in enumerate(queryset, 1):
            symbol = metric.coin.symbol
            ts = metric.timestamp

            try:
                high, low = self.get_hourly_ohlcv(symbol, ts)
                if high is not None:
                    metric.high_24h = high
                if low is not None:
                    metric.low_24h = low
                metric.save()
            except Exception as e:
                self.stdout.write(f"❌ [{i}/{queryset.count()}] Error for {symbol} @ {ts}: {e}")
                continue

            if i % 100 == 0:
                time.sleep(2)

    def get_hourly_ohlcv(self, symbol, ts):
        url = f"{BASE_URL}/cryptocurrency/ohlcv/historical"
        params = {
            "symbol": symbol,
            "time_period": "hourly",
            "time_start": (ts - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "time_end": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "convert": "USD"
        }

        res = requests.get(url, headers=HEADERS, params=params)
        res.raise_for_status()
        quotes = res.json().get("data", {}).get("quotes", [])

        if not quotes:
            raise ValueError("No OHLCV data returned")

        quote = quotes[0].get("quote", {}).get("USD", {})
        high = Decimal(str(quote.get("high"))) if "high" in quote else None
        low = Decimal(str(quote.get("low"))) if "low" in quote else None

        return high, low
