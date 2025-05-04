from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
from scanner.models import RickisMetrics
import requests
import time
from decimal import Decimal

CMC_API_KEY = "7dd5dd98-35d0-475d-9338-407631033cd9"
HEADERS = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
CMC_OHLCV_URL = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/historical"

class Command(BaseCommand):
    help = "Backfill RickisMetrics with 24h high/low for every timestamp using CMC OHLCV historical API"

    def handle(self, *args, **kwargs):
        start_date = make_aware(datetime(2025, 3, 22))
        end_date = make_aware(datetime(2025, 5, 3))

        queryset = RickisMetrics.objects.select_related("coin").filter(
            timestamp__gte=start_date,
            timestamp__lt=end_date
        ).order_by("timestamp")

        total = queryset.count()
        self.stdout.write(f"📊 Processing {total} entries...")

        for idx, entry in enumerate(queryset):
            symbol = entry.coin.symbol
            date_str = entry.timestamp.strftime("%Y-%m-%d")
            next_day_str = (entry.timestamp + timedelta(days=1)).strftime("%Y-%m-%d")

            try:
                res = requests.get(
                    CMC_OHLCV_URL,
                    headers=HEADERS,
                    params={
                        "symbol": symbol,
                        "time_start": date_str,
                        "time_end": next_day_str,
                        "interval": "5m"
                    },
                    timeout=10
                )
                res.raise_for_status()
                data = res.json().get("data", {}).get("quotes", [])

                timestamp_unix = int(entry.timestamp.timestamp())
                closest = min(
                    data,
                    key=lambda item: abs(
                        int(datetime.fromisoformat(item["timestamp"]).timestamp()) - timestamp_unix
                    )
                )

                quote = closest["quote"]["USD"]
                entry.high_24h = Decimal(str(quote.get("high", 0)))
                entry.low_24h = Decimal(str(quote.get("low", 0)))
                entry.save()

                self.stdout.write(f"✅ [{idx + 1}/{total}] {symbol} {entry.timestamp} updated")

            except Exception as e:
                self.stdout.write(f"❌ [{idx + 1}/{total}] Error for {symbol} @ {entry.timestamp}: {e}")

            time.sleep(1.1)  # Rate limit respect
