
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
from scanner.models import RickisMetrics
import requests
import time
from decimal import Decimal

API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
CMC_OHLCV_URL = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/historical'

class Command(BaseCommand):
    help = 'Backfill 24h high and low into RickisMetrics from CMC hourly OHLCV data'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 4, 13))

        entries = RickisMetrics.objects.filter(
            timestamp__gte=start, timestamp__lt=end
        ).order_by('timestamp')

        total = entries.count()
        print(f"\U0001F4CA Processing {total} entries...")

        for i, rm in enumerate(entries, start=1):
            ts = rm.timestamp
            date_str = ts.strftime('%Y-%m-%d')
            coin_symbol = rm.coin.symbol

            try:
                data = self.fetch_ohlcv_hourly(symbol=coin_symbol, date=date_str)
                if not data:
                    print(f"\u274C [{i}/{total}] No OHLCV data returned for {coin_symbol} @ {ts}")
                    continue

                # Find the hourly bar matching the metric timestamp
                ts_unix = int(ts.timestamp())
                matching_bar = next((item for item in data if int(datetime.fromisoformat(item['quote']['USD']['timestamp'].replace('Z', '+00:00')).timestamp()) == ts_unix), None)

                if matching_bar:
                    quote = matching_bar['quote']['USD']
                    rm.high_24h = Decimal(str(quote.get('high', 0)))
                    rm.low_24h = Decimal(str(quote.get('low', 0)))
                    rm.save()
                    print(f"✅ [{i}/{total}] {coin_symbol} @ {ts} - High: {rm.high_24h}, Low: {rm.low_24h}")
                else:
                    print(f"\u274C [{i}/{total}] No matching hourly bar for {coin_symbol} @ {ts}")

            except Exception as e:
                print(f"\u274C [{i}/{total}] Error for {coin_symbol} @ {ts}: {e}")

            time.sleep(1.1)  # Prevent hitting rate limits

    def fetch_ohlcv_hourly(self, symbol, date):
        headers = {"X-CMC_PRO_API_KEY": API_KEY}
        params = {
            "symbol": symbol,
            "time_period": "hourly",
            "time_start": date,  # ISO date only
            "time_end": date,
            "convert": "USD"
        }

        response = requests.get(CMC_OHLCV_URL, headers=headers, params=params)
        response.raise_for_status()
        quotes = response.json().get("data", {}).get("quotes", [])
        return quotes
