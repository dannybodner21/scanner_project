from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
from scanner.models import RickisMetrics
import requests
from decimal import Decimal
import time

API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
CMC_URL = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/historical'

class Command(BaseCommand):
    help = 'Backfill high_24h and low_24h using hourly OHLCV data from CoinMarketCap'

    def handle(self, *args, **kwargs):
        start_date = make_aware(datetime(2025, 3, 22))
        end_date = make_aware(datetime(2025, 4, 13))

        metrics = RickisMetrics.objects.filter(timestamp__gte=start_date, timestamp__lt=end_date).order_by('timestamp')
        total = metrics.count()
        print(f"\U0001F4CA Processing {total} entries...")

        for idx, rm in enumerate(metrics, start=1):
            symbol = rm.coin.symbol
            ts = rm.timestamp.replace(minute=0, second=0, microsecond=0)
            time_start = (ts - timedelta(hours=1)).isoformat()
            time_end = ts.isoformat()

            try:
                data = self.fetch_hourly_ohlcv(symbol, time_start, time_end)
                if data:
                    ohlcv = data[0]['quote']['USD']
                    rm.high_24h = Decimal(str(ohlcv.get('high', 0)))
                    rm.low_24h = Decimal(str(ohlcv.get('low', 0)))
                    rm.save()
                else:
                    print(f"⚠️ No OHLCV data for {symbol} @ {ts}")
            except Exception as e:
                print(f"❌ [{idx}/{total}] Error for {symbol} @ {ts}: {e}")

            time.sleep(1.1)  # CMC rate limit

    def fetch_hourly_ohlcv(self, symbol, time_start, time_end):
        headers = {"X-CMC_PRO_API_KEY": API_KEY}
        params = {
            "symbol": symbol,
            "time_start": time_start,
            "time_end": time_end,
            "time_period": "hourly"
        }
        res = requests.get(CMC_URL, headers=headers, params=params)
        res.raise_for_status()
        return res.json().get('data', {}).get('quotes', [])
