from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
from scanner.models import Coin, RickisMetrics
from decimal import Decimal
import requests
import time

API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
CMC_OHLCV_URL = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/historical'

class Command(BaseCommand):
    help = 'Backfill RickisMetrics with high_24h and low_24h using hourly OHLCV data from CMC'

    def handle(self, *args, **kwargs):
        start_date = make_aware(datetime(2025, 3, 22))
        end_date = make_aware(datetime(2025, 4, 13))

        coins = Coin.objects.all()
        total = RickisMetrics.objects.filter(timestamp__gte=start_date, timestamp__lt=end_date).count()
        queryset = RickisMetrics.objects.filter(timestamp__gte=start_date, timestamp__lt=end_date).select_related('coin').order_by('timestamp')

        self.stdout.write(f"📊 Processing {total} entries...")

        for idx, metric in enumerate(queryset, 1):
            symbol = metric.coin.symbol
            ts = metric.timestamp
            date_str = ts.strftime('%Y-%m-%d')
            hour_before = (ts - timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S')

            try:
                ohlcv_data = self.fetch_hourly_ohlcv(symbol, hour_before, ts.strftime('%Y-%m-%dT%H:%M:%S'))
                if ohlcv_data is None:
                    raise ValueError("No OHLCV data returned")

                # Match the exact timestamp in results
                quote = next((q for q in ohlcv_data if q['quote']['USD']['timestamp'].startswith(ts.strftime('%Y-%m-%dT%H'))), None)
                if not quote:
                    raise ValueError("No matching OHLCV record for timestamp")

                metric.high_24h = Decimal(str(quote['quote']['USD']['high']))
                metric.low_24h = Decimal(str(quote['quote']['USD']['low']))
                metric.save()
            except Exception as e:
                self.stdout.write(f"❌ [{idx}/{total}] Error for {symbol} @ {ts}: {e}")

            if idx % 100 == 0:
                time.sleep(1.5)

    def fetch_hourly_ohlcv(self, symbol, time_start, time_end):
        headers = {"X-CMC_PRO_API_KEY": API_KEY}
        params = {
            "symbol": symbol,
            "time_period": "hourly",
            "time_start": time_start,
            "time_end": time_end,
            "convert": "USD"
        }

        res = requests.get(CMC_OHLCV_URL, headers=headers, params=params)
        res.raise_for_status()
        data = res.json().get("data", {})
        return data.get("quotes")
