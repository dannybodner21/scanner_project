
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime
from scanner.models import RickisMetrics
import requests
from decimal import Decimal
import time

API_KEY = '6520549c-03bb-41cd-86e3-30355ece87ba'
url = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/historical'
headers = {"X-CMC_PRO_API_KEY": API_KEY}

class Command(BaseCommand):

    help = 'Backfill high, low, open, close'

    def handle(self, *args, **kwargs):

        # get metrics from given dates
        metrics = Metrics.objects.filter(
            timestamp__gte=make_aware(datetime(2025, 5, 9)),
            timestamp__lt=make_aware(datetime(2025, 5, 24)),
        ).order_by("timestamp")

        # loop through metrics
        for i, entry in enumerate(metrics, start=1):

            date_str = entry.timestamp.date().isoformat()
            symbol = entry.coin.symbol

            # fetch ohlcv data and save
            try:
                data = self.fetch_ohlcv(symbol, date_str)
                if not data:
                    print(f"⚠️ No OHLCV data for {symbol} on {date_str}")
                    continue

                quote = data["quote"]["USD"]
                entry.open = Decimal(str(quote["open"]))
                entry.high_24h = Decimal(str(quote["high"]))
                entry.low_24h = Decimal(str(quote["low"]))
                entry.close = Decimal(str(quote["close"]))
                entry.save()
                print(f"✅ Updated {symbol} @ {entry.timestamp}")

            except Exception as e:
                print(f"❌ Error for {symbol} on {date_str}: {e}")

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


'''
CHECK EACH METRIC FOR MISSING HIGH, LOW, OPEN, CLOSE

python manage.py shell -c "
from django.utils.timezone import make_aware
from datetime import datetime
from django.db.models import Q
from scanner.models import RickisMetrics

start = make_aware(datetime(2025, 5, 9))
end = make_aware(datetime(2025, 5, 24))

missing = RickisMetrics.objects.filter(
    timestamp__gte=start,
    timestamp__lt=end
).filter(
    Q(open__isnull=True) | Q(open=0) |
    Q(close__isnull=True) | Q(close=0) |
    Q(high_24h__isnull=True) | Q(high_24h=0) |
    Q(low_24h__isnull=True) | Q(low_24h=0)
).count()

print(f'Missing or invalid entries (open/close/high/low): {missing}')
"


'''
