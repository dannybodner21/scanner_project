
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
    help = 'Backfill high_24h, low_24h, open, close from CoinMarketCap OHLCV data'

    def handle(self, *args, **kwargs):
        start_date = make_aware(datetime(2025, 5, 9))
        end_date = make_aware(datetime(2025, 5, 24))

        current_date = start_date
        while current_date < end_date:
            for symbol in RickisMetrics.objects.filter(timestamp__date=current_date.date()).values_list('coin__symbol', flat=True).distinct():
                metrics = RickisMetrics.objects.filter(
                    coin__symbol=symbol,
                    timestamp__date=current_date.date()
                )

                try:
                    data = self.fetch_ohlcv(symbol, current_date.date().isoformat())
                    if not data:
                        print(f"⚠️ No OHLCV data for {symbol} on {current_date.date()}")
                        continue

                    quote = data["quote"]["USD"]
                    for entry in metrics:
                        entry.open = Decimal(str(quote["open"]))
                        entry.high_24h = Decimal(str(quote["high"]))
                        entry.low_24h = Decimal(str(quote["low"]))
                        entry.close = Decimal(str(quote["close"]))
                        entry.save()

                    print(f"✅ Updated {symbol} on {current_date.date()} ({len(metrics)} entries)")

                except Exception as e:
                    print(f"❌ Error for {symbol} on {current_date.date()}: {e}")

                time.sleep(1.2)
            current_date += timedelta(days=1)

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
