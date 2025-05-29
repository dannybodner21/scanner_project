from django.core.management.base import BaseCommand
from django.db.models import Q
from scanner.models import RickisMetrics, Coin
from datetime import datetime, timedelta
import requests
import time

CMC_API_KEY = '6520549c-03bb-41cd-86e3-30355ece87ba'
HEADERS = {
    "Accepts": "application/json",
    "X-CMC_PRO_API_KEY": CMC_API_KEY
}
CMC_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"

RATE_LIMIT_CALLS_PER_MIN = 20
SECONDS_BETWEEN_CALLS = 60.0 / RATE_LIMIT_CALLS_PER_MIN

class Command(BaseCommand):
    help = 'Fix high, low, open, close values using CMC OHLCV historical data'

    def handle(self, *args, **options):
        start_date = datetime(2025, 3, 23)
        end_date = datetime(2025, 5, 23)
        days = (end_date - start_date).days + 1

        coins = Coin.objects.all()

        for coin in coins:
            for i in range(days):
                date = start_date + timedelta(days=i)
                date_str = date.strftime('%Y-%m-%d')

                try:
                    response = requests.get(
                        CMC_URL,
                        headers=HEADERS,
                        params={
                            'symbol': coin.symbol,
                            'time_start': date_str,
                            'time_end': date_str,
                            'interval': 'daily'
                        }
                    )
                    data = response.json()

                    quotes = data.get("data", {}).get("quotes", [])
                    if not quotes:
                        print(f"❌ No OHLC data for {coin.symbol} on {date_str}")
                        continue

                    quote = quotes[0]
                    open_price = quote['quote']['USD']['open']
                    close_price = quote['quote']['USD']['close']
                    high_price = quote['quote']['USD']['high']
                    low_price = quote['quote']['USD']['low']

                    updated_count = RickisMetrics.objects.filter(
                        coin=coin,
                        timestamp__date=date.date()
                    ).filter(
                        Q(high_24h__isnull=True) | ~Q(high_24h=high_price) |
                        Q(low_24h__isnull=True) | ~Q(low_24h=low_price) |
                        Q(open__isnull=True) | ~Q(open=open_price) |
                        Q(close__isnull=True) | ~Q(close=close_price)
                    ).update(
                        high_24h=high_price,
                        low_24h=low_price,
                        open=open_price,
                        close=close_price
                    )

                    print(f"✅ Updated {updated_count} rows for {coin.symbol} on {date_str}")
                    time.sleep(SECONDS_BETWEEN_CALLS)

                except Exception as e:
                    print(f"❌ Error for {coin.symbol} on {date_str}: {e}")
                    time.sleep(SECONDS_BETWEEN_CALLS)
                    continue
