
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics, Coin
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
import requests
import time
from decimal import Decimal

CMC_API_KEY = "7dd5dd98-35d0-475d-9338-407631033cd9"
CMC_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"

HEADERS = {
    "X-CMC_PRO_API_KEY": CMC_API_KEY
}

MAX_RETRIES = 5


class Command(BaseCommand):
    help = "Backfill OHLCV (daily) data into RickisMetrics"

    def handle(self, *args, **kwargs):
        coins = Coin.objects.all()

        for coin in coins:
            symbol = coin.symbol.upper()
            metric_dates = RickisMetrics.objects.filter(
                coin=coin,
                open__isnull=True
            ).datetimes("timestamp", "day")

            if not metric_dates:
                continue

            print(f"\n🔍 {symbol} | {len(metric_dates)} dates to update")

            for day in metric_dates:
                start_str = day.strftime("%Y-%m-%d")
                end_str = (day + timedelta(days=1)).strftime("%Y-%m-%d")

                retries = 0
                success = False

                while retries < MAX_RETRIES and not success:
                    params = {
                        "symbol": symbol,
                        "time_start": start_str,
                        "time_end": end_str,
                        "interval": "daily",
                        "convert": "USD"
                    }

                    try:
                        time.sleep(1)
                        res = requests.get(CMC_URL, headers=HEADERS, params=params)

                        if res.status_code == 429:
                            wait_time = 15 + retries * 15
                            print(f"⏳ {symbol} {start_str} — Rate limit hit. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                            retries += 1
                            continue

                        if res.status_code != 200:
                            print(f"❌ {symbol} {start_str} — API error {res.status_code}")
                            break

                        quotes = res.json().get("data", {}).get("quotes", [])
                        if not quotes:
                            print(f"⚠️ {symbol} {start_str} — No data returned")
                            break

                        quote = quotes[0]["quote"]["USD"]
                        open_price = Decimal(str(quote.get("open")))
                        high_price = Decimal(str(quote.get("high")))
                        low_price = Decimal(str(quote.get("low")))
                        close_price = Decimal(str(quote.get("close")))

                        updated = RickisMetrics.objects.filter(
                            coin=coin,
                            timestamp__date=day,
                            open__isnull=True
                        ).update(
                            open=open_price,
                            high_24h=high_price,
                            low_24h=low_price,
                            close=close_price
                        )

                        print(f"✅ {symbol} {start_str} — Updated {updated} rows")
                        success = True

                    except Exception as e:
                        print(f"💥 {symbol} {start_str} — Error: {e}")
                        break

                time.sleep(3)  # base delay between requests
