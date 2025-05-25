from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from dateutil.parser import isoparse
from scanner.models import RickisMetrics, Coin
import requests
import time

CMC_API_KEY = '6520549c-03bb-41cd-86e3-30355ece87ba'
HEADERS = {"Accepts": "application/json", "X-CMC_PRO_API_KEY": CMC_API_KEY}
CMC_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"

def round_to_five_minutes(dt):
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)

class Command(BaseCommand):
    help = 'Verify and correct historical prices for RickisMetrics from March 22 to April 23, 2025'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 4, 23))

        coins = Coin.objects.filter(symbol="BONK")  # Narrow for debugging; expand when fixed
        errors = []

        for coin in coins:
            print(f"\n🔍 Checking {coin.symbol}")
            current = start
            while current <= end:
                rounded_ts = round_to_five_minutes(current)
                metric = RickisMetrics.objects.filter(coin=coin, timestamp=rounded_ts).first()

                if not metric:
                    current += timedelta(minutes=5)
                    continue

                try:
                    # Query using exact timestamp
                    ts_iso = rounded_ts.isoformat()
                    response = requests.get(
                        CMC_URL,
                        headers=HEADERS,
                        params={
                            "symbol": coin.symbol,
                            "time_start": ts_iso,
                            "time_end": ts_iso,
                            "interval": "5m",
                            "convert": "USD"
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    quotes = data.get("data", {}).get("quotes", [])

                    if not quotes:
                        print(f"⚠️ No quote for {coin.symbol} at {rounded_ts}")
                        current += timedelta(minutes=5)
                        continue

                    quote_ts = isoparse(quotes[0]["timestamp"]).replace(second=0, microsecond=0)
                    if abs((quote_ts - rounded_ts).total_seconds()) > 300:
                        print(f"⚠️ Quote too far from target for {coin.symbol} at {rounded_ts}")
                        current += timedelta(minutes=5)
                        continue

                    cmc_price = float(quotes[0]["quote"]["USD"]["close"])
                    db_price = float(metric.price)

                    if abs(db_price - cmc_price) / cmc_price > 0.01:
                        print(f"❌ {coin.symbol} at {rounded_ts} - DB: {db_price}, CMC: {cmc_price}")
                        metric.price = cmc_price
                        metric.save()
                        errors.append((coin.symbol, rounded_ts))

                except Exception as e:
                    print(f"❌ API Error for {coin.symbol} at {rounded_ts}: {e}")

                current += timedelta(minutes=5)
                time.sleep(2)  # Stay within CMC rate limits

        print("\n✅ Finished checking prices.")
        print(f"Total corrections made: {len(errors)}")
        for sym, ts in errors:
            print(f"Corrected {sym} at {ts}")
