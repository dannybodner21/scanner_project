from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from django.db import close_old_connections
from scanner.models import RickisMetrics, Coin
import requests
import time

# nohup python manage.py polygon_check > output.log 2>&1 &
# tail -f output.log

POLYGON_URL = 'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}'

# Replace with your actual Polygon.io API Key
POLYGON_API_KEY = 'qq9Sptr4VfkonQimqFJEgc3oyXoaJ54L'

class Command(BaseCommand):
    help = 'Full range price accuracy check between Polygon.io and RickisMetrics for BTC-USD from March 23 to May 23, 2025'

    def handle(self, *args, **kwargs):
        symbol = "BTC-USD"
        polygon_symbol = "X:BTC-USD"

        start_date = datetime(2025, 3, 23)
        end_date = datetime(2025, 5, 23)

        print(f"🚀 Fetching 5m candles for {symbol} from {start_date.date()} to {end_date.date()}")

        try:
            coin = Coin.objects.get(symbol="BTC")
        except Coin.DoesNotExist:
            print(f"❌ BTC not found in your Coin table.")
            return

        current_date = start_date
        total = 0
        matches = 0
        mismatches = 0

        while current_date < end_date:
            from_date = current_date.strftime('%Y-%m-%d')
            to_date = (current_date + timedelta(days=1)).strftime('%Y-%m-%d')

            candles = self.fetch_polygon_candles(polygon_symbol, from_date, to_date)
            if not candles:
                print(f"⚠️ No candles fetched for {current_date.date()}")
                current_date += timedelta(days=1)
                continue

            for candle in candles:
                ts = int(candle['t']) // 1000
                ts_dt = make_aware(datetime.utcfromtimestamp(ts))
                close_price = float(candle['c'])

                close_old_connections()
                metric = RickisMetrics.objects.filter(coin=coin, timestamp=ts_dt).first()
                if not metric:
                    print(f"⚠️ No metric found at {ts_dt} — skipping.")
                    continue

                db_price = float(metric.price)
                difference = abs(db_price - close_price) / close_price

                if difference <= 0.01:
                    matches += 1
                else:
                    mismatches += 1
                    print(f"❌ Mismatch at {ts_dt}: DB={db_price}, Polygon={close_price}, Diff={difference:.5f}")

                total += 1

            print(f"✅ Done checking {current_date.date()} — total candles: {len(candles)}")
            current_date += timedelta(days=1)
            time.sleep(1)  # Sleep between days to avoid burst limits

        print(f"\n🎯 Final Summary for {symbol} {start_date.date()} to {end_date.date()}")
        print(f"Total checked: {total}")
        print(f"✅ Matches: {matches}")
        print(f"❌ Mismatches: {mismatches}")

    def fetch_polygon_candles(self, symbol, from_date, to_date, multiplier=5, timespan='minute'):
        url = POLYGON_URL.format(
            ticker=symbol,
            multiplier=multiplier,
            timespan=timespan,
            from_date=from_date,
            to_date=to_date
        )
        params = {
            "apiKey": POLYGON_API_KEY,
            "limit": 10000
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            time.sleep(1)  # Sleep to respect Polygon rate limits
            data = response.json()
            return data.get('results', [])
        except Exception as e:
            print(f"❌ Error fetching candles: {e}")
            return []
