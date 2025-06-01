

from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from django.db import close_old_connections
from scanner.models import RickisMetrics, Coin
import requests
import time

# nohup python manage.py binance_check > output.log 2>&1 &
# tail -f output.log

BINANCE_URL = 'https://api.binance.com/api/v3/klines'

class Command(BaseCommand):
    help = 'Full range price accuracy check between Binance.US and RickisMetrics for BTCUSD from March 23 to May 23, 2025'

    def handle(self, *args, **kwargs):
        symbol = "BTCUSD"
        binance_symbol = "BTCUSD"

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
            start_time = int(current_date.timestamp() * 1000)
            end_time = int((current_date + timedelta(days=1)).timestamp() * 1000)

            candles = self.fetch_binance_candles(binance_symbol, start_time, end_time)
            if not candles:
                print(f"⚠️ No candles fetched for {current_date.date()}")
                current_date += timedelta(days=1)
                continue

            for candle in candles:
                ts = int(candle[0]) // 1000
                ts_dt = make_aware(datetime.utcfromtimestamp(ts))
                close_price = float(candle[4])

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
                    print(f"❌ Mismatch at {ts_dt}: DB={db_price}, Binance={close_price}, Diff={difference:.5f}")

                total += 1

            print(f"✅ Done checking {current_date.date()} — total candles: {len(candles)}")
            current_date += timedelta(days=1)
            time.sleep(1)  # Sleep between days to avoid burst limits

        print(f"\n🎯 Final Summary for BTCUSD {start_date.date()} to {end_date.date()}")
        print(f"Total checked: {total}")
        print(f"✅ Matches: {matches}")
        print(f"❌ Mismatches: {mismatches}")

    def fetch_binance_candles(self, symbol, start_time, end_time, interval="5m"):
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
        }
        try:
            response = requests.get(BINANCE_URL, params=params, timeout=10)
            response.raise_for_status()
            time.sleep(1)
            return response.json()
        except Exception as e:
            print(f"❌ Error fetching candles: {e}")
            return []
