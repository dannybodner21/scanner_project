from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from django.db import close_old_connections
from scanner.models import RickisMetrics, Coin
import requests
import time

BINANCE_US_URL = 'https://api.binance.us/api/v3/klines'

class Command(BaseCommand):
    help = 'Test price accuracy between Binance.US and local RickisMetrics for BTCUSD on March 23, 2025'

    def handle(self, *args, **kwargs):
        symbol = "BTCUSD"
        binance_symbol = "BTCUSD"  # Binance.US uses USD instead of USDT

        start_date = datetime(2025, 3, 23)
        end_date = start_date + timedelta(days=1)

        start_time = int(start_date.timestamp() * 1000)  # in milliseconds
        end_time = int(end_date.timestamp() * 1000)

        print(f"🚀 Fetching 5m candles for {symbol} on {start_date.date()}")

        candles = self.fetch_binance_candles(binance_symbol, start_time, end_time)

        if not candles:
            print(f"❌ No candles fetched for {symbol}")
            return

        try:
            coin = Coin.objects.get(symbol="BTC")
        except Coin.DoesNotExist:
            print(f"❌ BTC not found in your database Coin table.")
            return

        total = 0
        matches = 0
        mismatches = 0

        for candle in candles:
            ts = int(candle[0]) // 1000  # Binance gives open time in ms
            ts_dt = make_aware(datetime.utcfromtimestamp(ts))
            close_price = float(candle[4])  # Close price

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

        print(f"🎯 Total checked: {total}")
        print(f"✅ Matches: {matches}")
        print(f"❌ Mismatches: {mismatches}")

    def fetch_binance_candles(self, symbol, start_time, end_time, interval="5m"):
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000  # Max allowed per request
        }
        try:
            response = requests.get(BINANCE_US_URL, params=params, timeout=10)
            response.raise_for_status()
            time.sleep(1)  # Sleep a little to be safe with rate limits
            return response.json()
        except Exception as e:
            print(f"❌ Error fetching candles: {e}")
            return []
