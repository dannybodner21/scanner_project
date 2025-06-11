import time
import requests
from datetime import datetime, timedelta, timezone
from django.core.management.base import BaseCommand
from scanner.models import BinancePrice

class Command(BaseCommand):
    help = "Backfill Binance BTCUSDT 5-minute candles starting from 2025-01-01"

    def handle(self, *args, **kwargs):
        symbol = "BTCUSDT"
        interval = "5m"
        start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end_date = datetime.now(timezone.utc)

        base_url = "https://api.binance.com/api/v3/klines"
        limit = 1000  # Binance max per call

        current_start = int(start_date.timestamp() * 1000)

        while True:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "limit": limit,
            }
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data:
                print("✅ All data fetched.")
                break

            rows_saved = 0

            for candle in data:
                open_time = datetime.utcfromtimestamp(candle[0] / 1000).replace(tzinfo=timezone.utc)
                open_price = float(candle[1])
                high_price = float(candle[2])
                low_price = float(candle[3])
                close_price = float(candle[4])
                volume = float(candle[5])

                obj, created = BinancePrice.objects.update_or_create(
                    coin=symbol,
                    timestamp=open_time,
                    defaults={
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": volume,
                    }
                )
                if created:
                    rows_saved += 1

            print(f"✅ Saved {rows_saved} rows for startTime {current_start}")

            # Advance to next batch
            current_start = data[-1][0] + (5 * 60 * 1000)  # next 5-minute candle

            # Respect Binance API limits (avoid getting banned)
            time.sleep(0.2)
