import time
import requests
from datetime import datetime, timedelta, timezone
from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice

class Command(BaseCommand):
    help = 'Backfill BTCUSDT 5-minute candles from CoinAPI'

    def handle(self, *args, **options):
        API_KEY = '01293e2a-dcf1-4e81-8310-c6aa9d0cb743'
        BASE_URL = 'https://rest.coinapi.io/v1/ohlcv/BINANCE_SPOT_BTC_USDT/history'
        INTERVAL = '5MIN'
        LIMIT = 1000  # max allowed

        # Jan 1, 2025 00:00 UTC
        start_time = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        # June 10, 2025 23:59 UTC rounded down to 23:55:00
        end_time = datetime(2025, 6, 10, 23, 55, 0, tzinfo=timezone.utc)

        current_start = start_time

        while current_start <= end_time:
            params = {
                'period_id': INTERVAL,
                'time_start': current_start.isoformat(),
                'limit': LIMIT
            }

            headers = {'X-CoinAPI-Key': API_KEY}

            response = requests.get(BASE_URL, params=params, headers=headers)
            if response.status_code == 429:
                self.stdout.write('Rate limit hit. Sleeping 60 seconds.')
                time.sleep(60)
                continue
            response.raise_for_status()
            data = response.json()

            if not data:
                self.stdout.write(f"No data returned for {current_start}.")
                break

            for candle in data:
                timestamp = datetime.fromisoformat(candle['time_period_start'].replace("Z", "+00:00"))
                timestamp = timestamp.replace(second=0, microsecond=0)

                obj, created = CoinAPIPrice.objects.get_or_create(
                    coin='BTCUSDT',
                    timestamp=timestamp,
                    defaults={
                        'open': candle['price_open'],
                        'high': candle['price_high'],
                        'low': candle['price_low'],
                        'close': candle['price_close'],
                        'volume': candle['volume_traded'],
                    }
                )

            self.stdout.write(f"Inserted candles up to {timestamp}.")

            # Move to next batch
            current_start = timestamp + timedelta(minutes=5)

            time.sleep(1.1)  # stay below CoinAPI rate limits

        self.stdout.write("✅ Backfill complete.")
