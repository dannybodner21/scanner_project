# management/commands/fill_btc_gaps.py

import requests
from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice
from django.utils import timezone
from datetime import timedelta
import time

COINAPI_KEY = '01293e2a-dcf1-4e81-8310-c6aa9d0cb743'
BASE_URL = 'https://rest.coinapi.io/v1/ohlcv/BINANCE_SPOT_BTC_USDT/history'
HEADERS = {'X-CoinAPI-Key': COINAPI_KEY}

class Command(BaseCommand):
    help = 'Fill BTC price gaps'

    def handle(self, *args, **options):
        # Fetch all BTC prices ordered by timestamp
        prices = CoinAPIPrice.objects.filter(coin="BTC").order_by('timestamp')

        last_ts = None
        missing = []

        for price in prices:
            if last_ts:
                delta = price.timestamp - last_ts
                if delta > timedelta(minutes=5):
                    gap_minutes = int(delta.total_seconds() / 60)
                    # Find all missing 5-min intervals
                    for i in range(5, gap_minutes, 5):
                        missing_ts = last_ts + timedelta(minutes=i)
                        missing.append(missing_ts)
            last_ts = price.timestamp

        print(f"Found {len(missing)} missing timestamps")

        for ts in missing:
            print(f"Filling {ts}")

            params = {
                'period_id': '5MIN',
                'time_start': ts.isoformat(),
                'time_end': (ts + timedelta(minutes=5)).isoformat(),
                'limit': 1
            }

            try:
                response = requests.get(BASE_URL, headers=HEADERS, params=params)
                response.raise_for_status()
                data = response.json()

                if data:
                    candle = data[0]
                    CoinAPIPrice.objects.update_or_create(
                        coin='BTC',
                        timestamp=candle['time_period_start'],
                        defaults={
                            'open': candle['price_open'],
                            'high': candle['price_high'],
                            'low': candle['price_low'],
                            'close': candle['price_close'],
                            'volume': candle['volume_traded'],
                        }
                    )
                else:
                    print(f"No data returned for {ts}")

            except Exception as e:
                print(f"Error for {ts}: {e}")

            # avoid hitting rate limits
            time.sleep(0.25)  # 4 requests/sec

        print("Gap fill complete ✅")
