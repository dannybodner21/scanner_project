
from django.core.management.base import BaseCommand
from datetime import datetime, timedelta, timezone
from scanner.models import CoinAPIPrice
import requests
import time

COINAPI_KEY = '01293e2a-dcf1-4e81-8310-c6aa9d0cb743'

class Command(BaseCommand):
    help = 'Detect and fill missing BTCUSDT 5-min candles'

    def handle(self, *args, **options):
        symbol = 'BTCUSDT'
        coinapi_symbol = 'BINANCE_SPOT_BTC_USDT'
        start_time = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2025, 6, 10, 23, 55, tzinfo=timezone.utc)

        expected_timestamps = set()
        t = start_time
        while t <= end_time:
            expected_timestamps.add(t)
            t += timedelta(minutes=5)

        existing_timestamps = set(
            CoinAPIPrice.objects
            .filter(coin=symbol, timestamp__gte=start_time, timestamp__lte=end_time)
            .values_list('timestamp', flat=True)
        )

        missing = expected_timestamps - existing_timestamps

        if not missing:
            self.stdout.write(self.style.SUCCESS("✅ No missing timestamps — data is fully complete"))
            return

        self.stdout.write(self.style.WARNING(f"Found {len(missing)} missing timestamps. Starting fill..."))

        for ts in sorted(missing):
            iso_time = ts.isoformat().replace('+00:00', 'Z')
            url = f'https://rest.coinapi.io/v1/ohlcv/{coinapi_symbol}/history?period_id=5MIN&time_start={iso_time}&limit=1'
            headers = {'X-CoinAPI-Key': COINAPI_KEY}

            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()

                if data:
                    candle = data[0]

                    CoinAPIPrice.objects.create(
                        coin=symbol,
                        timestamp=datetime.fromisoformat(candle['time_period_start'].replace('Z', '+00:00')),
                        open=candle['price_open'],
                        high=candle['price_high'],
                        low=candle['price_low'],
                        close=candle['price_close'],
                        volume=candle['volume_traded']
                    )
                    print(f"✅ Filled {ts}")
                else:
                    print(f"⚠️ No data returned for {ts}")

            except Exception as e:
                print(f"❌ Error fetching {ts}: {e}")

            time.sleep(1.1)  # Avoid hitting rate limits

        self.stdout.write(self.style.SUCCESS("🔥 Gap filling complete 🔥"))
