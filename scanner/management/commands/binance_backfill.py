from django.core.management.base import BaseCommand
from scanner.models import Coin, ShortIntervalData
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
from decimal import Decimal
import requests
import time

BINANCE_URL = "https://api.binance.com/api/v3/klines"

def round_to_five_minutes(dt):
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)

class Command(BaseCommand):
    help = "Backfill ShortIntervalData for BTC using Binance 5m candles"

    def add_arguments(self, parser):
        parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
        parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')

    def handle(self, *args, **kwargs):
        start_date = datetime.strptime(kwargs['start'], "%Y-%m-%d")
        end_date = datetime.strptime(kwargs['end'], "%Y-%m-%d")
        end_date += timedelta(days=1)  # include full end day

        coin = Coin.objects.get(symbol="BTC")
        symbol = "BTCUSDT"
        interval = "5m"
        limit = 1000  # max Binance allows

        current = start_date
        while current < end_date:
            start_ts = int(current.timestamp() * 1000)
            end_ts = int((current + timedelta(minutes=limit * 5)).timestamp() * 1000)

            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ts,
                "endTime": end_ts,
                "limit": limit
            }

            try:
                response = requests.get(BINANCE_URL, params=params)
                response.raise_for_status()
                candles = response.json()

                for candle in candles:
                    raw_time = datetime.utcfromtimestamp(candle[0] / 1000)
                    rounded_time = round_to_five_minutes(raw_time)
                    open_time = make_aware(rounded_time)
                    close_price = Decimal(candle[4])
                    volume = Decimal(candle[5])

                    obj, created = ShortIntervalData.objects.get_or_create(
                        coin=coin,
                        timestamp=open_time,
                        defaults={
                            'price': close_price,
                            'volume_5min': volume
                        }
                    )

                    if created:
                        print(f"✅ Inserted BTC at {open_time}")
                    else:
                        print(f"⏩ Skipped (exists) BTC at {open_time}")

                time.sleep(0.5)  # avoid hammering API

            except Exception as e:
                print(f"❌ Error fetching candles: {e}")
                break

            current += timedelta(minutes=limit * 5)

        print("✅ Backfill complete.")
