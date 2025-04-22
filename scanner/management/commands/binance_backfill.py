from django.core.management.base import BaseCommand
from scanner.models import Coin, ShortIntervalData
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
from decimal import Decimal
import requests
import time

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/{id}/market_chart/range"

COINGECKO_IDS = {
    "BTC": "bitcoin"
}

def round_to_five_minutes(dt):
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)

class Command(BaseCommand):
    help = "Backfill ShortIntervalData for BTC using CoinGecko 5m data"

    def add_arguments(self, parser):
        parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
        parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')

    def handle(self, *args, **kwargs):
        start_date = datetime.strptime(kwargs['start'], "%Y-%m-%d")
        end_date = datetime.strptime(kwargs['end'], "%Y-%m-%d") + timedelta(days=1)

        coin = Coin.objects.get(symbol="BTC")
        gecko_id = COINGECKO_IDS["BTC"]

        from_timestamp = int(start_date.timestamp())
        to_timestamp = int(end_date.timestamp())

        params = {
            "vs_currency": "usd",
            "from": from_timestamp,
            "to": to_timestamp
        }

        try:
            response = requests.get(COINGECKO_URL.format(id=gecko_id), params=params)
            response.raise_for_status()
            data = response.json()
            prices = data.get("prices", [])
            volumes = data.get("total_volumes", [])

            volume_dict = {round_to_five_minutes(datetime.utcfromtimestamp(v[0] / 1000)): Decimal(v[1]) for v in volumes}

            for entry in prices:
                ts = round_to_five_minutes(datetime.utcfromtimestamp(entry[0] / 1000))
                aware_ts = make_aware(ts)
                price = Decimal(entry[1])
                volume = volume_dict.get(ts, Decimal("0"))

                obj, created = ShortIntervalData.objects.get_or_create(
                    coin=coin,
                    timestamp=aware_ts,
                    defaults={
                        'price': price,
                        'volume_5min': volume
                    }
                )

                if created:
                    print(f"✅ Inserted BTC at {aware_ts}")
                else:
                    print(f"⏩ Skipped (exists) BTC at {aware_ts}")

        except Exception as e:
            print(f"❌ Error fetching data from CoinGecko: {e}")

        print("✅ Backfill complete.")
