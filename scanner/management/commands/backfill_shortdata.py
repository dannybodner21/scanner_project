from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
from scanner.models import Coin, ShortIntervalData
import requests
import time
from decimal import Decimal


def round_to_five_minutes(dt):
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)

API_KEY_TWO = 'c35740fd-4f78-45b5-9350-c4afdd929432'
CMC_URL = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'

class Command(BaseCommand):

    help = 'Backfill ShortIntervalData entries between a start and end time'

    def add_arguments(self, parser):
        parser.add_argument('--start', type=str, help='Start time in YYYY-MM-DDTHH:MM format (UTC)')
        parser.add_argument('--end', type=str, help='End time in YYYY-MM-DDTHH:MM format (UTC)')
        parser.add_argument('--symbol', type=str, help='(Optional) Coin symbol to backfill (e.g., BTC)')

    # python manage.py backfill_shortdata --start="2024-03-22T00:00" --end="2024-04-20T23:55" --symbol=BTC

    def handle(self, *args, **kwargs):
        start = make_aware(datetime.strptime(kwargs['start'], "%Y-%m-%dT%H:%M"))
        end = make_aware(datetime.strptime(kwargs['end'], "%Y-%m-%dT%H:%M"))

        symbol = kwargs.get('symbol')
        if symbol:
            coins = Coin.objects.filter(symbol=symbol.upper())
        else:
            coins = Coin.objects.all()

        cmc_ids = [coin.cmc_id for coin in coins]

        current = start
        batch_size = 100

        while current < end:

            print(f"⏳ Backfilling for {current}...")

            for i in range(0, len(cmc_ids), batch_size):

                batch = cmc_ids[i:i + batch_size]
                params = {
                    "id": ",".join(map(str, batch)),
                    "convert": "USD"
                }
                headers = {
                    "Accepts": "application/json",
                    "X-CMC_PRO_API_KEY": API_KEY_TWO,
                }

                try:
                    response = requests.get(CMC_URL, headers=headers, params=params)
                    response.raise_for_status()
                    data = response.json().get("data", {})

                    for cmc_id in batch:

                        coin = Coin.objects.get(cmc_id=cmc_id)
                        quote = data.get(str(cmc_id), {}).get("quote", {}).get("USD", {})

                        timestamp = round_to_five_minutes(current)
                        price = quote.get("price")
                        volume_24h = quote.get("volume_24h")
                        circulating_supply = data.get(str(cmc_id), {}).get("circulating_supply")

                        if price is None or volume_24h is None:
                            continue

                        short_data, created = ShortIntervalData.objects.get_or_create(
                            coin=coin,
                            timestamp=timestamp,
                            defaults={
                                'price': Decimal(str(price)),
                                'volume_5min': Decimal(str(volume_24h)),
                                'circulating_supply': circulating_supply
                            }
                        )

                        if not created:
                            updated = False
                            if short_data.price is None:
                                short_data.price = Decimal(str(price))
                                updated = True
                            if short_data.volume_5min is None:
                                short_data.volume_5min = Decimal(str(volume_24h))
                                updated = True
                            if short_data.circulating_supply is None and circulating_supply is not None:
                                short_data.circulating_supply = circulating_supply
                                updated = True

                            if updated:
                                short_data.save()
                                print(f"🔁 Updated missing fields for {coin.symbol} at {timestamp}")
                            else:
                                print(f"⏩ Skipped (already exists) {coin.symbol} at {timestamp}")

                        else:
                            print(f"✅ Inserted {coin.symbol} at {timestamp}")

                        print(f"✅ Inserted {coin.symbol} at {timestamp}")

                    time.sleep(2)

                except Exception as e:
                    print(f"❌ Error during API call: {e}")

            current += timedelta(minutes=5)

        print("✅ Backfill complete.")
