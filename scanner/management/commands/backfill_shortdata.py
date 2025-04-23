
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
from scanner.models import Coin, ShortIntervalData
import requests
from decimal import Decimal
import time


# python manage.py backfill_shortdata --start="2024-03-22T00:00" --end="2024-04-20T23:55" --symbol=BTC

# python manage.py backfill_shortdata --start="2024-03-24T00:00" --end="2024-03-25T00:05"

# python manage.py backfill_shortdata --start="2024-03-28T00:20" --end="2024-04-20T00:05"


def round_to_five_minutes(dt):
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)


API_KEY_TWO = 'c35740fd-4f78-45b5-9350-c4afdd929432'
CMC_URL = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'

class Command(BaseCommand):
    help = 'Backfill ShortIntervalData entries between a start and end time'

    def add_arguments(self, parser):
        parser.add_argument('--start', type=str, help='Start time in YYYY-MM-DDTHH:MM format (UTC)')
        parser.add_argument('--end', type=str, help='End time in YYYY-MM-DDTHH:MM format (UTC)')

    def handle(self, *args, **kwargs):
        start = make_aware(datetime.strptime(kwargs['start'], "%Y-%m-%dT%H:%M"))
        end = make_aware(datetime.strptime(kwargs['end'], "%Y-%m-%dT%H:%M"))

        target_symbols = [
            "BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LEO", "LINK",
            "AVAX", "XLM", "TON", "SHIB", "SUI", "HBAR", "BCH", "DOT", "LTC", "HYPE",
            "BGB", "DAI", "PI", "XMR", "UNI", "PEPE", "OKB", "APT", "GT", "NEAR",
            "ONDO", "TAO", "ICP", "ETC", "RENDER", "MNT", "KAS", "CRO", "AAVE", "POL",
            "VET", "FIL", "TRUMP", "ALGO", "ENA", "ATOM", "TIA", "FET", "ARB", "S",
            "KCS", "DEXE", "OP", "JUP", "MKR", "XDC", "STX", "FLR", "EOS", "WLD",
            "IP", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT", "FORM", "QNT", "PAXG",
            "CRV", "JASMY", "SAND", "GALA", "NEXO", "CORE", "RAY", "KAIA", "LDO", "THETA",
            "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI", "XCN"
        ]

        coins = Coin.objects.filter(symbol__in=target_symbols)
        coin_map = {coin.cmc_id: coin for coin in coins}
        cmc_ids = list(coin_map.keys())

        current = start
        batch_size = 100

        # Cache all existing entries in one query
        existing_entries = set(
            ShortIntervalData.objects.filter(
                coin__in=coins,
                timestamp__gte=start,
                timestamp__lte=end
            ).values_list("coin_id", "timestamp")
        )

        while current < end:
            print(f"⏳ Backfilling for {current}...")
            timestamp = round_to_five_minutes(current)

            for i in range(0, len(cmc_ids), batch_size):

                time.sleep(1.6)

                batch = cmc_ids[i:i + batch_size]
                params = {"id": ",".join(map(str, batch)), "convert": "USD"}
                headers = {"Accepts": "application/json", "X-CMC_PRO_API_KEY": API_KEY_TWO}

                try:
                    response = requests.get(CMC_URL, headers=headers, params=params)
                    response.raise_for_status()
                    data = response.json().get("data", {})
                except Exception as e:
                    print(f"❌ Error during API call: {e}")
                    continue

                to_create = []
                to_update = []

                for cmc_id in batch:
                    coin = coin_map.get(cmc_id)
                    quote = data.get(str(cmc_id), {}).get("quote", {}).get("USD", {})

                    price = quote.get("price")
                    volume_24h = quote.get("volume_24h")
                    circulating_supply = data.get(str(cmc_id), {}).get("circulating_supply")

                    if price is None or volume_24h is None:
                        continue

                    key = (coin.id, timestamp)
                    if key in existing_entries:
                        existing = ShortIntervalData.objects.filter(coin=coin, timestamp=timestamp).first()
                        updated = False
                        if existing:
                            if existing.price is None:
                                existing.price = Decimal(str(price))
                                updated = True
                            if existing.volume_5min is None:
                                existing.volume_5min = Decimal(str(volume_24h))
                                updated = True
                            if existing.circulating_supply is None and circulating_supply is not None:
                                existing.circulating_supply = circulating_supply
                                updated = True
                            if updated:
                                to_update.append(existing)
                    else:
                        to_create.append(ShortIntervalData(
                            coin=coin,
                            timestamp=timestamp,
                            price=Decimal(str(price)),
                            volume_5min=Decimal(str(volume_24h)),
                            circulating_supply=circulating_supply
                        ))
                        existing_entries.add(key)

                if to_create:
                    ShortIntervalData.objects.bulk_create(to_create, batch_size=100)
                    print(f"✅ Inserted {len(to_create)} new entries at {timestamp}")

                for obj in to_update:
                    obj.save()
                    print(f"🔁 Updated {obj.coin.symbol} at {timestamp}")

            current += timedelta(minutes=5)

        print("✅ Backfill complete.")
