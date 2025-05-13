from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from scanner.models import RickisMetrics, Coin
import requests
import time

CMC_API_KEY = '6520549c-03bb-41cd-86e3-30355ece87ba'
HEADERS = {"Accepts": "application/json", "X-CMC_PRO_API_KEY": CMC_API_KEY}
CMC_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"

class Command(BaseCommand):
    help = 'Ensure complete RickisMetrics entries every 5 minutes from March 22 to May 12'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 4, 20))
        interval = timedelta(minutes=5)

        coins = Coin.objects.all()
        total_filled = 0

        symbols = [
            "BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LINK",
            "AVAX", "XLM", "TON", "SHIB", "SUI", "HBAR", "BCH", "DOT", "LTC",
            "XMR", "UNI", "PEPE", "APT", "NEAR", "ONDO", "TAO", "ICP", "ETC",
            "RENDER", "MNT", "KAS", "CRO", "AAVE", "POL", "VET", "FIL", "ALGO",
            "ENA", "ATOM", "TIA", "ARB", "DEXE", "OP", "JUP", "MKR", "STX",
            "EOS", "WLD", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT",
            "PAXG", "CRV", "JASMY", "SAND", "GALA", "CORE", "KAIA", "LDO",
            "THETA", "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI"
        ] 

        for symbol in symbols:
            coin = Coin.objects.get(sybmol=symbol)
            print(f"🔍 Checking {coin.symbol}")
            timestamp = start

            while timestamp <= end:
                exists = RickisMetrics.objects.filter(coin=coin, timestamp=timestamp).exists()

                if not exists:
                    unix_ts = int(timestamp.timestamp())
                    params = {
                        "id": coin.cmc_id,
                        "time_start": unix_ts - 300,
                        "time_end": unix_ts + 300,
                        "interval": "5m",
                        "convert": "USD"
                    }

                    try:
                        response = requests.get(CMC_URL, headers=HEADERS, params=params)
                        data = response.json()

                        quote = data["data"]["quotes"][0]["quote"]["USD"]
                        price = quote["price"]

                        RickisMetrics.objects.create(
                            coin=coin,
                            timestamp=timestamp,
                            price=price,
                            volume=quote.get("volume_24h", 0),
                            high_24h=quote.get("high", 0),
                            low_24h=quote.get("low", 0),
                            open=quote.get("open", 0),
                            close=quote.get("close", 0),
                        )
                        total_filled += 1
                        print(f"✅ Filled {coin.symbol} at {timestamp}")

                    except Exception as e:
                        print(f"❌ Failed {coin.symbol} at {timestamp}: {e}")

                    time.sleep(1.1)

                timestamp += interval

        print(f"🎉 Completed. {total_filled} entries filled.")
