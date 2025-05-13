from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
from scanner.models import RickisMetrics, Coin
import requests
import time

CMC_API_KEY = '6520549c-03bb-41cd-86e3-30355ece87ba'
HEADERS = {"Accepts": "application/json", "X-CMC_PRO_API_KEY": CMC_API_KEY}
CMC_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"

class Command(BaseCommand):
    help = 'Fix RickisMetrics price == 0 by fetching from CMC historical quotes API'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 4, 20))
        end = make_aware(datetime(2025, 5, 12))

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
            coin = Coin.objects.filter(symbol=symbol)
            print(f"🔍 Checking: {coin.symbol}")
            metrics = RickisMetrics.objects.filter(
                coin=coin,
                timestamp__gte=start,
                timestamp__lt=end,
                price=0
            ).order_by('timestamp')

            for metric in metrics:
                timestamp_start = int(metric.timestamp.timestamp())
                timestamp_end = timestamp_start + 300
                params = {
                    "symbol": coin.symbol,
                    "time_start": timestamp_start,
                    "time_end": timestamp_end,
                    "interval": "5m",
                    "convert": "USD"
                }

                try:
                    response = requests.get(CMC_URL, headers=HEADERS, params=params)
                    data = response.json()
                    quote = data["data"]["quotes"][0]["quote"]["USD"]
                    new_price = quote["price"]

                    if new_price:
                        metric.price = new_price
                        metric.save()
                        print(f"✅ Updated {coin.symbol} at {metric.timestamp} to price {new_price}")
                    else:
                        print(f"⚠️ No price returned for {coin.symbol} at {metric.timestamp}")

                except Exception as e:
                    print(f"❌ Failed {coin.symbol} at {metric.timestamp}: {e}")

                time.sleep(1.2)

        print("🎉 Price fix completed.")
