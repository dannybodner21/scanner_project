from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
from scanner.models import RickisMetrics, Coin
import requests
import time
import pytz

CMC_API_KEY = '6520549c-03bb-41cd-86e3-30355ece87ba'
HEADERS = {"Accepts": "application/json", "X-CMC_PRO_API_KEY": CMC_API_KEY}
CMC_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"
UTC = pytz.UTC

symbols = [
    "BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LINK",
    "AVAX", "XLM", "TON", "SHIB", "SUI", "HBAR", "BCH", "DOT", "LTC",
    "XMR", "UNI", "PEPE", "APT", "NEAR", "ONDO", "TAO", "ICP", "ETC",
    "RENDER", "MNT", "KAS", "CRO", "AAVE", "POL", "VET", "FIL", "ALGO",
    "ENA", "ATOM", "TIA", "ARB", "DEXE", "OP", "JUP", "MKR", "STX",
    "EOS", "WLD", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT",
    "PAXG", "CRV", "JASMY", "SAND", "GALA", "CORE",
    "KAIA", "LDO",
    "THETA", "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI"
]

class Command(BaseCommand):
    help = 'Fix RickisMetrics price == 0 using CMC historical quotes with rounded timestamps.'

    def handle(self, *args, **kwargs):
        start_date = datetime(2025, 4, 20)
        end_date = datetime(2025, 5, 12)

        for symbol in symbols:
            coin = Coin.objects.filter(symbol=symbol).first()
            if not coin:
                print(f"❌ Coin not found: {symbol}")
                continue

            for n in range((end_date - start_date).days + 1):
                date = start_date + timedelta(days=n)
                time_start = int(datetime(date.year, date.month, date.day, 0, 0, tzinfo=UTC).timestamp())
                time_end = int(datetime(date.year, date.month, date.day, 23, 59, tzinfo=UTC).timestamp())

                params = {
                    "symbol": symbol,
                    "time_start": time_start,
                    "time_end": time_end,
                    "interval": "5m",
                    "convert": "USD"
                }

                try:
                    response = requests.get(CMC_URL, headers=HEADERS, params=params)
                    data = response.json()
                    quotes = data.get("data", {}).get("quotes", [])

                    updated = 0
                    for quote in quotes:
                        quote_time = datetime.strptime(quote["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
                        rounded_time = quote_time.replace(second=0, microsecond=0)
                        rounded_minute = rounded_time.minute - (rounded_time.minute % 5)
                        rounded_time = rounded_time.replace(minute=rounded_minute)

                        metric = RickisMetrics.objects.filter(coin=coin, timestamp=rounded_time, price=0).first()
                        if metric:
                            price = quote["quote"]["USD"]["price"]
                            metric.price = price
                            metric.save()
                            updated += 1

                    print(f"✅ {symbol} on {date.strftime('%Y-%m-%d')}: {updated} prices updated")

                except Exception as e:
                    print(f"❌ Error for {symbol} on {date.strftime('%Y-%m-%d')}: {e}")

                time.sleep(1.2)

        print("🎉 Price update complete.")
