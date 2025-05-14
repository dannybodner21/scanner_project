from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from scanner.models import RickisMetrics, Coin
import requests
import time

CMC_API_KEY = '6520549c-03bb-41cd-86e3-30355ece87ba'
HEADERS = {"Accepts": "application/json", "X-CMC_PRO_API_KEY": CMC_API_KEY}
CMC_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"

RATE_LIMIT_CALLS_PER_MIN = 25
SECONDS_BETWEEN_CALLS = 60.0 / RATE_LIMIT_CALLS_PER_MIN

class Command(BaseCommand):
    help = 'Fill missing volume, high, low, open, and close in RickisMetrics between March 22 and May 12'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 4, 20))
        interval = timedelta(minutes=5)

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

        total_updates = 0

        for symbol in symbols:
            try:
                coin = Coin.objects.get(symbol=symbol)
            except Coin.DoesNotExist:
                print(f"❌ Coin {symbol} not found.")
                continue

            print(f"🔍 Checking {symbol}")
            metrics = RickisMetrics.objects.filter(coin=coin, timestamp__gte=start, timestamp__lte=end)

            for metric in metrics:
                needs_update = any([
                    not metric.volume or metric.volume == 0,
                    not metric.high_24h or metric.high_24h == 0,
                    not metric.low_24h or metric.low_24h == 0,
                    not metric.open or metric.open == 0,
                    not metric.close or metric.close == 0,
                ])

                if not needs_update:
                    continue

                ts = int(metric.timestamp.timestamp())
                params = {
                    "symbol": coin.symbol,
                    "time_start": ts - 300,
                    "time_end": ts + 300,
                    "interval": "5m",
                    "convert": "USD"
                }

                try:
                    response = requests.get(CMC_URL, headers=HEADERS, params=params)
                    data = response.json()
                    quotes = data.get("data", {}).get("quotes", [])

                    if not quotes:
                        print(f"⚠️ No data for {symbol} at {metric.timestamp}")
                        continue

                    # Use closest quote
                    closest = min(
                        quotes,
                        key=lambda q: abs(datetime.strptime(q["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ") - metric.timestamp.replace(tzinfo=None))
                    )
                    quote = closest["quote"]["USD"]

                    metric.volume = metric.volume or quote.get("volume", 0)
                    metric.high_24h = metric.high_24h or quote.get("high", 0)
                    metric.low_24h = metric.low_24h or quote.get("low", 0)
                    metric.open = metric.open or quote.get("open", 0)
                    metric.close = metric.close or quote.get("close", 0)

                    metric.save()
                    total_updates += 1
                    print(f"✅ Updated {symbol} at {metric.timestamp}")

                except Exception as e:
                    print(f"❌ Failed {symbol} at {metric.timestamp}: {e}")

                time.sleep(SECONDS_BETWEEN_CALLS)

        print(f"🎉 Done. {total_updates} metrics updated.")
