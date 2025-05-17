from datetime import datetime, timedelta
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics, Coin
import requests
import time

CMC_API_KEY = '6520549c-03bb-41cd-86e3-30355ece87ba'
HEADERS = {"Accepts": "application/json", "X-CMC_PRO_API_KEY": CMC_API_KEY}
CMC_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"

RATE_LIMIT_CALLS_PER_MIN = 25
SECONDS_BETWEEN_CALLS = 60.0 / RATE_LIMIT_CALLS_PER_MIN

class Command(BaseCommand):
    help = 'Fill missing change_1h and change_24h using CMC historical data (no local calculation)'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 4, 22))
        end = make_aware(datetime(2025, 5, 13))
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

        for symbol in symbols:
            try:
                coin = Coin.objects.get(symbol=symbol)
            except Coin.DoesNotExist:
                print(f"❌ Coin {symbol} not found")
                continue

            print(f"🔍 Processing {symbol}")
            timestamp = start

            while timestamp <= end:
                try:
                    metric = RickisMetrics.objects.get(coin=coin, timestamp=timestamp)

                    if metric.change_1h not in [None, 0] or metric.change_24h not in [None, 0]:
                        timestamp += interval
                        continue

                    unix_ts = int(timestamp.timestamp())
                    params = {
                        "symbol": symbol,
                        "time_start": unix_ts - 300,
                        "time_end": unix_ts + 300,
                        "interval": "5m",
                        "convert": "USD"
                    }

                    response = requests.get(CMC_URL, headers=HEADERS, params=params)
                    data = response.json()
                    quotes = data.get("data", {}).get("quotes", [])

                    if not quotes:
                        print(f"⚠️ No data for {symbol} at {timestamp}")
                        timestamp += interval
                        continue

                    quote = quotes[0]["quote"]["USD"]
                    change_1h = quote.get("percent_change_1h")
                    change_24h = quote.get("percent_change_24h")

                    if change_1h is not None:
                        metric.change_1h = change_1h
                    if change_24h is not None:
                        metric.change_24h = change_24h

                    metric.save()
                    print(f"✅ {symbol} @ {timestamp} — 1h: {change_1h}, 24h: {change_24h}")

                except RickisMetrics.DoesNotExist:
                    print(f"⏩ No metric for {symbol} at {timestamp}")
                except Exception as e:
                    print(f"❌ Error for {symbol} at {timestamp}: {e}")

                timestamp += interval
                time.sleep(SECONDS_BETWEEN_CALLS)

        print("🎉 Done updating change_1h and change_24h")
