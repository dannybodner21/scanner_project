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
    help = 'Fill missing change_1h and change_24h using CMC historical data'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 4, 23))
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

            print(f"🔍 Checking {symbol}")
            timestamp = start

            while timestamp <= end:
                try:
                    metric = RickisMetrics.objects.get(coin=coin, timestamp=timestamp)

                    if (metric.change_1h and metric.change_1h != 0) and \
                       (metric.change_24h and metric.change_24h != 0):
                        timestamp += interval
                        continue

                    # Get target and comparison timestamps
                    ts_current = int(timestamp.timestamp())
                    ts_1h_ago = int((timestamp - timedelta(hours=1)).timestamp())
                    ts_24h_ago = int((timestamp - timedelta(hours=24)).timestamp())

                    # Fetch historical data (current, 1h ago, 24h ago)
                    prices = {}
                    for label, ts in [('current', ts_current), ('1h', ts_1h_ago), ('24h', ts_24h_ago)]:
                        params = {
                            "symbol": symbol,
                            "time_start": ts - 300,
                            "time_end": ts + 300,
                            "interval": "5m",
                            "convert": "USD"
                        }
                        res = requests.get(CMC_URL, headers=HEADERS, params=params)
                        data = res.json()
                        quotes = data.get("data", {}).get("quotes", [])
                        if quotes:
                            prices[label] = quotes[0]["quote"]["USD"]["price"]
                        else:
                            print(f"⚠️ No price found for {symbol} at {label} {datetime.utcfromtimestamp(ts)}")
                        time.sleep(SECONDS_BETWEEN_CALLS)

                    # Calculate changes
                    if "current" in prices and "1h" in prices and prices["1h"] != 0:
                        metric.change_1h = ((prices["current"] - prices["1h"]) / prices["1h"]) * 100

                    if "current" in prices and "24h" in prices and prices["24h"] != 0:
                        metric.change_24h = ((prices["current"] - prices["24h"]) / prices["24h"]) * 100

                    metric.save()
                    print(f"✅ Updated {symbol} at {timestamp}")

                except RickisMetrics.DoesNotExist:
                    print(f"⏩ No RickisMetric for {symbol} at {timestamp}")
                except Exception as e:
                    print(f"❌ Error for {symbol} at {timestamp}: {e}")

                timestamp += interval
