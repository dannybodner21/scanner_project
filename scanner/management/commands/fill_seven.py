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
    help = 'Batch-fill missing change_1h and change_24h using CMC historical data efficiently'

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

            print(f"🔄 Fetching data for {symbol}")
            current = start

            while current <= end:
                # Pull 25 hours of data to compute change_1h and change_24h
                ts_start = int((current - timedelta(hours=25)).timestamp())
                ts_end = int(current.timestamp())

                params = {
                    "symbol": symbol,
                    "time_start": ts_start,
                    "time_end": ts_end,
                    "interval": "5m",
                    "convert": "USD"
                }

                try:
                    response = requests.get(CMC_URL, headers=HEADERS, params=params)
                    response.raise_for_status()
                    data = response.json().get("data", {})
                    quotes = data.get("quotes", [])

                    if not quotes:
                        print(f"⚠️ No quotes for {symbol} at {current}")
                        time.sleep(2)
                        current += timedelta(hours=24)
                        continue

                    price_map = {
                        datetime.strptime(q["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ"): q["quote"]["USD"]["price"]
                        for q in quotes
                    }

                    timestamps = sorted(price_map.keys())

                    for i in range(len(timestamps)):
                        ts = make_aware(timestamps[i])
                        if ts < start or ts > end:
                            continue

                        try:
                            metric = RickisMetrics.objects.get(coin=coin, timestamp=ts)
                        except RickisMetrics.DoesNotExist:
                            continue

                        if (metric.change_1h and metric.change_1h != 0) and \
                           (metric.change_24h and metric.change_24h != 0):
                            continue

                        price_now = price_map.get(ts)
                        price_1h_ago = price_map.get(ts - timedelta(hours=1))
                        price_24h_ago = price_map.get(ts - timedelta(hours=24))

                        if price_now and price_1h_ago and price_1h_ago != 0:
                            metric.change_1h = ((price_now - price_1h_ago) / price_1h_ago) * 100
                        if price_now and price_24h_ago and price_24h_ago != 0:
                            metric.change_24h = ((price_now - price_24h_ago) / price_24h_ago) * 100

                        metric.save()

                    print(f"✅ {symbol} — Updated data from {current} to {current + timedelta(hours=24)}")

                except Exception as e:
                    print(f"❌ Error on {symbol} @ {current}: {e}")

                time.sleep(2)
                current += timedelta(hours=24)

        print("🎉 change_1h and change_24h backfill complete.")
