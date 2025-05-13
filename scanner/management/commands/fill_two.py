from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from scanner.models import RickisMetrics, Coin
import requests
import time

CMC_API_KEY = '6520549c-03bb-41cd-86e3-30355ece87ba'
HEADERS = {"Accepts": "application/json", "X-CMC_PRO_API_KEY": CMC_API_KEY}
CMC_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"

RATE_LIMIT_CALLS_PER_MIN = 25
SECONDS_BETWEEN_CALLS = 60.0 / RATE_LIMIT_CALLS_PER_MIN

class Command(BaseCommand):
    help = 'Ensure complete RickisMetrics entries every 5 minutes from March 22 to May 12'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 4, 20))
        end = make_aware(datetime(2025, 5, 12))
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

        total_filled = 0

        for symbol in symbols:
            try:
                coin = Coin.objects.get(symbol=symbol)
            except Coin.DoesNotExist:
                print(f"❌ Coin {symbol} not found.")
                continue

            print(f"🔍 Checking {symbol}")
            current = start
            timestamps_to_fill = []

            while current <= end:
                if not RickisMetrics.objects.filter(coin=coin, timestamp=current).exists():
                    timestamps_to_fill.append(current)
                current += interval

            # Group timestamps by day for efficiency
            grouped_by_day = {}
            for ts in timestamps_to_fill:
                day = ts.date()
                grouped_by_day.setdefault(day, []).append(ts)

            for day, timestamps in grouped_by_day.items():
                unix_start = int(make_aware(datetime.combine(day, datetime.min.time())).timestamp())
                unix_end = int((make_aware(datetime.combine(day, datetime.max.time()))).timestamp())
                params = {
                    "symbol": symbol,
                    "time_start": unix_start,
                    "time_end": unix_end,
                    "interval": "5m",
                    "convert": "USD"
                }

                try:
                    response = requests.get(CMC_URL, headers=HEADERS, params=params)
                    data = response.json()
                    quotes = data.get("data", {}).get("quotes", [])

                    for ts in timestamps:
                        closest = min(
                            quotes,
                            key=lambda q: abs(make_aware(datetime.strptime(q["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ")) - ts)
                        )
                        quote = closest["quote"]["USD"]
                        RickisMetrics.objects.create(
                            coin=coin,
                            timestamp=ts,
                            price=quote["price"],
                            volume=quote.get("volume_24h", 0),
                            high_24h=quote.get("high", 0),
                            low_24h=quote.get("low", 0),
                            open=quote.get("open", 0),
                            close=quote.get("close", 0),
                        )
                        total_filled += 1
                        print(f"✅ Filled {symbol} at {ts}")

                except Exception as e:
                    print(f"❌ Error retrieving quotes for {symbol} on {day}: {e}")

                time.sleep(SECONDS_BETWEEN_CALLS)

        print(f"🎉 Completed. {total_filled} entries filled.")
