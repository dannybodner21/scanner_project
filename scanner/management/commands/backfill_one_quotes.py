from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics, Coin
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
import requests
import time

CMC_API_KEY = "6520549c-03bb-41cd-86e3-30355ece87ba"
HEADERS = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
BASE_URL = "https://pro-api.coinmarketcap.com/v2"

class Command(BaseCommand):
    help = "Backfill RickisMetrics data (price, volume, change_1h, change_24h) using CMC historical quotes"

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 5, 10))
        end = make_aware(datetime(2025, 5, 23))

        rickisSymbols = [
            "BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LINK",
            "AVAX", "XLM", "TON", "SHIB", "SUI", "HBAR", "BCH", "DOT", "LTC",
            "XMR", "UNI", "PEPE", "APT", "NEAR", "ONDO", "TAO", "ICP", "ETC",
            "RENDER", "MNT", "KAS", "CRO", "AAVE", "POL", "VET", "FIL", "ALGO",
            "ENA", "ATOM", "TIA", "ARB", "DEXE", "OP", "JUP", "MKR", "STX",
            "EOS", "WLD", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT",
            "PAXG", "CRV", "JASMY", "SAND", "GALA", "CORE", "KAIA", "LDO",
            "THETA", "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI"
        ]

        for symbol in rickisSymbols:
            try:
                coin = Coin.objects.get(symbol=symbol)
            except Coin.DoesNotExist:
                print(f"❌ Coin not found: {symbol}")
                continue

            current = start
            while current < end:
                next_day = current + timedelta(days=1)
                print(f"🔍 Fetching {symbol} data for {current.date()}")

                try:
                    quotes = self.get_quotes(symbol, current)
                except Exception as e:
                    print(f"❌ Error fetching quotes for {symbol} on {current.date()}: {e}")
                    current = next_day
                    continue

                metrics = RickisMetrics.objects.filter(
                    coin=coin,
                    timestamp__gte=current,
                    timestamp__lt=next_day,
                )

                updated = 0
                for metric in metrics:
                    ts = int(metric.timestamp.timestamp())
                    quote = quotes.get(ts)
                    if not quote:
                        continue

                    needs_update = (
                        not metric.price or metric.price == 0 or
                        not metric.volume or metric.volume == 0 or
                        metric.change_1h in [None, 0] or
                        metric.change_24h in [None, 0]
                    )

                    if needs_update:
                        metric.price = quote.get("price")
                        metric.volume = quote.get("volume_24h")
                        metric.change_1h = quote.get("percent_change_1h")
                        metric.change_24h = quote.get("percent_change_24h")
                        metric.save()
                        updated += 1

                print(f"✅ {symbol} on {current.date()}: {updated} metrics updated")
                current = next_day
                time.sleep(1.2)

        print("🎉 Backfill complete.")

    def get_quotes(self, symbol, date):
        url = f"{BASE_URL}/cryptocurrency/quotes/historical"
        params = {
            "symbol": symbol,
            "interval": "5m",
            "time_start": date.strftime("%Y-%m-%d"),
            "time_end": (date + timedelta(days=1)).strftime("%Y-%m-%d"),
        }

        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()

        data = response.json().get("data", {}).get("quotes", [])

        return {
            int(datetime.fromisoformat(q["timestamp"]).timestamp()): {
                "price": q["quote"]["USD"]["price"],
                "volume_24h": q["quote"]["USD"]["volume_24h"],
                "percent_change_1h": q["quote"]["USD"].get("percent_change_1h"),
                "percent_change_24h": q["quote"]["USD"].get("percent_change_24h"),
            }
            for q in data
        }
