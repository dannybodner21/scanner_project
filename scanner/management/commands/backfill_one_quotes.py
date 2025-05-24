from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics, Coin
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
import requests
import time
from dateutil import parser
import pytz

CMC_API_KEY = "6520549c-03bb-41cd-86e3-30355ece87ba"
HEADERS = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
BASE_URL = "https://pro-api.coinmarketcap.com/v2"
UTC = pytz.UTC

class Command(BaseCommand):
    help = "Ensure every 5-minute RickisMetrics entry from May 5 to May 23, 2025 exists and has full price/volume/change data."

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 5, 5))
        end = make_aware(datetime(2025, 5, 24))

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
            coin = Coin.objects.filter(symbol=symbol).first()
            if not coin:
                self.stdout.write(f"❌ Coin not found: {symbol}")
                continue

            current = start
            while current < end:
                next_day = current + timedelta(days=1)
                self.stdout.write(f"🔍 Checking {symbol} for {current.date()}...")

                try:
                    quotes = self.get_quotes(symbol, current)
                except Exception as e:
                    self.stdout.write(f"❌ Error fetching quotes for {symbol} on {current.date()}: {e}")
                    current = next_day
                    continue

                updated = 0
                for q_ts, q_data in quotes.items():
                    quote_time = datetime.fromtimestamp(q_ts, tz=UTC)
                    rounded_minute = quote_time.minute - (quote_time.minute % 5)
                    rounded_time = quote_time.replace(minute=rounded_minute, second=0, microsecond=0)

                    metric, created = RickisMetrics.objects.get_or_create(
                        coin=coin,
                        timestamp=rounded_time,
                        defaults={
                            "price": q_data.get("price"),
                            "volume": q_data.get("volume_24h"),
                            "change_1h": q_data.get("percent_change_1h"),
                            "change_24h": q_data.get("percent_change_24h"),
                        },
                    )

                    if not created:
                        needs_update = False
                        if metric.price in [None, 0]:
                            metric.price = q_data.get("price")
                            needs_update = True
                        if metric.volume in [None, 0]:
                            metric.volume = q_data.get("volume_24h")
                            needs_update = True
                        if metric.change_1h in [None, 0]:
                            metric.change_1h = q_data.get("percent_change_1h")
                            needs_update = True
                        if metric.change_24h in [None, 0]:
                            metric.change_24h = q_data.get("percent_change_24h")
                            needs_update = True
                        if needs_update:
                            metric.save()
                            updated += 1
                    else:
                        updated += 1

                self.stdout.write(f"✅ {symbol} on {current.date()}: {updated} metrics filled or updated")
                current = next_day
                time.sleep(1.2)

        self.stdout.write("\n🎉 Full backfill complete.")

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
            int(parser.isoparse(item["timestamp"]).timestamp()): {
                "price": item["quote"]["USD"]["price"],
                "volume_24h": item["quote"]["USD"]["volume_24h"],
                "percent_change_1h": item["quote"]["USD"].get("percent_change_1h"),
                "percent_change_24h": item["quote"]["USD"].get("percent_change_24h"),
            }
            for item in data
        }
