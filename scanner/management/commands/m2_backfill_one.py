from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
import requests
import time

CMC_API_KEY = "c35740fd-4f78-45b5-9350-c4afdd929432"
HEADERS = {"X-CMC_PRO_API_KEY": CMC_API_KEY}

BASE_URL = "https://pro-api.coinmarketcap.com/v2"

class Command(BaseCommand):

    help = "Backfill raw price/volume/change/high/low data into RickisMetrics from CoinMarketCap"

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 4, 23))
        end = make_aware(datetime(2025, 5, 3))

        symbols = RickisMetrics.objects.filter(
            timestamp__gte=start, timestamp__lt=end
        ).values_list("coin__symbol", flat=True).distinct()

        for symbol in symbols:
            print(f"📊 Processing {symbol}...")
            current = start

            while current < end:
                next_day = current + timedelta(days=1)

                # Fetch OHLCV data
                try:
                    ohlcv = self.get_ohlcv(symbol, current)
                except Exception as e:
                    print(f"❌ OHLCV error for {symbol} on {current.date()}: {e}")
                    current = next_day
                    continue

                # Fetch Historical Quotes
                try:
                    quotes = self.get_quotes(symbol, current)
                except Exception as e:
                    print(f"❌ Quotes error for {symbol} on {current.date()}: {e}")
                    current = next_day
                    continue

                # Map timestamp to RickisMetrics and update
                for rm in RickisMetrics.objects.filter(
                    coin__symbol=symbol,
                    timestamp__gte=current,
                    timestamp__lt=next_day,
                ):
                    ts = int(rm.timestamp.timestamp())

                    if str(ts) in quotes:
                        q = quotes[str(ts)]
                        rm.price = q.get("price")
                        rm.volume = q.get("volume_24h")
                        rm.change_1h = q.get("percent_change_1h")
                        rm.change_24h = q.get("percent_change_24h")

                    if str(ts) in ohlcv:
                        o = ohlcv[str(ts)]
                        rm.high_24h = o.get("high")
                        rm.low_24h = o.get("low")

                    rm.save()

                print(f"✅ {symbol} on {current.date()} done.")
                current = next_day
                time.sleep(1.1)  # Respect API rate limits

    def get_ohlcv(self, symbol, date):
        url = f"{BASE_URL}/cryptocurrency/ohlcv/historical"
        params = {
            "symbol": symbol,
            "time_start": date.strftime("%Y-%m-%d"),
            "time_end": (date + timedelta(days=1)).strftime("%Y-%m-%d"),
            "interval": "5m"
        }
        res = requests.get(url, headers=HEADERS, params=params)
        res.raise_for_status()
        data = res.json()["data"]["quotes"]
        return {
            int(datetime.fromisoformat(item["timestamp"]).timestamp()): {
                "high": item["quote"]["USD"]["high"],
                "low": item["quote"]["USD"]["low"]
            }
            for item in data
        }

    def get_quotes(self, symbol, date):
        url = f"{BASE_URL}/cryptocurrency/quotes/historical"
        params = {
            "symbol": symbol,
            "interval": "5m",
            "time_start": date.strftime("%Y-%m-%d"),
            "time_end": (date + timedelta(days=1)).strftime("%Y-%m-%d"),
        }
        res = requests.get(url, headers=HEADERS, params=params)
        res.raise_for_status()
        data = res.json()["data"]["quotes"]
        return {
            int(datetime.fromisoformat(item["timestamp"]).timestamp()): {
                "price": item["quote"]["USD"]["price"],
                "volume_24h": item["quote"]["USD"]["volume_24h"],
                "percent_change_1h": item["quote"]["USD"].get("percent_change_1h"),
                "percent_change_24h": item["quote"]["USD"].get("percent_change_24h"),
            }
            for item in data
        }
