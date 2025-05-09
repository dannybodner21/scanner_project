from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
import requests
import time

CMC_API_KEY = "7dd5dd98-35d0-475d-9338-407631033cd9"


HEADERS = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
BASE_URL = "https://pro-api.coinmarketcap.com/v2"

class Command(BaseCommand):

    help = "Backfill price/volume/price change data"

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 5, 3))

        symbols = Metrics.objects.filter(
            timestamp__gte=start, timestamp__lt=end
        ).values_list("coin__symbol", flat=True).distinct()

        for symbol in symbols:

            current = start

            while current < end:
                next_day = current + timedelta(days=1)

                # get historical quotes from Coinmarketcap
                try:
                    quotes = self.get_quotes(symbol, current)

                except Exception as e:
                    print(f"error for {symbol} on {current.date()}: {e}")
                    current = next_day
                    continue

                # loop through Metrics
                for metric in Metrics.objects.filter(
                    coin__symbol=symbol,
                    timestamp__gte=current,
                    timestamp__lt=next_day,
                ):
                    ts = int(metric.timestamp.timestamp())

                    # fill in Metric
                    if str(ts) in quotes:
                        q = quotes[str(ts)]
                        metric.price = q.get("price")
                        metric.volume = q.get("volume_24h")
                        metric.change_1h = q.get("percent_change_1h")
                        metric.change_24h = q.get("percent_change_24h")
                        metric.save()

                current = next_day
                time.sleep(1.2)

    # function to fetch historical quotes from Coinmarketcap
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
        data = response.json()["data"]["quotes"]

        # return the relevant data
        return {
            int(datetime.fromisoformat(item["timestamp"]).timestamp()): {
                "price": item["quote"]["USD"]["price"],
                "volume_24h": item["quote"]["USD"]["volume_24h"],
                "percent_change_1h": item["quote"]["USD"].get("percent_change_1h"),
                "percent_change_24h": item["quote"]["USD"].get("percent_change_24h"),
            }
            for item in data
        }
