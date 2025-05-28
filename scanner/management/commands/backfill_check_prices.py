from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from django.db import close_old_connections
from scanner.models import Coin, RickisMetrics
import requests
import time
import threading
from queue import Queue

API_KEY = '6520549c-03bb-41cd-86e3-30355ece87ba'
CMC_QUOTES_URL = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical'
REQUEST_INTERVAL = 2.5  # 25 req/min
MAX_WORKERS = 5         # Number of concurrent threads

class Command(BaseCommand):
    help = 'Backfill RickisMetrics prices safely with CoinMarketCap data.'

    def handle(self, *args, **kwargs):
        start_date = make_aware(datetime(2025, 3, 23))
        end_date = make_aware(datetime(2025, 4, 23))
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

        coins = {c.symbol: c for c in Coin.objects.filter(symbol__in=symbols)}

        queue = Queue()
        for symbol in symbols:
            coin = coins.get(symbol)
            if not coin:
                print(f"❌ {symbol} not found in DB.")
                continue
            current = start_date
            while current < end_date:
                queue.put((symbol, coin.id, current))
                current += timedelta(days=1)

        def worker():
            while not queue.empty():
                symbol, coin_id, date = queue.get()
                try:
                    self.process_day(symbol, coin_id, date)
                except Exception as e:
                    print(f"💥 {symbol} {date.date()} failed: {e}")
                time.sleep(REQUEST_INTERVAL)
                queue.task_done()

        threads = [threading.Thread(target=worker) for _ in range(MAX_WORKERS)]
        for t in threads: t.start()
        for t in threads: t.join()

    def process_day(self, symbol, coin_id, date):
        quotes = self.fetch_cmc_quotes(symbol, date)
        coin = Coin.objects.get(id=coin_id)
        corrected = 0

        for quote in quotes:
            try:
                ts_str = quote.get("timestamp")
                if not ts_str:
                    continue
                ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                ts = make_aware(ts.replace(second=0, microsecond=0))
                ts = ts.replace(minute=ts.minute - (ts.minute % 5))

                close_old_connections()
                metric = RickisMetrics.objects.filter(coin=coin, timestamp=ts).first()
                if not metric:
                    continue

                cmc_price = float(quote["quote"]["USD"]["price"])
                db_price = float(metric.price)
                if abs(db_price - cmc_price) / cmc_price > 0.01:
                    metric.price = cmc_price
                    metric.save()
                    corrected += 1
            except Exception as e:
                print(f"💥 {symbol} at {ts_str}: {e}")

        print(f"✅ {symbol} on {date.date()} — fixed: {corrected}")

    def fetch_cmc_quotes(self, symbol, date, retries=3):
        headers = {"X-CMC_PRO_API_KEY": API_KEY}
        params = {
            "symbol": symbol,
            "interval": "5m",
            "time_start": date.strftime("%Y-%m-%d"),
            "time_end": (date + timedelta(days=1)).strftime("%Y-%m-%d"),
            "convert": "USD"
        }

        for attempt in range(retries):
            try:
                response = requests.get(CMC_QUOTES_URL, headers=headers, params=params, timeout=10)
                if response.status_code == 429:
                    raise Exception("Rate limit hit (429)")
                response.raise_for_status()
                return response.json().get("data", {}).get("quotes", [])
            except Exception as e:
                print(f"❌ Error (attempt {attempt + 1}) fetching {symbol} on {date.date()}: {e}")
                time.sleep(5 * (attempt + 1))

        return []
