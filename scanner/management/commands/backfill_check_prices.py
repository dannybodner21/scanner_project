from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from django.db import close_old_connections
from scanner.models import Coin, RickisMetrics
import requests
import time
import threading
from queue import Queue
from collections import deque

# nohup python manage.py backfill_check_prices > output.log 2>&1 &
# tail -f output.log

API_KEY = '6520549c-03bb-41cd-86e3-30355ece87ba'
CMC_QUOTES_URL = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical'
MAX_WORKERS = 5
REQUEST_SPACING = 3  # Seconds between any 2 global requests
MAX_PER_MIN = 20       # CMC rate limit

request_lock = threading.Lock()
request_times = deque()

def wait_for_request_slot():
    while True:
        with request_lock:
            now = time.time()
            while request_times and now - request_times[0] > 60:
                request_times.popleft()
            if len(request_times) < MAX_PER_MIN:
                request_times.append(now)
                return
        time.sleep(0.5)

class Command(BaseCommand):
    help = 'Backfill RickisMetrics prices safely with CoinMarketCap data.'

    def handle(self, *args, **kwargs):
        start_date = make_aware(datetime(2025, 4, 18))
        end_date = make_aware(datetime(2025, 5, 23))
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

    def fetch_cmc_quotes(self, symbol, date, retries=6):
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
            response = requests.get(CMC_QUOTES_URL, headers=headers, params=params, timeout=20)
            if response.status_code == 429:
                print(f"❌ 429 rate limit hit for {symbol} on {date.date()} (attempt {attempt + 1})")
                time.sleep(15)
                continue
            if response.status_code >= 500:
                print(f"❌ {response.status_code} server error for {symbol} on {date.date()} (attempt {attempt + 1})")
                time.sleep(10 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json().get("data", {}).get("quotes", [])
        except Exception as e:
            print(f"❌ Error (attempt {attempt + 1}) fetching {symbol} on {date.date()}: {e}")
            time.sleep(8 * (attempt + 1))

    print(f"🚫 Failed to fetch {symbol} on {date.date()} after {retries} attempts.")
    return []
