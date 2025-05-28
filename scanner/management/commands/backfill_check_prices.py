from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from scanner.models import Coin, RickisMetrics
import requests
import concurrent.futures
import time

# nohup python manage.py backfill_check_prices > output.log 2>&1 &
# tail -f output.log

API_KEY = '6520549c-03bb-41cd-86e3-30355ece87ba'
CMC_QUOTES_URL = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical'
MAX_WORKERS = 25  # CoinMarketCap allows 25 req/min

class Command(BaseCommand):
    help = 'Efficiently verify and correct RickisMetrics prices from historical CMC quotes using concurrency.'

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

        coins = Coin.objects.filter(symbol__in=symbols)
        coin_map = {coin.symbol: coin for coin in coins}
        corrections = []

        def check_day(symbol, coin, date):
            quotes = self.fetch_cmc_quotes(symbol, date)
            day_corrections = []
            for quote in quotes:
                try:
                    ts_str = quote.get("timestamp")
                    if not ts_str:
                        continue
                    ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                    ts = make_aware(ts.replace(second=0, microsecond=0))
                    ts = ts.replace(minute=ts.minute - (ts.minute % 5))
                    metric = RickisMetrics.objects.filter(coin=coin, timestamp=ts).first()
                    if not metric:
                        continue

                    cmc_price = float(quote["quote"]["USD"]["price"])
                    db_price = float(metric.price)

                    if abs(db_price - cmc_price) / cmc_price > 0.01:
                        metric.price = cmc_price
                        metric.save()
                        day_corrections.append((symbol, ts))
                except Exception as e:
                    print(f"💥 Error at {symbol} {quote.get('timestamp')}: {e}")
            return (symbol, date, day_corrections)

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for symbol in symbols:
                coin = coin_map.get(symbol)
                if not coin:
                    self.stdout.write(self.style.ERROR(f"❌ {symbol} coin not found."))
                    continue
                date = start_date
                while date < end_date:
                    futures.append(executor.submit(check_day, symbol, coin, date))
                    date += timedelta(days=1)

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    symbol, date, fixed = future.result()
                    if fixed:
                        for sym, ts in fixed:
                            print(f"✔️ Fixed {sym} at {ts}")
                            corrections.append((sym, ts))
                    print(f"✅ {symbol} on {date.date()} checked")
                except Exception as e:
                    print(f"❌ Worker error: {e}")

        print(f"\n🎯 Total Corrections: {len(corrections)}")

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
                response.raise_for_status()
                return response.json().get("data", {}).get("quotes", [])
            except Exception as e:
                print(f"❌ Error (attempt {attempt + 1}) fetching {symbol} on {date.date()}: {e}")
                time.sleep(3 * (attempt + 1))
        return []
