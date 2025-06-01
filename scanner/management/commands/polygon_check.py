

# nohup python manage.py polygon_check > output.log 2>&1 &
# tail -f output.log

from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from django.db import close_old_connections
from scanner.models import RickisMetrics, Coin
import requests
import time

POLYGON_URL = 'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}'
POLYGON_API_KEY = 'qq9Sptr4VfkonQimqFJEgc3oyXoaJ54L'

TRACKED_SYMBOLS = [
    "BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "DOGE", "ADA", "LINK",
    "AVAX", "XLM", "TON", "SHIB", "SUI", "HBAR", "BCH", "DOT", "LTC",
    "XMR", "UNI", "PEPE", "APT", "NEAR", "ONDO", "TAO", "ICP", "ETC",
    "RENDER", "MNT", "KAS", "CRO", "AAVE", "POL", "VET", "FIL", "ALGO",
    "ENA", "ATOM", "TIA", "ARB", "DEXE", "OP", "JUP", "MKR", "STX",
    "EOS", "WLD", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT",
    "PAXG", "CRV", "JASMY", "SAND", "GALA", "CORE", "KAIA", "LDO",
    "THETA", "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI"
]

class Command(BaseCommand):
    help = 'Fix wrong prices in RickisMetrics by fetching correct close price from Polygon.io'

    def handle(self, *args, **kwargs):
        start_date = datetime(2025, 3, 23)
        end_date = datetime(2025, 5, 23)

        coins = Coin.objects.filter(symbol__in=TRACKED_SYMBOLS)
        print(f"🚀 Found {coins.count()} tracked coins to process.")

        for coin in coins:
            polygon_symbol = f"X:{coin.symbol}-USD"
            print(f"\n🔍 Checking {coin.symbol} availability...")

            if not self.is_symbol_available(polygon_symbol):
                print(f"⚠️ Skipping {coin.symbol} — Not available on Polygon.")
                continue

            print(f"✅ {coin.symbol} is available. Processing...")

            current_date = start_date
            while current_date < end_date:
                from_date = current_date.strftime('%Y-%m-%dT00:00:00')
                to_date = current_date.strftime('%Y-%m-%dT23:55:00')

                candles = self.fetch_polygon_candles(polygon_symbol, from_date, to_date)
                if not candles:
                    print(f"⚠️ No candles fetched for {current_date.date()}")
                    current_date += timedelta(days=1)
                    continue

                for candle in candles:
                    ts = int(candle['t']) // 1000
                    ts_dt = make_aware(datetime.utcfromtimestamp(ts))
                    close_price = float(candle['c'])

                    close_old_connections()
                    metric = RickisMetrics.objects.filter(coin=coin, timestamp=ts_dt).first()
                    if not metric:
                        continue

                    db_price = float(metric.price)
                    difference = abs(db_price - close_price) / close_price

                    if difference > 0.01:
                        print(f"🔧 Fixing {ts_dt} — Old: {db_price}, New: {close_price}")
                        metric.price = close_price
                        metric.save(update_fields=['price'])

                print(f"✅ Done checking {current_date.date()} — candles: {len(candles)}")
                current_date += timedelta(days=1)
                time.sleep(0.5)  # Respect rate limits

        print("\n🎯 All tracked coins have been processed.")

    def fetch_polygon_candles(self, symbol, from_date, to_date, multiplier=5, timespan='minute'):
        url = POLYGON_URL.format(
            ticker=symbol,
            multiplier=multiplier,
            timespan=timespan,
            from_date=from_date,
            to_date=to_date
        )
        params = {
            "apiKey": POLYGON_API_KEY,
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            time.sleep(0.5)  # Sleep to respect Polygon rate limits
            data = response.json()
            return data.get('results', [])
        except Exception as e:
            print(f"❌ Error fetching candles for {symbol} from {from_date}: {e}")
            return []

    def is_symbol_available(self, symbol):
        """Check if the symbol exists on Polygon.io"""
        url = POLYGON_URL.format(
            ticker=symbol,
            multiplier=5,
            timespan='minute',
            from_date='2025-04-01T00:00:00',
            to_date='2025-04-01T23:55:00'
        )
        params = {
            "apiKey": POLYGON_API_KEY,
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if 'results' in data and len(data['results']) > 0:
                return True
            return False
        except:
            return False
