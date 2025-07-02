# scanner/management/commands/fetch_ohlcv_command.py

from django.core.management.base import BaseCommand
from datetime import datetime, timedelta
import requests
import pandas as pd
from scanner.models import CoinAPIPrice


# python manage.py mimick BTCUSDT


COINAPI_KEY = "01293e2a-dcf1-4e81-8310-c6aa9d0cb743"
BASE_URL = "https://rest.coinapi.io/v1/ohlcv"
COINAPI_SYMBOL_MAP = {
    "BTCUSDT": "BINANCE_SPOT_BTC_USDT",
}

def fetch_ohlcv(coin, limit=300, end_time=None):
    coinapi_symbol = COINAPI_SYMBOL_MAP.get(coin)
    if not coinapi_symbol:
        raise ValueError(f"CoinAPI symbol mapping not found for {coin}")

    if end_time is None:
        #end_time = datetime.utcnow().replace(second=0, microsecond=0)
        end_time = datetime.utcnow().replace(second=0, microsecond=0) - timedelta(minutes=10)

    url = f"{BASE_URL}/{coinapi_symbol}/history?period_id=5MIN&time_end={end_time.isoformat()}&limit={limit}"
    headers = {"X-CoinAPI-Key": COINAPI_KEY}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame([{
        'timestamp': pd.to_datetime(c['time_period_start']),
        'open': c['price_open'],
        'high': c['price_high'],
        'low': c['price_low'],
        'close': c['price_close'],
        'volume': c['volume_traded']
    } for c in data])

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

from scanner.models import Coin
def fetch_recent_db_candles(symbol="BTC", limit=20):

    coin_obj = Coin.objects.get(symbol=symbol)
    qs = CoinAPIPrice.objects.filter(coin=coin_obj).order_by("-timestamp")[:limit]
    records = list(qs.values("timestamp", "open", "high", "low", "close", "volume"))
    df = pd.DataFrame(records)
    df.sort_values("timestamp", inplace=True)
    return df


class Command(BaseCommand):
    help = "Fetch recent OHLCV data from CoinAPI for comparison"

    def add_arguments(self, parser):
        parser.add_argument("coin", type=str, help="Coin symbol (e.g., BTCUSDT)")

    def handle(self, *args, **options):

        coin = options["coin"]
        df = fetch_ohlcv(coin)
        print(df.tail(10).to_string(index=False))

        db_symbol = coin.replace("USDT", "")
        print(f"\n🗄️ Fetched from Database ({db_symbol}):")
        df_db = fetch_recent_db_candles(db_symbol)
        print(df_db.tail(10).to_string(index=False))
