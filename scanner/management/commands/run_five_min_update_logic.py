from django.core.management.base import BaseCommand
from scanner.models import Coin, CoinAPIPrice, ModelTrade, RealTrade
from django.utils.timezone import make_aware, is_naive, now
from datetime import datetime, timedelta, timezone
import requests
from decimal import Decimal
import time
from django.utils import timezone



FINNHUB_API_KEY = 'cuf7nohr01qno7m552hgcuf7nohr01qno7m552i0'

def get_chart_patterns_for_coin(symbol, finnhub_api_key):
    url = "https://finnhub.io/api/v1/scan/pattern"
    patterns = {}
    resolutions = [5, 15, 60]

    for res in resolutions:
        params = {
            "symbol": symbol,
            "resolution": res,
            "token": finnhub_api_key
        }
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            if data.get('points'):
                latest = data['points'][0]

                status = latest.status or ""
                patterntype = latest.patterntype or "no pattern"
                patternname = latest.patternname or "no name"
                target = latest.profit1 or "no target"

                chart_string = f'{status} {patterntype} {patternname} | target: {target}'

                patterns[f'pattern_{res}m'] = chart_string
            else:
                patterns[f'pattern_{res}m'] = 'No Pattern'
        except Exception as e:
            print(f"❌ Error fetching pattern for {symbol} at {res}m: {e}")
            patterns[f'pattern_{res}m'] = 'No Pattern'

    return patterns


COINAPI_KEY = "01293e2a-dcf1-4e81-8310-c6aa9d0cb743"
COINAPI_SYMBOL_MAP = {
    "BTCUSDT": "BINANCE_SPOT_BTC_USDT",
    "ETHUSDT": "BINANCE_SPOT_ETH_USDT",
    "XRPUSDT": "BINANCE_SPOT_XRP_USDT",
    "LTCUSDT": "BINANCE_SPOT_LTC_USDT",
    "SOLUSDT": "BINANCE_SPOT_SOL_USDT",
    "DOGEUSDT": "BINANCE_SPOT_DOGE_USDT",
    "LINKUSDT": "BINANCE_SPOT_LINK_USDT",
    "DOTUSDT": "BINANCE_SPOT_DOT_USDT",
    "SHIBUSDT": "BINANCE_SPOT_SHIB_USDT",
    "ADAUSDT": "BINANCE_SPOT_ADA_USDT",
    "UNIUSDT": "BINANCE_SPOT_UNI_USDT",
    "AVAXUSDT": "BINANCE_SPOT_AVAX_USDT",
    "XLMUSDT": "BINANCE_SPOT_XLM_USDT",
}

LATEST_PRICE_BASE_URL = "https://rest.coinapi.io/v1/quotes"
OHLCV_BASE_URL = "https://rest.coinapi.io/v1/ohlcv"


def get_last_closed_candle_time():
    now_utc = datetime.utcnow().replace(second=0, microsecond=0)
    closed = now_utc - timedelta(minutes=1)
    aligned = closed - timedelta(minutes=closed.minute % 5)
    return aligned.replace(tzinfo=timezone.utc), (aligned + timedelta(minutes=5)).replace(tzinfo=timezone.utc)



def run_five_min_update_logic():

    print(f"\n⏱️ Start: {datetime.utcnow()}")

    headers = {"X-CoinAPI-Key": COINAPI_KEY}

    '''

    coins = Coin.objects.all()
    headers = {"X-CoinAPI-Key": COINAPI_KEY}
    ts_start, ts_end = get_last_closed_candle_time()

    # STEP 1: Update CoinAPIPrice
    for coin in coins:
        symbol = coin.symbol + "USDT"
        coinapi_symbol = COINAPI_SYMBOL_MAP.get(symbol)
        if not coinapi_symbol:
            continue

        url = (
            f"{OHLCV_BASE_URL}/{coinapi_symbol}/history?"
            f"period_id=5MIN"
            f"&time_start={ts_start.isoformat()}"
            f"&time_end={ts_end.isoformat()}"
            f"&limit=1"
        )

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not data:
                print(f"⚠️ No candle returned for {symbol}")
                continue

            candle = data[0]
            timestamp = datetime.fromisoformat(candle["time_period_start"].replace("Z", "+00:00"))

            CoinAPIPrice.objects.update_or_create(
                coin=coin,
                timestamp=timestamp,
                defaults={
                    "open": Decimal(str(candle["price_open"])),
                    "high": Decimal(str(candle["price_high"])),
                    "low": Decimal(str(candle["price_low"])),
                    "close": Decimal(str(candle["price_close"])),
                    "volume": Decimal(str(candle["volume_traded"]))
                }
            )

        except Exception as e:
            print(f"❌ Error updating CoinAPIPrice for {symbol}: {e}")

    '''

    # STEP 2: Evaluate all open trades
    open_trades = ModelTrade.objects.filter(exit_timestamp__isnull=True)

    for trade in open_trades:

        try:
            price_entry = float(trade.entry_price)
            symbol = trade.coin.symbol + "USDT"
            coinapi_symbol = COINAPI_SYMBOL_MAP.get(symbol)
            if not coinapi_symbol:
                continue

            latest_url = f"{LATEST_PRICE_BASE_URL}/{coinapi_symbol}/latest"
            response = requests.get(latest_url, headers=headers, timeout=10)
            response.raise_for_status()
            quotes = response.json()

            if not quotes or "bid_price" not in quotes[0] or quotes[0]["bid_price"] is None:
                print(f"⚠️ Missing bid_price for {symbol}, falling back to latest close")

                try:
                    ohlcv_url = (
                        f"{OHLCV_BASE_URL}/{coinapi_symbol}/history?"
                        f"period_id=5MIN"
                        f"&time_start={ts_start.isoformat()}"
                        f"&time_end={ts_end.isoformat()}"
                        f"&limit=1"
                    )
                    ohlcv_resp = requests.get(ohlcv_url, headers=headers, timeout=10)
                    ohlcv_resp.raise_for_status()
                    ohlcv_data = ohlcv_resp.json()

                    if not ohlcv_data:
                        print(f"⚠️ No OHLCV fallback data for {symbol}, skipping")
                        continue

                    price_now = float(ohlcv_data[0]["price_close"])

                except Exception as e:
                    print(f"❌ OHLCV fallback failed for {symbol}: {e}")
                    continue

            else:
                price_now = float(quotes[0]["bid_price"])


            result = True

            if trade.trade_type == "long":
                if price_now >= price_entry * 1.04:
                    status = "💰 TAKE PROFIT"
                elif price_now <= price_entry * 0.98:
                    status = "🛑 STOP LOSS"
                    result = False
                else:
                    continue

            else:
                if price_now <= price_entry * 0.96:
                    status = "💰 TAKE PROFIT"
                elif price_now >= price_entry * 1.02:
                    status = "🛑 STOP LOSS"
                    result = False
                else:
                    continue

            trade.exit_price = price_now
            trade.exit_timestamp = now()
            trade.result = result
            trade.save()

            print(f"{status} | {trade.trade_type.upper()} {trade.coin.symbol} @ {price_now:.6f}")

        except Exception as e:
            print(f"❌ Error evaluating trade for {trade.coin.symbol}: {e}")

    print("\n✅ Five minute update complete.")


class Command(BaseCommand):
    help = 'Run the five minute update logic'

    def handle(self, *args, **kwargs):
        run_five_min_update_logic()
