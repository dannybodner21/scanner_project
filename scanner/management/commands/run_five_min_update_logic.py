from django.core.management.base import BaseCommand
from scanner.models import Coin, CoinAPIPrice, ModelTrade
from scanner.helpers import (
    round_to_five_minutes, get_recent_prices, get_recent_volumes,
    calculate_rsi, calculate_ema_from_prices, calculate_macd,
    calculate_stochastic, calculate_support_resistance, calculate_avg_volume_1h,
    calculate_relative_volume, calculate_sma, calculate_ema, calculate_price_slope_1h,
    calculate_stddev_1h, calculate_atr_1h, calculate_price_change_five_min,
    calculate_change_since_high, calculate_change_since_low, calculate_obv,
    calculate_fib_distances, calculate_bollinger_bands, calculate_adx
)
from django.utils.timezone import make_aware, is_naive, now
from datetime import datetime
import requests
import time


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
                patternname = lastest.patternname or "no name"
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
}

BASE_URL = "https://rest.coinapi.io/v1/ohlcv"


from django.core.management.base import BaseCommand
from scanner.models import Coin, CoinAPIPrice, ModelTrade
from django.utils.timezone import now
from datetime import datetime
import requests

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
}

BASE_URL = "https://rest.coinapi.io/v1/ohlcv"


def run_five_min_update_logic():
    start = datetime.now()
    print(f"\n⏱️ Start: {start}")

    coins = Coin.objects.all()
    headers = {"X-CoinAPI-Key": COINAPI_KEY}

    # STEP 1: Update CoinAPIPrice
    for coin in coins:
        symbol = coin.symbol + "USDT"
        coinapi_symbol = COINAPI_SYMBOL_MAP.get(symbol)
        if not coinapi_symbol:
            continue

        url = f"{BASE_URL}/{coinapi_symbol}/latest?period_id=5MIN&limit=1"

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            candle = data[0]

            timestamp = datetime.fromisoformat(candle["time_period_start"].replace("Z", "+00:00"))

            CoinAPIPrice.objects.update_or_create(
                coin=coin,
                timestamp=timestamp,
                defaults={
                    "open": candle["price_open"],
                    "high": candle["price_high"],
                    "low": candle["price_low"],
                    "close": candle["price_close"],
                    "volume": candle["volume_traded"]
                }
            )
        except Exception as e:
            print(f"❌ Error updating CoinAPIPrice for {symbol}: {e}")

    # STEP 2: Evaluate all open trades
    open_trades = ModelTrade.objects.filter(exit_timestamp__isnull=True)

    for trade in open_trades:
        try:
            price_entry = float(trade.entry_price)
            latest_price = CoinAPIPrice.objects.filter(coin=trade.coin).order_by("-timestamp").first()
            if not latest_price:
                continue

            price_now = float(latest_price.close)
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
