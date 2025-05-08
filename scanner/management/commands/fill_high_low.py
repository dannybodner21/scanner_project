from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from scanner.models import RickisMetrics, Coin
import requests
import time

CMC_API_KEY = "7dd5dd98-35d0-475d-9338-407631033cd9"
BASE_URL = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/historical"

rickis_symbols = [
    "BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LINK",
    "AVAX", "XLM", "TON", "SHIB", "SUI", "HBAR", "BCH", "DOT", "LTC",
    "XMR", "UNI", "PEPE", "APT", "NEAR", "ONDO", "TAO", "ICP", "ETC",
    "RENDER", "MNT", "KAS", "CRO", "AAVE", "POL", "VET", "FIL", "ALGO",
    "ENA", "ATOM", "TIA", "ARB", "DEXE", "OP", "JUP", "MKR", "STX",
    "EOS", "WLD", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT",
    "PAXG", "CRV", "JASMY", "SAND", "GALA", "CORE", "KAIA", "LDO",
    "THETA", "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI"
]

class Command(BaseCommand):
    help = "Backfill missing OHLCV fields in RickisMetrics"

    def handle(self, *args, **kwargs):
        coins = Coin.objects.filter(symbol__in=rickis_symbols)
        coin_map = {coin.symbol: coin for coin in coins}

        start_date = datetime(2025, 3, 22)
        end_date = datetime(2025, 4, 20)

        for symbol in rickis_symbols:
            coin = coin_map.get(symbol)
            if not coin:
                print(f"❌ Coin not found: {symbol}")
                continue

            print(f"🔄 Fetching: {symbol}")
            date = start_date

            while date <= end_date:
                time_str = date.strftime("%Y-%m-%d")
                try:
                    url = f"{BASE_URL}?symbol={symbol}&time_start={time_str}&time_end={time_str}&interval=5m"
                    headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
                    res = requests.get(url, headers=headers)
                    res.raise_for_status()

                    candles = res.json().get("data", {}).get("quotes", [])
                    if not candles:
                        print(f"⚠️ No data for {symbol} on {time_str}")
                        date += timedelta(days=1)
                        continue

                    for candle in candles:
                        ts = make_aware(datetime.strptime(candle["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ"))
                        rm = RickisMetrics.objects.filter(coin=coin, timestamp=ts).first()
                        if not rm:
                            continue

                        modified = False
                        ohlcv = candle["quote"]["USD"]
                        if rm.open is None:
                            rm.open = ohlcv["open"]
                            modified = True
                        if rm.high_24h is None:
                            rm.high_24h = ohlcv["high"]
                            modified = True
                        if rm.low_24h is None:
                            rm.low_24h = ohlcv["low"]
                            modified = True
                        if rm.close is None:
                            rm.close = ohlcv["close"]
                            modified = True
                        if modified:
                            rm.save()

                    print(f"✅ {symbol} done for {time_str}")

                except Exception as e:
                    print(f"❌ Error for {symbol} on {time_str}: {e}")

                date += timedelta(days=1)
                time.sleep(1)  # respect API limits
