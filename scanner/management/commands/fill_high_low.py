from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from scanner.models import RickisMetrics, Coin
import requests
import time

CMC_API_KEY = "6520549c-03bb-41cd-86e3-30355ece87ba"
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

            print(f"🔄 Fetching OHLCV for: {symbol}")
            date = start_date
            request_count = 0

            while date <= end_date:
                time_end = date.strftime("%Y-%m-%d")
                time_start = (date - timedelta(days=1)).strftime("%Y-%m-%d")

                try:
                    url = f"{BASE_URL}?symbol={symbol}&time_start={time_start}&time_end={time_end}&interval=daily"
                    headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
                    res = requests.get(url, headers=headers)
                    res.raise_for_status()

                    request_count += 1
                    if request_count % 10 == 0:
                        time.sleep(12)

                    candles = res.json().get("data", {}).get("quotes", [])
                    if not candles:
                        print(f"⚠️ No data for {symbol} on {time_end}")
                        date += timedelta(days=1)
                        continue

                    modified_count = 0
                    for candle in candles:
                        try:
                            ts_raw = candle["timestamp"]
                            try:
                                ts = make_aware(datetime.strptime(ts_raw, "%Y-%m-%dT%H:%M:%S.%fZ"))
                            except ValueError:
                                ts = make_aware(datetime.strptime(ts_raw, "%Y-%m-%dT%H:%M:%SZ"))

                            rm = RickisMetrics.objects.filter(coin=coin, timestamp=ts).first()
                            if not rm:
                                continue

                            ohlcv = candle["quote"]["USD"]
                            modified = False

                            if rm.open is None:
                                rm.open = ohlcv.get("open")
                                modified = True
                            if rm.high_24h is None:
                                rm.high_24h = ohlcv.get("high")
                                modified = True
                            if rm.low_24h is None:
                                rm.low_24h = ohlcv.get("low")
                                modified = True
                            if rm.close is None:
                                rm.close = ohlcv.get("close")
                                modified = True

                            if modified:
                                rm.save()
                                modified_count += 1

                        except Exception as inner_e:
                            print(f"⚠️ Error parsing candle for {symbol} on {time_end}: {inner_e}")

                    print(f"✅ {symbol} - {time_end}: {modified_count} updated")

                except Exception as e:
                    print(f"❌ Request error for {symbol} on {time_end}: {e}")

                date += timedelta(days=1)
                time.sleep(1.2)  # base rate limit safety

        print("🎉 Backfill completed.")
