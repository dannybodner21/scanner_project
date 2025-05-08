from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from scanner.models import RickisMetrics, Coin
import requests
import time

CMC_API_KEY = "6520549c-03bb-41cd-86e3-30355ece87ba"
BASE_URL = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/historical"

rickis_symbols_one = [
    "BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LINK",
    "AVAX", "XLM", "TON", "SHIB", "SUI", "HBAR", "BCH", "DOT", "LTC",
    "XMR", "UNI", "PEPE", "APT", "NEAR", "ONDO", "TAO", "ICP", "ETC",
    "RENDER", "MNT", "KAS", "CRO","AAVE", "POL", "VET", "FIL", "ALGO",
    "ENA", "ATOM", "TIA", "ARB", "DEXE", "OP", "JUP", "MKR", "STX",
    "EOS", "WLD", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT",
    "PAXG", "CRV", "JASMY", "SAND", "GALA", "CORE", "KAIA", "LDO",
    "THETA", "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI"
]

rickis_symbols = ["FLOW"]

class Command(BaseCommand):
    help = "Backfill missing OHLCV fields in RickisMetrics"

    def handle(self, *args, **kwargs):
        coins = Coin.objects.filter(symbol__in=rickis_symbols)
        coin_map = {coin.symbol: coin for coin in coins}

        start_date = datetime(2025, 4, 12)
        end_date = datetime(2025, 4, 15)

        for symbol in rickis_symbols:
            coin = coin_map.get(symbol)
            if not coin:
                print(f"❌ Coin not found: {symbol}")
                continue

            print(f"\n🔄 Fetching OHLCV for: {symbol}")
            current_date = start_date
            request_count = 0

            while current_date <= end_date:
                api_time_start = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")
                api_time_end = current_date.strftime("%Y-%m-%d")

                url = f"{BASE_URL}?symbol={symbol}&time_start={api_time_start}&time_end={api_time_end}&interval=daily"
                headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}

                try:
                    res = requests.get(url, headers=headers)
                    res.raise_for_status()

                    request_count += 1
                    if request_count % 10 == 0:
                        time.sleep(12)

                    json_response = res.json()
                    # Extract list of entries under the symbol key
                    entries = json_response.get("data", {}).get(symbol, [])

                    # Find matching entry by CMC ID
                    record = None
                    for entry in entries:
                        if entry.get("id") == coin.cmc_id:
                            record = entry
                            break
                    if record is None and entries:
                        record = entries[0]

                    candles = record.get("quotes", []) if record else []

                    if not candles:
                        print(f"⚠️ No data for {symbol} on {api_time_end}. Full response: {json_response}")
                        current_date += timedelta(days=1)
                        continue

                    daily_ohlcv = candles[0]["quote"]["USD"]

                    day_start = make_aware(datetime.combine(current_date, datetime.min.time()))
                    day_end = day_start + timedelta(days=1)

                    rms = RickisMetrics.objects.filter(
                        coin=coin,
                        timestamp__gte=day_start,
                        timestamp__lt=day_end
                    )

                    modified_count = 0
                    for rm in rms:
                        modified = False
                        if rm.open is None:
                            rm.open = daily_ohlcv["open"]
                            modified = True
                        if rm.high_24h is None:
                            rm.high_24h = daily_ohlcv["high"]
                            modified = True
                        if rm.low_24h is None:
                            rm.low_24h = daily_ohlcv["low"]
                            modified = True
                        if rm.close is None:
                            rm.close = daily_ohlcv["close"]
                            modified = True

                        if modified:
                            rm.save()
                            modified_count += 1

                    print(f"✅ {symbol} {current_date.strftime('%Y-%m-%d')}: {modified_count} metrics updated")

                except requests.HTTPError as http_err:
                    print(f"❌ HTTP error fetching {symbol} on {api_time_end}: {http_err}, Response: {res.text}")
                except Exception as e:
                    print(f"❌ Error processing {symbol} on {api_time_end}: {e}, Response: {res.text}")

                current_date += timedelta(days=1)
                time.sleep(1.2)

        print("\n🎉 Backfill completed successfully.")
