from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from scanner.models import RickisMetrics, Coin
import requests
import time

CMC_API_KEY = '6520549c-03bb-41cd-86e3-30355ece87ba'
HEADERS = {"Accepts": "application/json", "X-CMC_PRO_API_KEY": CMC_API_KEY}
CMC_URL = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/historical"

# nohup python manage.py backfill_check_prices > output.log 2>&1 &
# tail -f output.log

class Command(BaseCommand):
    help = 'Verify and correct historical prices for RickisMetrics from March 22 to May 23, 2025'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 5, 23, 23, 55))  # Inclusive to 5-min marks on May 23

        coins = Coin.objects.filter(symbol__in=[
            "BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LINK",
            "AVAX", "XLM", "TON", "SHIB", "SUI", "HBAR", "BCH", "DOT", "LTC",
            "XMR", "UNI", "PEPE", "APT", "NEAR", "ONDO", "TAO", "ICP", "ETC",
            "RENDER", "MNT", "KAS", "CRO", "AAVE", "POL", "VET", "FIL", "ALGO",
            "ENA", "ATOM", "TIA", "ARB", "DEXE", "OP", "JUP", "MKR", "STX",
            "EOS", "WLD", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT",
            "PAXG", "CRV", "JASMY", "SAND", "GALA", "CORE", "KAIA", "LDO",
            "THETA", "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI"
        ])

        errors = []

        for coin in coins:
            print(f"\n🔍 Checking {coin.symbol}")
            current = start
            while current <= end:
                ts_start = current.isoformat()
                ts_end = (current + timedelta(minutes=10)).isoformat()
                metric = RickisMetrics.objects.filter(coin=coin, timestamp=current).first()

                if not metric:
                    current += timedelta(minutes=5)
                    continue

                try:
                    response = requests.get(
                        CMC_URL,
                        headers=HEADERS,
                        params={
                            "symbol": coin.symbol,
                            "time_start": ts_start,
                            "time_end": ts_end,
                            "interval": "5m",
                            "convert": "USD"
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    quotes = data.get("data", {}).get("quotes", [])

                    matched_quote = next((q for q in quotes if q.get("timestamp") == current.isoformat()), None)
                    if not matched_quote:
                        print(f"⚠️ No matching quote for {coin.symbol} at {current}")
                        current += timedelta(minutes=5)
                        continue

                    cmc_price = float(matched_quote["quote"]["USD"]["close"])
                    db_price = float(metric.price)

                    if abs(db_price - cmc_price) / cmc_price > 0.01:
                        print(f"❌ {coin.symbol} at {current} - DB: {db_price}, CMC: {cmc_price}")
                        metric.price = cmc_price
                        metric.save()
                        errors.append((coin.symbol, current))

                except Exception as e:
                    print(f"❌ API Error for {coin.symbol} at {current}: {e}")

                current += timedelta(minutes=5)
                time.sleep(2)  # Rate limiting

        print("\n✅ Finished checking prices.")
        print(f"Total corrections made: {len(errors)}")
        for sym, ts in errors:
            print(f"Corrected {sym} at {ts}")
