from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from scanner.models import RickisMetrics, Coin
import requests
import time

CMC_API_KEY = '6520549c-03bb-41cd-86e3-30355ece87ba'
HEADERS = {"Accepts": "application/json", "X-CMC_PRO_API_KEY": CMC_API_KEY}
CMC_URL = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/historical"

class Command(BaseCommand):
    help = 'Verify and correct historical prices for RickisMetrics from March 22 to May 23, 2025'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 5, 24))

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
                ts_str = current.isoformat()
                metrics = RickisMetrics.objects.filter(coin=coin, timestamp=current).first()

                if not metrics:
                    current += timedelta(minutes=5)
                    continue

                try:
                    response = requests.get(
                        CMC_URL,
                        headers=HEADERS,
                        params={
                            "symbol": coin.symbol,
                            "time_start": ts_str,
                            "time_end": ts_str,
                            "interval": "5m",
                            "convert": "USD"
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    quote = data["data"]["quotes"][0]["quote"]["USD"]
                    cmc_price = float(quote["close"])

                    db_price = float(metrics.price)
                    if abs(db_price - cmc_price) / cmc_price > 0.01:  # More than 1% off
                        print(f"❌ {coin.symbol} at {current} - DB: {db_price}, CMC: {cmc_price}")
                        metrics.price = cmc_price
                        metrics.save()
                        errors.append((coin.symbol, current))

                except Exception as e:
                    print(f"❌ API Error for {coin.symbol} at {current}: {e}")

                current += timedelta(minutes=5)
                time.sleep(2)  # Rate limiting

        print("\n✅ Finished checking prices.")
        print(f"Total corrections made: {len(errors)}")
        for sym, ts in errors:
            print(f"Corrected {sym} at {ts}")
