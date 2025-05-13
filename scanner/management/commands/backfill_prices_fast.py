from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
from scanner.models import RickisMetrics, Coin
import requests
import time

CMC_API_KEY = '6520549c-03bb-41cd-86e3-30355ece87ba'
HEADERS = {"Accepts": "application/json", "X-CMC_PRO_API_KEY": CMC_API_KEY}
CMC_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"

class Command(BaseCommand):
    
    help = 'Efficiently fix RickisMetrics with price == 0'

    def handle(self, *args, **kwargs):
        start_date = datetime(2025, 4, 20)
        end_date = datetime(2025, 5, 12)
        delta = timedelta(days=1)

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

        for symbol in symbols:
            coin = Coin.objects.filter(symbol=symbol).first()
            if not coin:
                print(f"❌ Missing coin: {symbol}")
                continue

            current_date = start_date
            while current_date <= end_date:
                day_start = make_aware(datetime.combine(current_date, datetime.min.time()))
                day_end = day_start + timedelta(days=1)

                metrics = RickisMetrics.objects.filter(
                    coin=coin,
                    timestamp__gte=day_start,
                    timestamp__lt=day_end,
                    price=0
                ).order_by("timestamp")

                if not metrics.exists():
                    current_date += delta
                    continue

                unix_start = int(day_start.timestamp())
                unix_end = int(day_end.timestamp())

                params = {
                    "symbol": symbol,
                    "time_start": unix_start,
                    "time_end": unix_end,
                    "interval": "5m",
                    "convert": "USD",
                }

                try:
                    response = requests.get(CMC_URL, headers=HEADERS, params=params)
                    response.raise_for_status()
                    quotes = response.json().get("data", {}).get("quotes", [])

                    if not quotes:
                        print(f"⚠️ No data for {symbol} on {current_date.date()}")
                        current_date += delta
                        continue

                    # Convert to dict for quick lookup
                    quote_map = {
                        datetime.strptime(q["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ"): q["quote"]["USD"]["price"]
                        for q in quotes
                    }

                    updated = 0
                    for metric in metrics:
                        ts = metric.timestamp.replace(second=0, microsecond=0)
                        price = quote_map.get(ts)
                        if price:
                            metric.price = price
                            metric.save()
                            updated += 1

                    print(f"✅ {symbol} on {current_date.date()}: {updated} prices updated")

                except Exception as e:
                    print(f"❌ API error for {symbol} on {current_date.date()}: {e}")

                time.sleep(1.5)
                current_date += delta

        print("🎉 Efficient price patch complete.")
