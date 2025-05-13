from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from scanner.models import RickisMetrics, Coin
import requests
from collections import defaultdict

CMC_API_KEY = '6520549c-03bb-41cd-86e3-30355ece87ba'
HEADERS = {"Accepts": "application/json", "X-CMC_PRO_API_KEY": CMC_API_KEY}
CMC_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"

class Command(BaseCommand):
    help = 'Efficiently fill missing RickisMetrics entries from March 22 to April 20'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 4, 20))
        end = make_aware(datetime(2025, 5, 12))
        interval = timedelta(days=1)

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

        coins = Coin.objects.filter(symbol__in=symbols)
        symbol_map = {coin.symbol: coin for coin in coins}
        total_created = 0

        for symbol in symbols:
            coin = symbol_map.get(symbol)
            if not coin:
                print(f"⚠️ Missing Coin entry for symbol: {symbol}")
                continue

            print(f"🔍 {symbol}: Checking days for missing metrics...")
            day = start

            while day <= end:
                day_start = int(day.timestamp())
                day_end = int((day + timedelta(days=1)).timestamp())
                params = {
                    "symbol": symbol,
                    "time_start": day_start,
                    "time_end": day_end,
                    "interval": "5m",
                    "convert": "USD"
                }

                try:
                    response = requests.get(CMC_URL, headers=HEADERS, params=params)
                    response.raise_for_status()
                    quotes = response.json()["data"]["quotes"]

                    new_metrics = []
                    for quote in quotes:
                        timestamp = make_aware(datetime.strptime(quote["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ"))
                        exists = RickisMetrics.objects.filter(coin=coin, timestamp=timestamp).exists()

                        if not exists:
                            price = quote["quote"]["USD"].get("price")
                            if price:
                                new_metrics.append(RickisMetrics(
                                    coin=coin,
                                    timestamp=timestamp,
                                    price=price,
                                    volume=quote["quote"]["USD"].get("volume_24h", 0),
                                    high_24h=quote["quote"]["USD"].get("high", 0),
                                    low_24h=quote["quote"]["USD"].get("low", 0),
                                    open=quote["quote"]["USD"].get("open", 0),
                                    close=quote["quote"]["USD"].get("close", 0),
                                ))

                    if new_metrics:
                        RickisMetrics.objects.bulk_create(new_metrics)
                        total_created += len(new_metrics)
                        print(f"✅ {symbol} on {day.date()}: {len(new_metrics)} entries added")
                    else:
                        print(f"✅ {symbol} on {day.date()}: all entries exist")

                except Exception as e:
                    print(f"❌ Failed {symbol} on {day.date()}: {e}")

                day += interval

        print(f"🎉 Done. Total new RickisMetrics entries created: {total_created}")
