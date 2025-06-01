from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
from scanner.models import Coin, RickisMetrics
import requests
import time

API_KEY = '6520549c-03bb-41cd-86e3-30355ece87ba'
CMC_QUOTES_URL = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical'

class Command(BaseCommand):
    help = 'Ensure RickisMetrics exist for each 5-minute interval and backfill price, volume, change_1h, and change_24h from historical CMC quotes.'

    def handle(self, *args, **kwargs):
        start_date = make_aware(datetime(2025, 5, 9))
        end_date = make_aware(datetime(2025, 5, 24))

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
        coin_map = {coin.symbol: coin for coin in coins}

        for symbol in symbols:
            coin = coin_map.get(symbol)
            if not coin:
                self.stdout.write(self.style.ERROR(f"❌ {symbol} coin not found."))
                continue

            self.stdout.write(self.style.NOTICE(f"\n🚀 Processing {symbol}"))

            current_day = start_date
            while current_day < end_date:
                quotes = self.fetch_cmc_quotes(symbol, current_day)
                if not quotes:
                    self.stdout.write(self.style.WARNING(f"⚠️  No historical data for {symbol} on {current_day.date()}"))
                    current_day += timedelta(days=1)
                    continue

                for quote in quotes:
                    ts = datetime.strptime(quote["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ")
                    ts = ts.replace(second=0, microsecond=0)
                    ts = make_aware(ts)
                    rounded_minute = ts.minute - (ts.minute % 5)
                    ts = ts.replace(minute=rounded_minute)

                    metric, created = RickisMetrics.objects.get_or_create(
                        coin=coin,
                        timestamp=ts,
                        defaults={
                            'price': quote["quote"]["USD"].get("price"),
                            'volume': quote["quote"]["USD"].get("volume_24h"),
                            'change_1h': quote["quote"]["USD"].get("percent_change_1h"),
                            'change_24h': quote["quote"]["USD"].get("percent_change_24h")
                        }
                    )

                    updated = False
                    q = quote["quote"]["USD"]
                    if not created:
                        if metric.price in [None, 0]:
                            metric.price = q.get("price")
                            updated = True
                        if metric.volume in [None, 0]:
                            metric.volume = q.get("volume_24h")
                            updated = True
                        if metric.change_1h in [None, 0]:
                            metric.change_1h = q.get("percent_change_1h")
                            updated = True
                        if metric.change_24h in [None, 0]:
                            metric.change_24h = q.get("percent_change_24h")
                            updated = True
                        if updated:
                            metric.save()

                self.stdout.write(self.style.SUCCESS(f"✅ {symbol} on {current_day.date()} processed"))
                current_day += timedelta(days=1)
                time.sleep(1.2)

    def fetch_cmc_quotes(self, symbol, date):
        headers = {"X-CMC_PRO_API_KEY": API_KEY}
        params = {
            "symbol": symbol,
            "interval": "5m",
            "time_start": date.strftime("%Y-%m-%d"),
            "time_end": (date + timedelta(days=1)).strftime("%Y-%m-%d"),
            "convert": "USD"
        }
        try:
            response = requests.get(CMC_QUOTES_URL, headers=headers, params=params)
            response.raise_for_status()
            return response.json().get("data", {}).get("quotes", [])
        except Exception as e:
            print(f"Error fetching historical quotes for {symbol}: {e}")
            return []

'''
CHECK THAT METRICS EXIST ON THE 5 MIN TIMESTAMP

from datetime import datetime, timedelta
from django.utils.timezone import make_aware
from scanner.models import Coin, RickisMetrics

start = make_aware(datetime(2025, 3, 22))
end = make_aware(datetime(2025, 5, 23))
expected_count = int(((end - start).total_seconds()) // 300)

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
        print(f"❌ {symbol}: Coin not found")
        continue
    count = RickisMetrics.objects.filter(coin=coin, timestamp__gte=start, timestamp__lt=end).count()
    status = "✅" if count == expected_count else "⚠️"
    print(f"{status} {symbol}: {count} entries (expected {expected_count})")


'''

'''
CHECK EACH METRIC FOR FILLED OUT PRICE, VOLUME, CHANGE 1H AND CHANGE 24H

python manage.py shell -c "
from django.utils.timezone import make_aware
from datetime import datetime
from django.db.models import Q
from scanner.models import RickisMetrics

start = make_aware(datetime(2025, 3, 23))
end = make_aware(datetime(2025, 4, 1))

missing = RickisMetrics.objects.filter(
    timestamp__gte=start,
    timestamp__lt=end
).filter(
    Q(stochastic_d__isnull=True) | Q(stochastic_d=0)
).count()

print(f'Missing or invalid entries: {missing}')
"


python manage.py shell -c '
from scanner.models import RickisMetrics
from django.utils.timezone import make_aware
from datetime import datetime

TRACKED_SYMBOLS = [
    "BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "DOGE", "ADA", "LINK",
    "AVAX", "XLM", "TON", "SHIB", "SUI", "HBAR", "BCH", "DOT", "LTC",
    "XMR", "UNI", "PEPE", "APT", "NEAR", "ONDO", "TAO", "ICP", "ETC",
    "RENDER", "MNT", "KAS", "CRO", "AAVE", "POL", "VET", "FIL", "ALGO",
    "ENA", "ATOM", "TIA", "ARB", "DEXE", "OP", "JUP", "MKR", "STX",
    "EOS", "WLD", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT",
    "PAXG", "CRV", "JASMY", "SAND", "GALA", "CORE", "KAIA", "LDO",
    "THETA", "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI"
]

start = make_aware(datetime(2025, 3, 23))
end = make_aware(datetime(2025, 5, 23))

null_count = RickisMetrics.objects.filter(
    coin__symbol__in=TRACKED_SYMBOLS,
    timestamp__gte=start,
    timestamp__lt=end,
    ema_12__isnull=True
).count()

zero_count = RickisMetrics.objects.filter(
    coin__symbol__in=TRACKED_SYMBOLS,
    timestamp__gte=start,
    timestamp__lt=end,
    ema_12=0
).count()

print(f"NULL change_5m: {null_count}")
print(f"ZERO change_5m: {zero_count}")
'



'''
