from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime
from scanner.models import Coin, RickisMetrics
import requests
import time

API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
CMC_QUOTES_URL = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'

class Command(BaseCommand):
    help = 'Backfill RickisMetrics with CMC quotes (change_5m, change_1h, change_24h)'

    def handle(self, *args, **kwargs):
        start_date = make_aware(datetime(2025, 3, 22))
        end_date = make_aware(datetime(2025, 4, 23))

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

            self.stdout.write(self.style.NOTICE(f"\n🚀 Backfilling CMC quotes for {symbol}"))

            metrics = list(RickisMetrics.objects.filter(
                coin=coin,
                timestamp__gte=start_date,
                timestamp__lt=end_date
            ).order_by('timestamp'))

            to_update = []

            cmc_data = self.fetch_cmc_quotes(symbol)
            if not cmc_data:
                continue

            for entry in metrics:
                if symbol in cmc_data:
                    quote = cmc_data[symbol]['quote']['USD']
                    entry.change_5m = quote.get('percent_change_5m', 0)
                    entry.change_1h = quote.get('percent_change_1h', 0)
                    entry.change_24h = quote.get('percent_change_24h', 0)
                    to_update.append(entry)

            RickisMetrics.objects.bulk_update(to_update, fields=[
                'change_5m', 'change_1h', 'change_24h'
            ], batch_size=100)

            self.stdout.write(self.style.SUCCESS(f"✅ CMC quotes backfilled for {symbol}"))

            # Sleep between API calls to avoid rate limits
            time.sleep(2)

    def fetch_cmc_quotes(self, symbol):
        headers = {"X-CMC_PRO_API_KEY": API_KEY}
        params = {"symbol": symbol, "convert": "USD"}
        try:
            response = requests.get(CMC_QUOTES_URL, headers=headers, params=params)
            response.raise_for_status()
            return response.json().get("data", {})
        except Exception as e:
            print(f"Error fetching quotes for {symbol}: {e}")
            return None
