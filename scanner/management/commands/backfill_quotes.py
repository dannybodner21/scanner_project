from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime
from scanner.models import Coin, RickisMetrics
import requests
import time

API_KEY = '6520549c-03bb-41cd-86e3-30355ece87ba'
CMC_QUOTES_URL = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'

class Command(BaseCommand):
    help = 'Backfill RickisMetrics with latest CMC quotes (volume_24h, change_1h, change_24h)'

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

            metrics = list(RickisMetrics.objects.filter(
                coin=coin,
                timestamp__gte=start_date,
                timestamp__lt=end_date
            ).order_by('timestamp'))

            to_update = []

            cmc_data = self.fetch_cmc_quotes(symbol)
            if not cmc_data or symbol not in cmc_data:
                self.stdout.write(self.style.WARNING(f"⚠️  No CMC data for {symbol}"))
                continue

            quote = cmc_data[symbol]['quote']['USD']
            volume = quote.get('volume_24h')
            change_1h = quote.get('percent_change_1h')
            change_24h = quote.get('percent_change_24h')
            price = quote.get('price')

            for entry in metrics:
                entry.price = price
                entry.volume = volume
                entry.change_1h = change_1h
                entry.change_24h = change_24h
                to_update.append(entry)

            RickisMetrics.objects.bulk_update(to_update, fields=[
                'volume', 'change_1h', 'change_24h'
            ], batch_size=100)

            self.stdout.write(self.style.SUCCESS(f"✅ Quotes backfilled for {symbol}"))

            time.sleep(2)  # Respect rate limits

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
