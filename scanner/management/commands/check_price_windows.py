# management/commands/check_flat_prices.py

from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from scanner.models import RickisMetrics, Coin

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

class Command(BaseCommand):
    help = 'Check for flat price windows in RickisMetrics'

    def handle(self, *args, **kwargs):
        start_date = make_aware(datetime(2025, 3, 23))
        end_date = make_aware(datetime(2025, 5, 23))

        coins = Coin.objects.filter(symbol__in=TRACKED_SYMBOLS)
        print(f"🚀 Found {coins.count()} tracked coins to check for flat prices.")

        for coin in coins:
            print(f"\n🔍 Checking {coin.symbol} for flat windows...")

            metrics = RickisMetrics.objects.filter(
                coin=coin,
                timestamp__gte=start_date,
                timestamp__lt=end_date
            ).order_by('timestamp')

            if not metrics.exists():
                print(f"⚠️ No metrics found for {coin.symbol}. Skipping.")
                continue

            last_price = None
            flat_start = None
            flat_count = 0

            for metric in metrics:
                price = float(metric.price)

                if price == last_price:
                    flat_count += 1
                else:
                    if flat_count >= 10:  # Flat for 10 or more candles (~50 min)
                        print(f"❗ Flat window for {coin.symbol}: {flat_start} to {metric.timestamp - timedelta(minutes=5)} — Price: {last_price} — Candles: {flat_count}")
                    flat_start = metric.timestamp
                    flat_count = 1
                    last_price = price

            # Handle case if last window is flat
            if flat_count >= 10:
                print(f"❗ Flat window for {coin.symbol}: {flat_start} to {metric.timestamp} — Price: {last_price} — Candles: {flat_count}")

        print("\n🎯 Flat window check completed.")
