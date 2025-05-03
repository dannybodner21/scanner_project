from django.core.management.base import BaseCommand
from scanner.models import Coin, RickisMetrics
from django.utils.timezone import make_aware
from datetime import datetime, timedelta

class Command(BaseCommand):
    help = "Ensure RickisMetrics exists for tracked coins at every 5-minute interval between Mar 22 and May 2"

    def handle(self, *args, **kwargs):
        rickisSymbols = [
            "BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LINK",
            "AVAX", "XLM", "TON", "SHIB", "SUI", "HBAR", "BCH", "DOT", "LTC",
            "XMR", "UNI", "PEPE", "APT", "NEAR", "ONDO", "TAO", "ICP", "ETC",
            "RENDER", "MNT", "KAS", "CRO", "AAVE", "POL", "VET", "FIL", "ALGO",
            "ENA", "ATOM", "TIA", "ARB", "DEXE", "OP", "JUP", "MKR", "STX",
            "EOS", "WLD", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT",
            "PAXG", "CRV", "JASMY", "SAND", "GALA", "CORE", "KAIA", "LDO",
            "THETA", "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI"
        ]

        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 5, 3))  # exclusive upper bound

        def round_to_5min(dt):
            return dt.replace(second=0, microsecond=0, minute=(dt.minute // 5) * 5)

        coins = Coin.objects.filter(symbol__in=rickisSymbols)
        coin_map = {c.symbol: c for c in coins}

        total_created = 0
        current_time = start

        while current_time < end:
            rounded_time = round_to_5min(current_time)
            for symbol in rickisSymbols:
                coin = coin_map.get(symbol)
                if not coin:
                    continue

                exists = RickisMetrics.objects.filter(coin=coin, timestamp=rounded_time).exists()
                if not exists:
                    RickisMetrics.objects.create(
                        coin=coin,
                        timestamp=rounded_time,
                        price=0,
                        high_24h=0,
                        low_24h=0,
                        change_5m=None,
                        change_1h=None,
                        change_24h=None,
                        volume=0,
                        avg_volume_1h=0
                    )
                    total_created += 1
            current_time += timedelta(minutes=5)

        print(f"✅ Created {total_created} missing RickisMetrics entries.")
