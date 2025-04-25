from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
from scanner.models import Coin, ShortIntervalData

class Command(BaseCommand):
    help = 'Report missing ShortIntervalData timestamps for a list of coins between March 20, 2025, and April 22, 2025.'

    def handle(self, *args, **kwargs):
        start_date = make_aware(datetime(2025, 3, 22))
        end_date = make_aware(datetime(2025, 4, 23))

        def round_to_five_minutes(dt):
            return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)

        symbols = [
            "BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LINK",
            "AVAX", "XLM", "TON", "SHIB", "SUI", "HBAR", "BCH", "DOT", "LTC", "HYPE",
            "BGB", "DAI", "PI", "XMR", "UNI", "PEPE", "OKB", "APT", "GT", "NEAR",
            "ONDO", "TAO", "ICP", "ETC", "RENDER", "MNT", "KAS", "CRO", "AAVE", "POL",
            "VET", "FIL", "TRUMP", "ALGO", "ENA", "ATOM", "TIA", "FET", "ARB",
            "DEXE", "OP", "JUP", "MKR", "STX", "EOS", "WLD",
            "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT", "PAXG",
            "CRV", "JASMY", "SAND", "GALA", "CORE", "KAIA", "LDO", "THETA",
            "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI"
        ]

        for symbol in symbols:
            try:
                coin = Coin.objects.get(symbol=symbol)
            except Coin.DoesNotExist:
                self.stdout.write(f"❌ {symbol} coin not found in the database.")
                continue

            self.stdout.write(f"\n🔍 Checking data for: {coin.symbol}")
            current = start_date

            while current < end_date:
                next_day = current + timedelta(days=1)

                expected_times = []
                t = current
                while t < next_day:
                    expected_times.append(t)
                    t += timedelta(minutes=5)

                existing_times = set(
                    ShortIntervalData.objects.filter(
                        coin=coin,
                        timestamp__gte=current,
                        timestamp__lt=next_day
                    ).values_list("timestamp", flat=True)
                )

                missing_times = [ts for ts in expected_times if ts not in existing_times]

                if missing_times:
                    self.stdout.write(f"\n❗ {current.date()} - Missing {len(missing_times)} entries:")
                    for ts in missing_times:
                        self.stdout.write(f"   - {ts.strftime('%Y-%m-%d %H:%M')}")

                current = next_day

        self.stdout.write("\n✅ Missing data report complete.")
