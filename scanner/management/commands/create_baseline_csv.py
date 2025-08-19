from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice
import pandas as pd
from datetime import datetime, timezone

class Command(BaseCommand):
    help = 'Export baseline OHLCV data for all 13 coins from CoinAPIPrice model'

    def handle(self, *args, **options):
        coins = [
            'BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'LTCUSDT', 'SOLUSDT',
            'DOGEUSDT', 'LINKUSDT', 'DOTUSDT', 'SHIBUSDT', 'ADAUSDT',
            'UNIUSDT', 'AVAXUSDT', 'XLMUSDT', 'TRXUSDT', 'ATOMUSDT'
        ]

        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 8, 17, 23, 55, tzinfo=timezone.utc)

        all_data = []

        for coin in coins:
            self.stdout.write(f"🔍 Fetching {coin}...")
            qs = CoinAPIPrice.objects.filter(
                coin=coin,
                timestamp__gte=start,
                timestamp__lte=end
            ).order_by("timestamp")

            if not qs.exists():
                self.stdout.write(self.style.WARNING(f"⚠️ No data found for {coin}"))
                continue

            df = pd.DataFrame.from_records(qs.values(
                'timestamp', 'coin', 'open', 'high', 'low', 'close', 'volume'
            ))

            all_data.append(df)

        if not all_data:
            self.stdout.write(self.style.ERROR("❌ No data retrieved for any coins."))
            return

        final_df = pd.concat(all_data, ignore_index=True)
        final_df.sort_values(['timestamp', 'coin'], inplace=True)
        final_df.to_csv("baseline_ohlcv.csv", index=False)

        self.stdout.write(self.style.SUCCESS("✅ baseline_ohlcv.csv created successfully."))
