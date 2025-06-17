import csv
import os
from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice
from datetime import datetime, timezone

class Command(BaseCommand):
    help = "Export CoinAPIPrice data to CSV for out-of-sample evaluation."

    def handle(self, *args, **kwargs):
        coins = [
            'BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'LTCUSDT', 'SOLUSDT', 'DOGEUSDT',
            'PEPEUSDT', 'ADAUSDT', 'XLMUSDT', 'SUIUSDT', 'LINKUSDT', 'AVAXUSDT',
            'DOTUSDT', 'SHIBUSDT', 'HBARUSDT', 'UNIUSDT'
        ]

        start_date = datetime(2025, 6, 11, 0, 0, 0, tzinfo=timezone.utc)
        end_date = datetime(2025, 6, 13, 23, 59, 0, tzinfo=timezone.utc)

        os.makedirs('exported_data_eval', exist_ok=True)

        for coin in coins:
            filename = f"exported_data_eval/{coin}.csv"
            self.stdout.write(f"Exporting {coin}...")

            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])

                queryset = CoinAPIPrice.objects.filter(
                    coin=coin,
                    timestamp__gte=start_date,
                    timestamp__lte=end_date
                ).order_by('timestamp')

                for row in queryset.iterator(chunk_size=10000):
                    writer.writerow([
                        row.timestamp.isoformat(),
                        float(row.open),
                        float(row.high),
                        float(row.low),
                        float(row.close),
                        float(row.volume)
                    ])

            self.stdout.write(f"✅ Exported {coin} to {filename}")
