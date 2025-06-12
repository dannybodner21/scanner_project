import csv
import os
from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice

class Command(BaseCommand):
    help = "Export CoinAPIPrice data to CSV for local processing."

    def handle(self, *args, **kwargs):
        coins = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'LTCUSDT', 'SOLUSDT', 'DOGEUSDT', 'PEPEUSDT', 'ADAUSDT', 'XLMUSDT', 'SUIUSDT', 'LINKUSDT', 'AVAXUSDT', 'DOTUSDT', 'SHIBUSDT', 'HBARUSDT', 'UNIUSDT']

        os.makedirs('exported_data', exist_ok=True)

        for coin in coins:
            filename = f"exported_data/{coin}.csv"
            self.stdout.write(f"Exporting {coin}...")

            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])

                for row in CoinAPIPrice.objects.filter(coin=coin).order_by('timestamp').iterator(chunk_size=10000):
                    writer.writerow([
                        row.timestamp.isoformat(),
                        float(row.open),
                        float(row.high),
                        float(row.low),
                        float(row.close),
                        float(row.volume)
                    ])

            self.stdout.write(f"✅ Exported {coin} to {filename}")
