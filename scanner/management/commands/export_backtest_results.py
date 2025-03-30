import csv
import os
from django.core.management.base import BaseCommand
from django.conf import settings
from scanner.models import BacktestResult

class Command(BaseCommand):
    help = "Export BacktestResults with metrics to CSV for BigQuery"

    def handle(self, *args, **kwargs):
        output_path = os.path.join(settings.BASE_DIR, "backtest_export.csv")
        written = 0
        skipped = 0

        with open(output_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            header = [
                "coin", "timestamp", "entry_price", "exit_price", "success", "confidence",
                "price_change_5min", "price_change_10min", "price_change_1hr",
                "price_change_24hr", "price_change_7d",
                "five_min_relative_volume", "rolling_relative_volume", "twenty_min_relative_volume",
                "volume_24h"
            ]
            writer.writerow(header)

            for result in BacktestResult.objects.select_related("entry_metrics", "coin").iterator():
                m = result.entry_metrics
                if not m:
                    skipped += 1
                    continue
                try:
                    row = [
                        result.coin.symbol,
                        result.timestamp,
                        float(result.entry_price),
                        float(result.exit_price) if result.exit_price else "",
                        int(result.success),
                        result.confidence,
                        m.price_change_5min,
                        m.price_change_10min,
                        m.price_change_1hr,
                        m.price_change_24hr,
                        m.price_change_7d,
                        m.five_min_relative_volume,
                        m.rolling_relative_volume,
                        m.twenty_min_relative_volume,
                        m.volume_24h
                    ]
                    writer.writerow(row)
                    written += 1

                    if written % 10000 == 0:
                        self.stdout.write(f"✅ {written} rows written...")

                except Exception as e:
                    self.stderr.write(f"⚠️ Failed row: {e}")
                    skipped += 1

        self.stdout.write(f"✅ Finished: {written} written | ⏭️ {skipped} skipped")
        self.stdout.write(f"📁 File saved to: {output_path}")
