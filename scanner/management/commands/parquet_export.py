from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
import pandas as pd

class Command(BaseCommand):
    help = 'Export RickisMetrics to Parquet with ALL columns forced to float'

    def handle(self, *args, **kwargs):
        self.stdout.write("🚀 Exporting RickisMetrics...")

        qs = RickisMetrics.objects.filter(
            long_result__isnull=False,
            short_result__isnull=False
        ).values()

        df = pd.DataFrame(list(qs))

        if df.empty:
            self.stdout.write(self.style.ERROR("❌ No data found."))
            return

        # Force all columns to float except timestamp and coin symbol
        for col in df.columns:
            if col not in ['timestamp', 'coin__symbol']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df.rename(columns={'coin__symbol': 'coin_symbol'}, inplace=True)

        df.to_parquet('/workspace/scanner/metrics_export.parquet')

        self.stdout.write(self.style.SUCCESS(f"✅ Done. {len(df)} rows exported."))
