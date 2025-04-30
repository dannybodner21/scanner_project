from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
import pandas as pd

class Command(BaseCommand):
    help = 'Export RickisMetrics to Parquet with full real features + label, excluding incomplete rows'

    def handle(self, *args, **kwargs):
        self.stdout.write("🚀 Exporting RickisMetrics...")

        # Define the fields you want to include
        fields = [
            'price', 'volume', 'change_1h', 'change_24h',
            'high_24h', 'low_24h', 'avg_volume_1h', 'relative_volume',
            'sma_5', 'sma_20', 'ema_12', 'ema_26', 'macd', 'macd_signal',
            'rsi', 'stochastic_k', 'stochastic_d', 'support_level',
            'resistance_level', 'stddev_1h', 'price_slope_1h', 'atr_1h',
            'short_result'
        ]

        qs = RickisMetrics.objects.filter(short_result__isnull=False).values(*fields)
        df = pd.DataFrame(list(qs))

        if df.empty:
            self.stdout.write(self.style.ERROR("❌ No data found."))
            return

        # Convert to numeric and drop rows with missing/invalid data
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)

        df.to_parquet('/workspace/scanner/metrics_export_short_cleaned.parquet')

        self.stdout.write(self.style.SUCCESS(f"✅ Done. {len(df)} clean rows exported."))
