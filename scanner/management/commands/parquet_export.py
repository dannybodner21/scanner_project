from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
import pandas as pd

class Command(BaseCommand):
    help = 'Export RickisMetrics to Parquet file (fix types)'

    def handle(self, *args, **kwargs):
        self.stdout.write("🚀 Exporting RickisMetrics...")

        queryset = RickisMetrics.objects.filter(
            long_result__isnull=False,
            short_result__isnull=False
        ).values(
            'timestamp', 'coin__symbol',
            'price', 'volume', 'change_5m', 'change_1h', 'change_24h',
            'high_24h', 'low_24h', 'avg_volume_1h', 'relative_volume',
            'sma_5', 'sma_20', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'rsi', 'stddev_1h',
            'price_slope_1h', 'stochastic_k', 'stochastic_d',
            'support_level', 'resistance_level', 'atr_1h',
            'long_result', 'short_result'
        )

        df = pd.DataFrame(list(queryset))

        if df.empty:
            self.stdout.write(self.style.ERROR("❌ No RickisMetrics entries found."))
            return

        df.rename(columns={'coin__symbol': 'coin_symbol'}, inplace=True)

        # Force critical decimals to float32/float64
        float_fields = [
            'atr_1h', 'price_slope_1h', 'stddev_1h', 'relative_volume'
        ]
        for field in float_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')

        output_path = '/workspace/scanner/rickismetrics_export_fixed.parquet'
        df.to_parquet(output_path)

        self.stdout.write(self.style.SUCCESS(f"✅ Export complete. {len(df)} rows saved to {output_path}"))
