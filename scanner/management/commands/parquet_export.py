from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
from scanner.models import RickisMetrics
import pandas as pd

class Command(BaseCommand):
    help = 'Export Metrics (long_result as target) to Parquet for training, from March 22 to April 29, 2025.'

    def handle(self, *args, **kwargs):
        # Define inclusive date window
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 4, 29)) + timedelta(days=1)

        # Filter metrics that have been labeled (wins or losses) in the date range
        qs = (
            RickisMetrics.objects
            .filter(
                timestamp__gte=start,
                timestamp__lt=end,
                long_result__isnull=False
            )
            .select_related('coin')
        )

        # Build values list, include timestamp and coin symbol
        fields = [
            'coin__symbol', 'timestamp', 'price', 'volume',
            'change_5m', 'change_1h', 'change_24h',
            'high_24h', 'low_24h', 'open', 'close',
            'avg_volume_1h', 'relative_volume',
            'sma_5', 'sma_20', 'macd', 'macd_signal',
            'rsi', 'stochastic_k', 'stochastic_d',
            'support_level', 'resistance_level',
            'stddev_1h', 'atr_1h', 'obv',
            'change_since_high', 'change_since_low',
            'fib_distance_0_236', 'fib_distance_0_382',
            'fib_distance_0_5', 'fib_distance_0_618', 'fib_distance_0_786',
            'long_result'
        ]

        metrics = qs.values(*fields)
        df = pd.DataFrame.from_records(metrics)

        if df.empty:
            self.stdout.write("⚠️ No labeled long-result metrics found in the given date range.")
            return

        output_path = '/workspace/scanner/long_metrics_export.parquet'
        df.to_parquet(output_path, index=False)
        self.stdout.write(f"✅ Exported {len(df)} rows to {output_path}")
