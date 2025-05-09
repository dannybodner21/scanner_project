from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
import pandas as pd

class Command(BaseCommand):
    help = 'Export Metrics to Parquet'

    def handle(self, *args, **kwargs):

        # get all the Metrics with a long result
        metrics = RickisMetrics.objects.filter(
            long_result__isnull=False
        ).values(
            'price', 'volume', 'change_5m', 'change_1h', 'change_24h',
            'high_24h', 'low_24h', 'avg_volume_1h', 'relative_volume',
            'sma_5', 'sma_20', 'macd', 'macd_signal', 'open', 'close',
            'rsi', 'stochastic_k', 'stochastic_d', 'support_level',
            'resistance_level', 'stddev_1h', 'atr_1h', 'obv',
            'long_result', 'change_since_high', 'change_since_low',
            'fib_distance_0_236', 'fib_distance_0_382',
            'fib_distance_0_5', 'fib_distance_0_618',
            'fib_distance_0_786',
        )

        df = pd.DataFrame(list(metrics))

        if df.empty:
            self.stdout.write("no data found.")
            return

        df.to_parquet('/workspace/scanner/long_metrics_export.parquet')
