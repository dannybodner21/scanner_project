from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
from scanner.models import RickisMetrics
import pandas as pd

class Command(BaseCommand):
    help = 'Export RickisMetrics to Parquet for short model training (full features + label)'

    def handle(self, *args, **kwargs):
        self.stdout.write("🚀 Exporting RickisMetrics for short model...")

        # 1️⃣ Date window
        start = make_aware(datetime(2025, 4, 30))
        end   = make_aware(datetime(2025, 5, 2)) + timedelta(days=1)

        # 2️⃣ Fields including coin symbol and timestamp
        fields = [
            'price', 'volume',
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
        ]

        # 3️⃣ Filter to only completed labels in range
        qs = (
            RickisMetrics.objects
            .filter(timestamp__gte=start, timestamp__lt=end, short_result__isnull=False)
            .select_related('coin')
            .values(*fields)
        )

        df = pd.DataFrame.from_records(qs)
        if df.empty:
            self.stdout.write(self.style.ERROR("❌ No labeled short-result metrics found in the given date range."))
            return

        # 4️⃣ Cast high-precision decimals to float so Parquet writes them as DOUBLE
        decimal_cols = [
            "price", "high_24h", "low_24h", "open", "close",
            "avg_volume_1h", "support_level", "resistance_level",
            "atr_1h"
        ]
        for col in decimal_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)

        # 5️⃣ Clean numeric columns only (excluding coin__symbol and timestamp)
        numeric_cols = [c for c in df.columns if c not in ('coin__symbol', 'timestamp')]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # 6️⃣ Drop any rows with NaNs in numeric features or the label
        #df.dropna(subset=numeric_cols + ['short_result'], inplace=True)

        output_path = '/workspace/scanner/short_test.parquet'
        df.to_parquet(output_path, index=False)

        self.stdout.write(self.style.SUCCESS(f"✅ Exported {len(df)} rows to {output_path}"))
