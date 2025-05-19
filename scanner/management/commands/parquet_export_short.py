from datetime import datetime
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

class Command(BaseCommand):
    help = 'Efficient streamed export of labeled RickisMetrics for short model to Parquet (March 23 – May 12, 2025)'

    def handle(self, *args, **kwargs):
        self.stdout.write("🚀 Exporting RickisMetrics for short model...")

        # 1️⃣ Date window
        start = make_aware(datetime(2025, 3, 23))
        end = make_aware(datetime(2025, 5, 12))

        # 2️⃣ Fields to include
        fields = [
            'id',  # Required for ID pagination
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
            'short_result'
        ]

        output_path = '/workspace/scanner/short.parquet'
        batch_size = 5000
        total_written = 0
        last_id = 0
        writer = None

        while True:
            batch_qs = (
                RickisMetrics.objects
                .filter(
                    id__gt=last_id,
                    timestamp__gte=start,
                    timestamp__lt=end,
                    short_result__isnull=False
                )
                .order_by('id')
                .values(*fields)[:batch_size]
            )

            rows = list(batch_qs)
            if not rows:
                break

            # Update last ID for pagination
            last_id = max(row['id'] for row in rows)

            df = pd.DataFrame.from_records(rows)
            df.drop(columns=['id'], inplace=True)

            # Cast decimal fields to float
            decimal_cols = [
                "price", "high_24h", "low_24h", "open", "close",
                "avg_volume_1h", "support_level", "resistance_level", "atr_1h"
            ]
            for col in decimal_cols:
                if col in df.columns:
                    df[col] = df[col].astype(float)

            df = df.apply(pd.to_numeric, errors='coerce')
            df.dropna(subset=['short_result'], inplace=True)

            table = pa.Table.from_pandas(df)

            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)

            total_written += len(df)
            self.stdout.write(f"✅ Wrote {total_written} rows so far...")

        if writer:
            writer.close()

        self.stdout.write(self.style.SUCCESS(f"🎉 Export complete. Total rows written: {total_written} to {output_path}"))
