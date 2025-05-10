from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from django.utils.timezone import make_aware
from datetime import datetime
import pandas as pd

class Command(BaseCommand):
    help = 'Export holdout RickisMetrics input features to CSV for batch prediction'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 4, 30))
        end = make_aware(datetime(2025, 5, 3))

        # Pull only the fields your model was trained on
        fields = [
            "price", "volume", "change_5m", "change_1h", "change_24h",
            "high_24h", "low_24h", "avg_volume_1h", "relative_volume",
            "rsi", "macd", "macd_signal", "stochastic_k", "stochastic_d",
            "support_level", "resistance_level", "sma_5", "sma_20",
            "stddev_1h", "atr_1h", "obv", "change_since_high",
            "change_since_low", "fib_distance_0_236", "fib_distance_0_382",
            "fib_distance_0_5", "fib_distance_0_618", "fib_distance_0_786",
            "open", "close"
        ]

        # Keep timestamp/coin_symbol just for reference after prediction
        extra_fields = ["timestamp", "coin__symbol", "long_result", "short_result"]

        qs = RickisMetrics.objects.filter(timestamp__gte=start, timestamp__lt=end)
        df = pd.DataFrame.from_records(qs.values(*extra_fields, *fields))

        # Rename for consistency
        df.rename(columns={"coin__symbol": "coin_symbol"}, inplace=True)

        # Save full version (with timestamp + label for later scoring)
        df.to_csv("full_holdout.csv", index=False)

        # Save model input only for batch prediction
        input_df = df[fields]
        input_df.to_csv("holdout_input.csv", index=False)

        self.stdout.write("✅ Exported holdout_input.csv and full_holdout.csv")
