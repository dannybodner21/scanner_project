from datetime import datetime
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from scanner.helpers import (
    calculate_rsi,
    calculate_macd,
    calculate_price_change_five_min,
    calculate_avg_volume_1h,
    calculate_stochastic  # ⬅️ make sure it's imported!
)

# Run with:
# nohup python manage.py backfill_three_calc > output.log 2>&1 &
# tail -f output.log

class Command(BaseCommand):
    help = 'Recalculate missing core metrics: stochastic_k, stochastic_d for RickisMetrics from March 23 to April 23, 2025'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 23))
        end = make_aware(datetime(2025, 4, 23))

        metrics = (
            RickisMetrics.objects
            .filter(timestamp__gte=start, timestamp__lt=end)
            .filter(Q(stochastic_k__isnull=True) | Q(stochastic_d__isnull=True))
            .select_related("coin")
            .iterator(chunk_size=500)  # ⚡ efficient
        )

        count = 0

        for metric in metrics:
            coin = metric.coin
            timestamp = metric.timestamp
            updated = False

            try:
                # Only calculate if needed
                if metric.stochastic_k in [None, 0] or metric.stochastic_d in [None, 0]:
                    k, d = calculate_stochastic(coin, timestamp)

                    if k is not None and metric.stochastic_k != k:
                        metric.stochastic_k = k
                        updated = True

                    if d is not None and metric.stochastic_d != d:
                        metric.stochastic_d = d
                        updated = True

                if updated:
                    metric.save(update_fields=["stochastic_k", "stochastic_d"])
                    count += 1
                    if count % 100 == 0:  # Log every 100 updates
                        print(f"✅ Updated {count} metrics so far...")

            except Exception as e:
                print(f"❌ Error at {coin.symbol} {timestamp}: {e}")

        print(f"🎯 DONE: Total {count} stochastic_k and stochastic_d updated")
