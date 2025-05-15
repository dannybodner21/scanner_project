from datetime import datetime
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics, Coin
from scanner.helpers import (
    calculate_stochastic,
)

class Command(BaseCommand):
    help = 'Recalculate missing stochastic metrics for JASMY between April 20 and May 12'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 4, 22))

        try:
            jasmy = Coin.objects.get(symbol="JASMY")
        except Coin.DoesNotExist:
            print("❌ Coin JASMY not found.")
            return

        metrics = RickisMetrics.objects.filter(
            coin=jasmy,
            timestamp__gte=start,
            timestamp__lt=end
        ).select_related("coin")

        count = 0
        for metric in metrics:
            timestamp = metric.timestamp
            updated = False

            try:
                if (
                    metric.stochastic_k == 0 or metric.stochastic_d == 0 or
                    metric.stochastic_k is None or metric.stochastic_d is None
                ):
                    k, d = calculate_stochastic(jasmy, timestamp)
                    print(f"in stochastic: {k} {d} - {jasmy.symbol} - {timestamp}")
                    if k is not None and d is not None:
                        metric.stochastic_k = k
                        metric.stochastic_d = d
                        updated = True
                    else:
                        print(f"stochastic returned NONE: {jasmy.symbol} at {timestamp}")

                if updated:
                    count += 1
                    metric.save()

            except Exception as e:
                print(f"❌ Error at {jasmy.symbol} {timestamp}: {e}")

        print(f"✅ Recalculation complete for JASMY. {count} metrics updated.")
