from datetime import datetime
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from scanner.helpers import (
    calculate_rsi, calculate_macd, calculate_price_change_five_min,
    calculate_avg_volume_1h
)

# nohup python manage.py backfill_three_calc > output.log 2>&1 &
# tail -f output.log

class Command(BaseCommand):
    help = 'Recalculate missing core metrics: change_5m, avg_volume_1h, rsi, macd, macd_signal for RickisMetrics from May 9 to May 23, 2025'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 23))
        end = make_aware(datetime(2025, 4, 23))

        metrics = RickisMetrics.objects.filter(timestamp__gte=start, timestamp__lt=end).select_related("coin")
        count = 0

        for metric in metrics:
            coin = metric.coin
            timestamp = metric.timestamp
            updated = False

            try:


                if metric.stochastic_k in [None, 0] or metric.stochastic_d in [None, 0]:
                    k, d = calculate_stochastic(coin, timestamp)
                    if k is not None:
                        metric.stochastic_k = k
                        updated = True
                    if d is not None:
                        metric.stochastic_d = d
                        updated = True



                if updated:
                    metric.save()
                    count += 1
                    print(f"✅ Updated {count} metrics")

            except Exception as e:
                print(f"❌ Error at {coin.symbol} {timestamp}: {e}")

        print(f"✅ DONE: Updated {count} metrics")



'''
CHECK MISSING CHANGE5M, AVGVOLUME1H, RSI, MACD, MACD SIGNAL

python manage.py shell -c "
from scanner.models import RickisMetrics
from django.utils.timezone import make_aware
from django.db.models import Q
from datetime import datetime

start = make_aware(datetime(2025, 3, 23))
end = make_aware(datetime(2025, 5, 22))

count = RickisMetrics.objects.filter(
    timestamp__gte=start,
    timestamp__lt=end
).filter(
    Q(change_5m__isnull=True) |
    Q(avg_volume_1h__isnull=True) |
    Q(rsi__isnull=True) |
    Q(macd__isnull=True) |
    Q(macd_signal__isnull=True)
).count()

print(f'Missing or zero metrics: {count}')
"



CHECK OTHER MISSING VALUES

python manage.py shell -c "
from django.utils.timezone import make_aware
from datetime import datetime
from django.db.models import Q
from scanner.models import RickisMetrics

start = make_aware(datetime(2025, 3, 23))
end = make_aware(datetime(2025, 5, 23))

missing = RickisMetrics.objects.filter(
    timestamp__gte=start,
    timestamp__lt=end
).filter(
    Q(rsi__isnull=True)
).count()

print(f'Missing or zero stochastic values: {missing}')
"






'''
