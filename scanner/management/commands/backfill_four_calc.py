from datetime import datetime
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from scanner.helpers import (
    calculate_rsi, calculate_macd, calculate_price_change_five_min,
    calculate_avg_volume_1h, calculate_stochastic, calculate_support_resistance,
    calculate_relative_volume, calculate_sma, calculate_stddev_1h,
    calculate_atr_1h, calculate_obv
)

# nohup python manage.py backfill_four_calc > output.log 2>&1 &
# tail -f output.log

# need to full redo these:
#   calculate_stochastic
#   calculate_atr_1h
#   calculate_bollinger_bands
#   calculate_adx
#   fib -> the values being passed in
#   possibly change since high / low

class Command(BaseCommand):
    help = 'Recalculate missing metrics for RickisMetrics from May 9 to May 23, 2025'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 5, 9))
        end = make_aware(datetime(2025, 5, 23))

        metrics = RickisMetrics.objects.filter(timestamp__gte=start, timestamp__lt=end).select_related("coin")
        count = 0

        for metric in metrics:
            coin = metric.coin
            timestamp = metric.timestamp
            updated = False

            try:

                if metric.support_level in [None, 0] or metric.resistance_level in [None, 0]:
                    support, resistance = calculate_support_resistance(coin, timestamp)
                    if support is not None:
                        metric.support_level = support
                        updated = True
                    if resistance is not None:
                        metric.resistance_level = resistance
                        updated = True

                if metric.relative_volume in [None, 0]:
                    rel_vol = calculate_relative_volume(coin, timestamp)
                    if rel_vol is not None:
                        metric.relative_volume = rel_vol
                        updated = True

                if metric.sma_5 in [None, 0]:
                    sma5 = calculate_sma(coin, timestamp, 5)
                    if sma5 is not None:
                        metric.sma_5 = sma5
                        updated = True

                if metric.sma_20 in [None, 0]:
                    sma20 = calculate_sma(coin, timestamp, 20)
                    if sma20 is not None:
                        metric.sma_20 = sma20
                        updated = True

                if metric.stddev_1h in [None, 0]:
                    stddev = calculate_stddev_1h(coin, timestamp)
                    if stddev is not None:
                        metric.stddev_1h = stddev
                        updated = True

                if metric.obv in [None, 0]:
                    obv = calculate_obv(coin, timestamp)
                    if obv is not None:
                        metric.obv = obv
                        updated = True

                if updated:
                    metric.save()
                    count += 1
                    print(f"✅ Updated {count} metrics")

            except Exception as e:
                print(f"❌ Error at {coin.symbol} {timestamp}: {e}")

        print(f"✅ Done. Updated {count} metrics")


'''

python manage.py shell -c "
from scanner.models import RickisMetrics
from django.utils.timezone import make_aware
from django.db.models import Q
from datetime import datetime

start = make_aware(datetime(2025, 3, 23))
end = make_aware(datetime(2025, 5, 23))

fields = [
    'stochastic_k', 'stochastic_d', 'support_level', 'resistance_level',
    'relative_volume', 'sma_5', 'sma_20', 'stddev_1h', 'atr_1h', 'obv'
]

qs = RickisMetrics.objects.filter(timestamp__gte=start, timestamp__lt=end).filter(
    Q(stochastic_k__isnull=True) | Q(stochastic_k=0) |
    Q(stochastic_d__isnull=True) | Q(stochastic_d=0) |
    Q(support_level__isnull=True) | Q(support_level=0) |
    Q(resistance_level__isnull=True) | Q(resistance_level=0) |
    Q(relative_volume__isnull=True) | Q(relative_volume=0) |
    Q(sma_5__isnull=True) | Q(sma_5=0) |
    Q(sma_20__isnull=True) | Q(sma_20=0) |
    Q(stddev_1h__isnull=True) | Q(stddev_1h=0) |
    Q(atr_1h__isnull=True) | Q(atr_1h=0) |
    Q(obv__isnull=True) | Q(obv=0)
).values('timestamp', 'coin__symbol', *fields)

for row in qs:
    for field in fields:
        val = row.get(field)
        if val in [None, 0]:
            print(f\"{row['timestamp']} {row['coin__symbol']} {field} {val}\")
"



'''
