
# nohup python manage.py backfill_nine > output.log 2>&1 &
# tail -f output.log


from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from scanner.models import RickisMetrics, Coin
from scanner.helpers import (
    calculate_price_change_five_min,
    calculate_stochastic,
    calculate_stddev_1h,
    calculate_atr_1h,
    calculate_obv,
    calculate_fib_distances,
)
import time

TRACKED_SYMBOLS = [
    "BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "DOGE", "ADA", "LINK",
    "AVAX", "XLM", "TON", "SHIB", "SUI", "HBAR", "BCH", "DOT", "LTC",
    "XMR", "UNI", "PEPE", "APT", "NEAR", "ONDO", "TAO", "ICP", "ETC",
    "RENDER", "MNT", "KAS", "CRO", "AAVE", "POL", "VET", "FIL", "ALGO",
    "ENA", "ATOM", "TIA", "ARB", "DEXE", "OP", "JUP", "MKR", "STX",
    "EOS", "WLD", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT",
    "PAXG", "CRV", "JASMY", "SAND", "GALA", "CORE", "KAIA", "LDO",
    "THETA", "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI"
]

class Command(BaseCommand):
    help = 'Fill missing or zero metric fields for tracked coins between March 23 - May 23, 2025'

    def handle(self, *args, **kwargs):
        start_date = make_aware(datetime(2025, 3, 23))
        end_date = make_aware(datetime(2025, 5, 23))

        coins = Coin.objects.filter(symbol__in=TRACKED_SYMBOLS)
        print(f"🚀 Found {coins.count()} tracked coins to process.")

        batch_size = 500
        batch = []
        total_updates = 0

        for coin in coins:
            print(f"\n🔍 Processing {coin.symbol}...")

            metrics = RickisMetrics.objects.filter(
                coin=coin,
                timestamp__gte=start_date,
                timestamp__lt=end_date
            ).order_by('timestamp')

            total_metrics = metrics.count()
            print(f"📊 {total_metrics} metrics to check for {coin.symbol}.")

            for metric in metrics:
                update_needed = False

                if not metric.change_5m or metric.change_5m == 0:
                    metric.change_5m = calculate_price_change_five_min(coin, metric.timestamp)
                    update_needed = True

                if (not metric.stochastic_k or metric.stochastic_k == 0) or (not metric.stochastic_d or metric.stochastic_d == 0):
                    k, d = calculate_stochastic(coin, metric.timestamp)
                    metric.stochastic_k = k
                    metric.stochastic_d = d
                    update_needed = True

                if not metric.stddev_1h or metric.stddev_1h == 0:
                    metric.stddev_1h = calculate_stddev_1h(coin, metric.timestamp)
                    update_needed = True

                if not metric.atr_1h or metric.atr_1h == 0:
                    metric.atr_1h = calculate_atr_1h(coin, metric.timestamp)
                    update_needed = True

                if not metric.obv or metric.obv == 0:
                    metric.obv = calculate_obv(coin, metric.timestamp)
                    update_needed = True

                fibs = calculate_fib_distances(metric.high_24h, metric.low_24h, metric.price)

                if fibs:
                    if not metric.fib_distance_236 or metric.fib_distance_236 == 0:
                        metric.fib_distance_236 = fibs.get('fib_distance_0_236')
                        update_needed = True
                    if not metric.fib_distance_382 or metric.fib_distance_382 == 0:
                        metric.fib_distance_382 = fibs.get('fib_distance_0_382')
                        update_needed = True
                    if not metric.fib_distance_5 or metric.fib_distance_5 == 0:
                        metric.fib_distance_5 = fibs.get('fib_distance_0_5')
                        update_needed = True
                    if not metric.fib_distance_618 or metric.fib_distance_618 == 0:
                        metric.fib_distance_618 = fibs.get('fib_distance_0_618')
                        update_needed = True
                    if not metric.fib_distance_786 or metric.fib_distance_786 == 0:
                        metric.fib_distance_786 = fibs.get('fib_distance_0_786')
                        update_needed = True

                if update_needed:
                    batch.append(metric)
                    total_updates += 1

                if len(batch) >= batch_size:
                    RickisMetrics.objects.bulk_update(batch, [
                        'change_5m', 'stochastic_k', 'stochastic_d',
                        'stddev_1h', 'atr_1h', 'obv',
                        'fib_distance_236', 'fib_distance_382', 'fib_distance_5',
                        'fib_distance_618', 'fib_distance_786'
                    ])
                    print(f"💾 Saved {len(batch)} metrics (batch).")
                    batch = []
                    time.sleep(0.5)  # Sleep a bit to avoid DB hammering

            if batch:
                RickisMetrics.objects.bulk_update(batch, [
                    'change_5m', 'stochastic_k', 'stochastic_d',
                    'stddev_1h', 'atr_1h', 'obv',
                    'fib_distance_236', 'fib_distance_382', 'fib_distance_5',
                    'fib_distance_618', 'fib_distance_786'
                ])
                print(f"💾 Saved {len(batch)} metrics (final batch).")
                batch = []

        print(f"\n🎯 Metric filling completed. Total updates: {total_updates}")
