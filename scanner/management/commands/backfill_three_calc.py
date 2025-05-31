from datetime import datetime
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from scanner.helpers import calculate_stochastic
from django.db.models import Q

# Run with:
# nohup python manage.py backfill_three_calc > output.log 2>&1 &
# tail -f output.log

TRACKED_SYMBOLS = [
    "BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LINK",
    "AVAX", "XLM", "TON", "SHIB", "SUI", "HBAR", "BCH", "DOT", "LTC",
    "XMR", "UNI", "PEPE", "APT", "NEAR", "ONDO", "TAO", "ICP", "ETC",
    "RENDER", "MNT", "KAS", "CRO", "AAVE", "POL", "VET", "FIL", "ALGO",
    "ENA", "ATOM", "TIA", "ARB", "DEXE", "OP", "JUP", "MKR", "STX",
    "EOS", "WLD", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT",
    "PAXG", "CRV", "JASMY", "SAND", "GALA", "CORE", "KAIA", "LDO",
    "THETA", "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI"
]

class Command(BaseCommand):
    help = 'Fast Recalculate missing stochastic_k and stochastic_d'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 23))
        end = make_aware(datetime(2025, 4, 1))

        batch_size = 1000  # Tune this depending on your memory
        offset = 0
        count = 0

        while True:
            metrics_batch = list(
                RickisMetrics.objects
                .filter(timestamp__gte=start, timestamp__lt=end)
                .filter(coin__symbol__in=TRACKED_SYMBOLS)
                .filter(Q(stochastic_k__isnull=True) | Q(stochastic_d__isnull=True))
                .select_related("coin")
                .order_by('id')[offset:offset + batch_size]
            )

            if not metrics_batch:
                break

            for metric in metrics_batch:
                coin = metric.coin
                timestamp = metric.timestamp
                try:
                    k, d = calculate_stochastic(coin, timestamp)
                    if k is not None:
                        metric.stochastic_k = k
                    if d is not None:
                        metric.stochastic_d = d
                except Exception as e:
                    print(f"❌ Error at {coin.symbol} {timestamp}: {e}")

            # Bulk update
            RickisMetrics.objects.bulk_update(
                metrics_batch,
                fields=["stochastic_k", "stochastic_d"],
                batch_size=batch_size
            )

            count += len(metrics_batch)
            print(f"✅ Updated {count} metrics so far...")

            offset += batch_size

        print(f"🎯 DONE: Total {count} stochastic_k and stochastic_d updated")
