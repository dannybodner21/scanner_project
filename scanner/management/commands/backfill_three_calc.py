from datetime import datetime
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from scanner.helpers import (
    calculate_rsi,
    calculate_macd,
    calculate_price_change_five_min,
    calculate_avg_volume_1h,
    calculate_stochastic
)

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
    help = 'Recalculate missing core metrics: stochastic_k, stochastic_d for RickisMetrics from March 23 to April 23, 2025'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 23))
        end = make_aware(datetime(2025, 4, 23))

        metrics = (
            RickisMetrics.objects
            .filter(timestamp__gte=start, timestamp__lt=end)
            .filter(coin__symbol__in=TRACKED_SYMBOLS)  # <-- Only these coins
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
