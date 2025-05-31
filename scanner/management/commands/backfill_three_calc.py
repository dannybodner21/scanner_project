from datetime import datetime
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from scanner.helpers import calculate_stochastic
from django.db.models import Q
from collections import defaultdict

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
    help = 'Ultra Fast Recalculate stochastic_k and stochastic_d for RickisMetrics'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 23))
        end = make_aware(datetime(2025, 4, 1))

        print("🔍 Loading all metrics into memory...")

        # Load all relevant metrics
        all_metrics = list(
            RickisMetrics.objects.filter(
                timestamp__gte=start,
                timestamp__lt=end,
                coin__symbol__in=TRACKED_SYMBOLS
            ).select_related('coin')
            .order_by('timestamp')  # Important for sequential lookup
        )

        print(f"✅ Loaded {len(all_metrics)} metrics.")

        # Group metrics by coin
        metrics_by_coin = defaultdict(list)
        for metric in all_metrics:
            metrics_by_coin[metric.coin.symbol].append(metric)

        print("📚 Starting recalculation...")

        batch_size = 5000
        updates = []
        total_updated = 0

        for symbol, metrics in metrics_by_coin.items():
            print(f"⚡ Processing {symbol} with {len(metrics)} entries...")
            for idx, metric in enumerate(metrics):
                try:
                    # Smooth over last 3 entries
                    window = metrics[max(0, idx - 2):idx + 1]
                    if len(window) < 1:
                        continue

                    k_values = []
                    for win_metric in window:
                        close = float(win_metric.price)
                        high_24h = float(win_metric.high_24h)
                        low_24h = float(win_metric.low_24h)

                        if high_24h == low_24h:
                            k = 50.0
                        else:
                            k = (close - low_24h) / (high_24h - low_24h) * 100

                        k_values.append(k)

                    k = k_values[-1]  # latest %K
                    d = sum(k_values) / len(k_values)  # smoothed %D

                    metric.stochastic_k = k
                    metric.stochastic_d = d
                    updates.append(metric)

                except Exception as e:
                    print(f"❌ Error at {symbol} {metric.timestamp}: {e}")

                # Bulk update every batch_size
                if len(updates) >= batch_size:
                    RickisMetrics.objects.bulk_update(updates, ['stochastic_k', 'stochastic_d'], batch_size=batch_size)
                    total_updated += len(updates)
                    print(f"✅ {total_updated} metrics updated...")
                    updates.clear()

        # Final flush
        if updates:
            RickisMetrics.objects.bulk_update(updates, ['stochastic_k', 'stochastic_d'], batch_size=batch_size)
            total_updated += len(updates)
            print(f"✅ Final {total_updated} metrics updated...")

        print(f"🎯 DONE: Total {total_updated} stochastic_k and stochastic_d updated")
