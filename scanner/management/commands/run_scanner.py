from django.core.management.base import BaseCommand
from django.utils.timezone import now, timedelta
from scanner.models import Metrics

class Command(BaseCommand):
    help = "Debug: print recent metrics and highlight potential signals."

    def handle(self, *args, **kwargs):
        time_cutoff = now() - timedelta(minutes=10)

        recent_metrics = Metrics.objects.filter(timestamp__gte=time_cutoff)

        if not recent_metrics.exists():
            self.stdout.write("❌ No recent metrics found in the last 10 minutes.")
            return

        self.stdout.write(f"📊 Found {recent_metrics.count()} recent metrics:\n")

        for m in recent_metrics.order_by('-five_min_relative_volume'):
            signal = (
                m.five_min_relative_volume and m.five_min_relative_volume > 2.0 and
                m.price_change_5min and m.price_change_5min > 2.0 and
                m.price_change_1hr and m.price_change_1hr < 10.0 and
                m.market_cap and m.market_cap > 10_000_000
            )

            tag = "🚨 SIGNAL" if signal else "—"

            self.stdout.write(
                f"{tag} {m.coin.symbol} | "
                f"5m Δ: {m.price_change_5min:.2f}% | "
                f"1h Δ: {m.price_change_1hr:.2f}% | "
                f"Vol Spike: {m.five_min_relative_volume:.2f}x | "
                f"Market Cap: ${int(m.market_cap):,}"
            )
