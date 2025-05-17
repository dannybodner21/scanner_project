from datetime import datetime
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics, Coin
from scanner.helpers import calculate_macd, calculate_stochastic

class Command(BaseCommand):
    help = "Fix MACD for SHIB and Stochastic K/D for JASMY"

    def handle(self, *args, **kwargs):
        self.fix_macd_for_shib()
        self.fix_stochastic_for_jasmy()

    def fix_macd_for_shib(self):
        coin = Coin.objects.get(symbol="SHIB")
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 4, 23))
        updated_count = 0

        metrics = RickisMetrics.objects.filter(
            coin=coin, timestamp__gte=start, timestamp__lte=end
        )

        for metric in metrics:
            if metric.macd == 0 or metric.macd_signal == 0 or metric.macd is None or metric.macd_signal is None:
                macd, signal = calculate_macd(coin, metric.timestamp)
                if macd is not None and signal is not None:
                    metric.macd = macd
                    metric.macd_signal = signal
                    metric.save()
                    updated_count += 1

        print(f"✅ Fixed MACD for SHIB: {updated_count} entries updated.")

    def fix_stochastic_for_jasmy(self):
        coin = Coin.objects.get(symbol="JASMY")
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 5, 13))
        updated_count = 0

        metrics = RickisMetrics.objects.filter(
            coin=coin, timestamp__gte=start, timestamp__lte=end
        )

        for metric in metrics:
            if metric.stochastic_k == 0 or metric.stochastic_d == 0 or metric.stochastic_k is None or metric.stochastic_d is None:
                k, d = calculate_stochastic(coin, metric.timestamp)
                if k is not None and d is not None:
                    metric.stochastic_k = k
                    metric.stochastic_d = d
                    metric.save()
                    updated_count += 1

        print(f"✅ Fixed Stochastic K/D for JASMY: {updated_count} entries updated.")
