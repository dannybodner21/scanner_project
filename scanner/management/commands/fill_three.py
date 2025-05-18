from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime
from scanner.models import RickisMetrics
from scanner.helpers import calculate_price_change_five_min, calculate_stochastic

class Command(BaseCommand):
    help = "Update change_5m, stochastic_k, and stochastic_d for RickisMetrics entries from March 22 to May 2"

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 22))
        end = make_aware(datetime(2025, 4, 22))
        qs = RickisMetrics.objects.filter(timestamp__gte=start, timestamp__lte=end).select_related("coin")

        total = qs.count()
        updated = 0
        null_or_zero_change = 0
        null_or_zero_k = 0
        null_or_zero_d = 0

        self.stdout.write(f"🔍 Updating {total} entries...")

        for metric in qs:
            try:
                change_5m = calculate_price_change_five_min(metric.coin)
                k, d = calculate_stochastic(metric.coin, metric.timestamp)

                # Only update if value is not None and not zero
                if change_5m is not None and change_5m != 0:
                    metric.change_5m = change_5m
                if k is not None and k != 0:
                    metric.stochastic_k = k
                if d is not None and d != 0:
                    metric.stochastic_d = d

                metric.save()
                updated += 1

                if not change_5m or change_5m == 0:
                    null_or_zero_change += 1
                if not k or k == 0:
                    null_or_zero_k += 1
                if not d or d == 0:
                    null_or_zero_d += 1

            except Exception as e:
                self.stdout.write(f"❌ Error at {metric.coin.symbol} {metric.timestamp}: {e}")

        self.stdout.write(f"\n✅ Updated: {updated}")
        self.stdout.write(f"⚠️ change_5m == 0 or None: {null_or_zero_change}")
        self.stdout.write(f"⚠️ stochastic_k == 0 or None: {null_or_zero_k}")
        self.stdout.write(f"⚠️ stochastic_d == 0 or None: {null_or_zero_d}")



        '''


march 23, 2025 (inclusive) - may 12, 2025 (not inclusive)
14,400 entries per coin - 50 days


from datetime import datetime
from django.utils.timezone import make_aware
from scanner.models import Coin, RickisMetrics

# Define date range
start = make_aware(datetime(2025, 3, 23))
end = make_aware(datetime(2025, 5, 12))

# Define the coin symbols to check
symbols = ["BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LINK",
"AVAX", "XLM", "TON", "SHIB", "SUI", "HBAR", "BCH", "DOT", "LTC",
"XMR", "UNI", "PEPE", "APT", "NEAR", "ONDO", "TAO", "ICP", "ETC",
"RENDER", "MNT", "KAS", "CRO", "AAVE", "POL", "VET", "FIL", "ALGO",
"ENA", "ATOM", "TIA", "ARB", "DEXE", "OP", "JUP", "MKR", "STX",
"EOS", "WLD", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT",
"PAXG", "CRV", "JASMY", "SAND", "GALA", "CORE", "KAIA", "LDO",
"THETA", "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI"]

# Go through each coin
for symbol in symbols:
    try:
        coin = Coin.objects.get(symbol=symbol)
        metrics = RickisMetrics.objects.filter(coin=coin, timestamp__gte=start, timestamp__lt=end)

        total = metrics.count()
        missing = metrics.filter(change_1h__isnull=True).count()
        zero = metrics.filter(change_1h=0).count()

        print(f"{symbol}: {total} entries — Missing: {missing}, Zero: {zero}")

    except Coin.DoesNotExist:
        print(f"❌ Coin not found: {symbol}")





python manage.py shell -c "
from scanner.models import RickisMetrics, Coin
coin = Coin.objects.get(symbol='ICP')
qs = RickisMetrics.objects.filter(coin=coin, change_1h__isnull=True).order_by('timestamp')
for m in qs: print(m.timestamp)
"




change_5m = models.FloatField(null=True) - good

NEED TO CALL HISTORICAL API FOR THESE:
change_1h = models.FloatField(null=True) - missing
change_24h = models.FloatField(null=True) - missing

volume = models.DecimalField(max_digits=30, decimal_places=2) - good
avg_volume_1h = models.DecimalField(max_digits=30, decimal_places=2, null=True) - good

rsi = models.FloatField(null=True) - good

macd = models.FloatField(null=True) - SHIB has zeros, rest is good
macd_signal = models.FloatField(null=True) - SHIB has zeros, rest is good
stochastic_k = models.FloatField(null=True) - JASMY 288 zeros, rest is good
stochastic_d = models.FloatField(null=True) - JASMY 286 zeros, rest is good

support_level = models.DecimalField(max_digits=20, decimal_places=10, null=True) - good
resistance_level = models.DecimalField(max_digits=20, decimal_places=10, null=True) - good
relative_volume = models.FloatField(null=True) - good
sma_5 = models.DecimalField(max_digits=20, decimal_places=10, null=True) - good
sma_20 = models.DecimalField(max_digits=20, decimal_places=10, null=True) - good

stddev_1h = models.FloatField(null=True) - zeros

atr_1h = models.DecimalField(max_digits=20, decimal_places=10, null=True) - good
obv = models.FloatField(null=True) - good
change_since_high = models.FloatField(null=True) - good
change_since_low = models.FloatField(null=True) - good

fib_distance_0_236 = models.FloatField(null=True) - good
fib_distance_0_382 = models.FloatField(null=True) - good
fib_distance_0_5   = models.FloatField(null=True) - good
fib_distance_0_618 = models.FloatField(null=True) - good
fib_distance_0_786 = models.FloatField(null=True) - good



long_result = models.BooleanField(null=True)

short_result = models.BooleanField(null=True)




nohup python manage.py fill_seven > output.log 2>&1 &

tail -f output.log


        '''
