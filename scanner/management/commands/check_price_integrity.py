from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice
from datetime import datetime, timedelta
from django.utils.timezone import make_aware



# python manage.py check_price_integrity --start 2022-01-01 --end 2025-08-04 --coins BTCUSDT ETHUSDT XRPUSDT LTCUSDT SOLUSDT DOGEUSDT LINKUSDT DOTUSDT SHIBUSDT ADAUSDT UNIUSDT AVAXUSDT XLMUSDT

# python manage.py check_price_integrity --start 2022-01-01 --end 2025-08-04 --coins GRTUSDT



    # ZRX-USD - done
    # GRT-USD - done
    # MATIC-USD - done
    # ETC-USD - done
    # EOS-USD - done
    # XTZ-USD - done
    # ATOM-USD - done
    # BAT-USD - done
    # ALGO-USD - done
    # AAVE-USD - done
    # SNX-USD - done
    # SUSHI-USD - done
    # FIL-USD - done
    # HBAR-USD - done


class Command(BaseCommand):
    help = 'Check CoinAPIPrice for missing candles, flat candles, and flat windows'

    def add_arguments(self, parser):
        parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
        parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
        parser.add_argument('--coins', nargs='+', required=True, help='List of coins to check')

    def handle(self, *args, **options):


        start_date = make_aware(datetime.strptime(options['start'], '%Y-%m-%d'))
        end_date = make_aware(datetime.strptime(options['end'], '%Y-%m-%d'))


        coins = options['coins']

        day = start_date
        while day <= end_date:
            next_day = day + timedelta(days=1)

            for coin in coins:
                candles = list(CoinAPIPrice.objects.filter(
                    coin=coin,
                    timestamp__gte=day,
                    timestamp__lt=next_day
                ).order_by('timestamp'))

                count = len(candles)
                if count != 288:
                    print(f'❌ MISSING: {coin} | {day.date()} | {count}/288 candles')

                # Check for flat candles and flat windows
                flat_candles = 0
                flat_window_count = 0
                current_flat_streak = 0

                for c in candles:
                    if c.open == c.high == c.low == c.close:
                        flat_candles += 1
                        current_flat_streak += 1
                    else:
                        if current_flat_streak >= 3:
                            flat_window_count += 1
                        current_flat_streak = 0

                if current_flat_streak >= 3:
                    flat_window_count += 1

                if flat_candles > 0:
                    print(f'⚠️ FLAT CANDLES: {coin} | {day.date()} | {flat_candles} found')

                if flat_window_count > 0:
                    print(f'⚠️ FLAT WINDOWS: {coin} | {day.date()} | {flat_window_count} windows (3+ flat candles)')

            day += timedelta(days=1)
