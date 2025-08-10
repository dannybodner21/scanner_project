from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice
from datetime import datetime, timedelta, timezone as dt_timezone
from django.utils.timezone import make_aware
from collections import Counter



# python manage.py check_price_integrity --start 2023-01-01 --end 2025-08-08 --coins BTCUSDT ETHUSDT XRPUSDT LTCUSDT SOLUSDT DOGEUSDT LINKUSDT DOTUSDT SHIBUSDT ADAUSDT UNIUSDT AVAXUSDT XLMUSDT TRXUSDT ATOMUSDT

# python manage.py check_price_integrity --start 2023-01-01 --end 2025-08-08 --coins ATOMUSDT




    


class Command(BaseCommand):
    help = 'Strictly validate CoinAPIPrice: missing candles, duplicates, off-grid, flat candles, flat windows, and invalid OHLC.'

    def add_arguments(self, parser):
        parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
        parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
        parser.add_argument('--coins', nargs='+', required=True, help='List of coins to check')

    def handle(self, *args, **options):
        # Use UTC for deterministic day boundaries
        start_date = make_aware(datetime.strptime(options['start'], '%Y-%m-%d'), dt_timezone.utc)
        end_date = make_aware(datetime.strptime(options['end'], '%Y-%m-%d'), dt_timezone.utc)

        coins = [c.strip().upper() for c in options['coins']]

        days_checked = {c: 0 for c in coins}
        days_clean = {c: 0 for c in coins}

        day = start_date
        while day <= end_date:
            next_day = day + timedelta(days=1)

            # Pre-compute expected 5m grid for the day in UTC
            interval = timedelta(minutes=5)
            expected_ts = [day + i * interval for i in range(288)]
            expected_set = set(expected_ts)

            for coin in coins:
                days_checked[coin] += 1

                qs = CoinAPIPrice.objects.filter(
                    coin=coin,
                    timestamp__gte=day,
                    timestamp__lt=next_day
                ).order_by('timestamp')

                candles = list(qs)
                ts_list = [c.timestamp for c in candles]
                ts_set = set(ts_list)

                issues_found = False

                # 1) Count check and missing/extraneous timestamps
                if len(candles) != 288:
                    missing = expected_set - ts_set
                    extra = ts_set - expected_set
                    issues_found = True
                    print(f'❌ MISSING: {coin} | {day.date()} | {len(candles)}/288 candles | missing={len(missing)} extra={len(extra)}')
                    if missing:
                        examples = sorted(list(missing))[:5]
                        print('   ↳ missing examples:', ", ".join(t.strftime('%H:%M') for t in examples))
                    if extra:
                        examples = sorted(list(extra))[:5]
                        print('   ↳ off-grid examples:', ", ".join(t.strftime('%H:%M') for t in examples))
                else:
                    # Even if count is 288, still validate exact grid alignment
                    if ts_set != expected_set:
                        missing = expected_set - ts_set
                        extra = ts_set - expected_set
                        issues_found = True
                        print(f'❌ GRID MISMATCH: {coin} | {day.date()} | missing={len(missing)} extra={len(extra)}')
                        if missing:
                            examples = sorted(list(missing))[:5]
                            print('   ↳ missing examples:', ", ".join(t.strftime('%H:%M') for t in examples))
                        if extra:
                            examples = sorted(list(extra))[:5]
                            print('   ↳ off-grid examples:', ", ".join(t.strftime('%H:%M') for t in examples))

                # 2) Duplicate timestamps
                counter = Counter(ts_list)
                dups = {t: c for t, c in counter.items() if c > 1}
                if dups:
                    issues_found = True
                    print(f'❌ DUPLICATES: {coin} | {day.date()} | {len(dups)} duplicate timestamps')
                    examples = list(dups.items())[:5]
                    print('   ↳ examples:', ", ".join(f"{t.strftime('%H:%M')}×{c}" for t, c in examples))

                # 3) Off-grid alignment by minute/second (defensive)
                misaligned = [t for t in ts_list if (t.second != 0 or t.microsecond != 0 or (t.minute % 5) != 0)]
                if misaligned:
                    issues_found = True
                    print(f'❌ MISALIGNED: {coin} | {day.date()} | {len(misaligned)} timestamps not on 5m boundaries')
                    examples = misaligned[:5]
                    print('   ↳ examples:', ", ".join(t.strftime('%H:%M:%S') for t in examples))

                # 4) Flat candles and flat windows (3+)
                flat_candles = 0
                flat_window_count = 0
                flat_streak = 0
                for c in candles:
                    is_flat = (c.open == c.high == c.low == c.close)
                    if is_flat:
                        flat_candles += 1
                        flat_streak += 1
                    else:
                        if flat_streak >= 3:
                            flat_window_count += 1
                        flat_streak = 0
                if flat_streak >= 3:
                    flat_window_count += 1

                if flat_candles > 0:
                    issues_found = True
                    print(f'⚠️ FLAT CANDLES: {coin} | {day.date()} | {flat_candles} flat')
                if flat_window_count > 0:
                    issues_found = True
                    print(f'⚠️ FLAT WINDOWS: {coin} | {day.date()} | {flat_window_count} windows (≥3 contiguous)')

                # 5) Invalid OHLC structure
                invalid_ohlc = 0
                for c in candles:
                    try:
                        # Convert to floats for comparisons (Decimal is fine, but float is ok for ordering checks)
                        o, h, l, cl = float(c.open or 0), float(c.high or 0), float(c.low or 0), float(c.close or 0)
                    except Exception:
                        invalid_ohlc += 1
                        continue
                    if any(v is None for v in [c.open, c.high, c.low, c.close]):
                        invalid_ohlc += 1
                        continue
                    if not (h >= max(o, cl) and l <= min(o, cl) and h >= l):
                        invalid_ohlc += 1

                if invalid_ohlc > 0:
                    issues_found = True
                    print(f'❌ INVALID OHLC: {coin} | {day.date()} | {invalid_ohlc} candles with inconsistent OHLC')

                if not issues_found:
                    days_clean[coin] += 1
                    print(f'✅ CLEAN: {coin} | {day.date()}')

            day += timedelta(days=1)

        # Summary per coin
        print('\n===== SUMMARY =====')
        for coin in coins:
            print(f'{coin}: {days_clean[coin]}/{days_checked[coin]} days clean')
