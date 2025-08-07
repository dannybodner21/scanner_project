
    # GRT-USD - done
    # XTZ-USD - done
    # AAVE-USD - done

    # ZRX-USD - done
    # MATIC-USD - done
    # ETC-USD - done
    # EOS-USD - done
    # ATOM-USD - done
    # BAT-USD - done
    # ALGO-USD - done
    # SNX-USD - done
    # SUSHI-USD - done
    # FIL-USD - done
    # HBAR-USD - done
    
# python manage.py clean_prices_with_coinapi GRTUSDT --start 2022-01-01 --end 2025-01-01 --fix-flat

from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice
from datetime import datetime, timedelta, timezone as dt_timezone
from decimal import Decimal
import requests
import time
from django.utils import timezone
from django.utils.timezone import make_aware

COINAPI_KEY = "01293e2a-dcf1-4e81-8310-c6aa9d0cb743"

COINAPI_SYMBOL_MAP = {
    "BTCUSDT": "BINANCE_SPOT_BTC_USDT", "ETHUSDT": "BINANCE_SPOT_ETH_USDT",
    "XRPUSDT": "BINANCE_SPOT_XRP_USDT", "LTCUSDT": "BINANCE_SPOT_LTC_USDT",
    "SOLUSDT": "BINANCE_SPOT_SOL_USDT", "DOGEUSDT": "BINANCE_SPOT_DOGE_USDT",
    "LINKUSDT": "BINANCE_SPOT_LINK_USDT", "DOTUSDT": "BINANCE_SPOT_DOT_USDT",
    "SHIBUSDT": "BINANCE_SPOT_SHIB_USDT", "ADAUSDT": "BINANCE_SPOT_ADA_USDT",
    "UNIUSDT": "BINANCE_SPOT_UNI_USDT", "AVAXUSDT": "BINANCE_SPOT_AVAX_USDT",
    "XLMUSDT": "BINANCE_SPOT_XLM_USDT", "GRTUSDT": "KRAKEN_SPOT_GRT_USD",
    "XTZUSDT": "BINANCE_SPOT_XTZ_USDT", "AAVEUSDT": "BINANCE_SPOT_AAVE_USDT",
}

class Command(BaseCommand):
    help = 'Fixes price data using a cost-effective, two-phase hybrid approach.'

    def add_arguments(self, parser):
        parser.add_argument("symbol", type=str)
        parser.add_argument("--start", type=str)
        parser.add_argument("--end", type=str)
        parser.add_argument("--fix-flat", action="store_true")

    def handle(self, *args, **options):
        symbol = options["symbol"]
        coinapi_symbol = COINAPI_SYMBOL_MAP.get(symbol)
        if not coinapi_symbol:
            self.stderr.write(self.style.ERROR(f"Symbol {symbol} not found."))
            return

        start_date = make_aware(datetime.fromisoformat(options["start"]))
        end_date = make_aware(datetime.fromisoformat(options["end"]))
        fix_flat = options["fix_flat"]

        headers = {"X-CoinAPI-Key": COINAPI_KEY}
        interval = timedelta(minutes=5)
        local_tz = timezone.get_current_timezone()

        day = start_date
        while day < end_date:
            next_day = day + timedelta(days=1)
            self.stdout.write(f"--- Processing {symbol} for {day.date()} ---")

            db_prices = CoinAPIPrice.objects.filter(coin=symbol, timestamp__gte=day, timestamp__lt=next_day)
            db_price_map = {p.timestamp: p for p in db_prices}
            
            initial_missing = {day + i * interval for i in range(288)} - set(db_price_map.keys())
            initial_flat = {p.timestamp: p for p in db_price_map.values() if p.open == p.high == p.low == p.close} if fix_flat else {}
            
            if not initial_missing and not initial_flat:
                self.stdout.write(self.style.SUCCESS(f"✅ Clean"))
                day = next_day
                continue
            
            self.stdout.write(f"🔎 Found {len(initial_missing)} missing, {len(initial_flat)} flat. Starting Phase 1...")

            # --- PHASE 1: Bulk Fetch & Fix (Cheap) ---
            fixed_in_bulk = set()
            try:
                start_utc, end_utc = day.astimezone(dt_timezone.utc), next_day.astimezone(dt_timezone.utc)
                url = (
                    f"https://rest.coinapi.io/v1/ohlcv/{coinapi_symbol}/history?"
                    f"period_id=5MIN&time_start={start_utc.replace(tzinfo=None).isoformat()}"
                    f"&time_end={end_utc.replace(tzinfo=None).isoformat()}&limit=5000"
                )
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                bulk_candles = response.json()

                bulk_ts_map = {
                    datetime.fromisoformat(c["time_period_start"].replace("Z", "+00:00")).astimezone(local_tz): c
                    for c in bulk_candles
                }

                # Fix missing candles found in bulk data
                insert_objs = []
                for ts in initial_missing:
                    if ts in bulk_ts_map:
                        c = bulk_ts_map[ts]
                        insert_objs.append(CoinAPIPrice(
                            coin=symbol, timestamp=ts, open=c['price_open'], high=c['price_high'],
                            low=c['price_low'], close=c['price_close'], volume=c['volume_traded']
                        ))
                        fixed_in_bulk.add(ts)
                
                if insert_objs:
                    CoinAPIPrice.objects.bulk_create(insert_objs, ignore_conflicts=True)
                    self.stdout.write(self.style.SUCCESS(f"  Phase 1: Inserted {len(insert_objs)} missing candles."))

                # Fix flat candles found in bulk data
                update_objs = []
                for ts, obj in initial_flat.items():
                    if ts in bulk_ts_map:
                        c = bulk_ts_map[ts]
                        obj.open, obj.high, obj.low, obj.close, obj.volume = c['price_open'], c['price_high'], c['price_low'], c['price_close'], c['volume_traded']
                        update_objs.append(obj)
                        fixed_in_bulk.add(ts)
                
                if update_objs:
                    CoinAPIPrice.objects.bulk_update(update_objs, ["open", "high", "low", "close", "volume"])
                    self.stdout.write(self.style.SUCCESS(f"  Phase 1: Updated {len(update_objs)} flat candles."))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"  Phase 1: Bulk fetch failed: {e}"))

            # --- PHASE 2: Targeted Fetch for Remaining Issues (Accurate but Expensive) ---
            unresolved_ts = (initial_missing | set(initial_flat.keys())) - fixed_in_bulk
            if not unresolved_ts:
                self.stdout.write("  Phase 1 fixed all issues.")
                day = next_day
                continue
            
            self.stdout.write(f"  Phase 2: {len(unresolved_ts)} issues remain. Fetching individually...")
            for ts_local in sorted(list(unresolved_ts)):
                time.sleep(0.1)  # Rate limit
                try:
                    start_utc = ts_local.astimezone(dt_timezone.utc)
                    url = (
                        f"https://rest.coinapi.io/v1/ohlcv/{coinapi_symbol}/history?"
                        f"period_id=5MIN&time_start={start_utc.replace(tzinfo=None).isoformat()}&limit=1"
                    )
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    data = response.json()

                    if not data:
                        self.stdout.write(self.style.WARNING(f"    - No data on exchange for {ts_local.strftime('%H:%M:%S')}"))
                        continue

                    c = data[0]
                    if ts_local in initial_missing:
                         CoinAPIPrice.objects.update_or_create(
                            coin=symbol, timestamp=ts_local,
                            defaults={'open':c['price_open'],'high':c['price_high'],'low':c['price_low'],'close':c['price_close'],'volume':c['volume_traded']}
                        )
                         self.stdout.write(self.style.SUCCESS(f"    + INSERTED missing candle at {ts_local.strftime('%H:%M:%S')}"))
                    elif ts_local in initial_flat:
                        CoinAPIPrice.objects.filter(coin=symbol, timestamp=ts_local).update(
                           open=c['price_open'],high=c['price_high'],low=c['price_low'],close=c['price_close'],volume=c['volume_traded']
                        )
                        self.stdout.write(self.style.SUCCESS(f"    ♻️ UPDATED flat candle at {ts_local.strftime('%H:%M:%S')}"))
                except Exception as e:
                    self.stderr.write(self.style.ERROR(f"    ❌ FAILED targeted fetch for {ts_local.strftime('%H:%M:%S')}: {e}"))
            
            day = next_day