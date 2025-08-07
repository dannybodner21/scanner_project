import time
from datetime import datetime, timedelta
import requests
from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils.timezone import make_aware
from scanner.models import CoinAPIPrice


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


POLYGON_API_KEY = "qq9Sptr4VfkonQimqFJEgc3oyXoaJ54L"
SYMBOL = 'X:HBARUSD'
DB_COIN = 'HBARUSDT'
INTERVAL = 5  # 5-minute candles
BATCH_DAYS = 7

BASE_URL = (
    'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/minute/'
    '{from_date}/{to_date}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}'
)

class Command(BaseCommand):
    help = 'Fetch GRT-USD OHLCV data from Polygon and save to CoinAPIPrice'

    def handle(self, *args, **kwargs):
        self.stdout.write("⚠️  Deleting existing data...")


        CoinAPIPrice.objects.filter(coin=DB_COIN).delete()

        start_date = datetime(2022, 1, 1)
        end_date = datetime.utcnow()
        delta = timedelta(days=BATCH_DAYS)
        total_inserted = 0
        current = start_date

        while current < end_date:
            to_date = min(current + delta - timedelta(seconds=1), end_date)
            url = BASE_URL.format(
                symbol=SYMBOL,
                multiplier=INTERVAL,
                from_date=current.strftime('%Y-%m-%d'),
                to_date=to_date.strftime('%Y-%m-%d'),
                api_key=POLYGON_API_KEY
            )

            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                results = response.json().get("results", [])
            except requests.RequestException as e:
                self.stderr.write(f"❌ Error fetching data for {current.date()}: {e}")
                current += delta
                time.sleep(0.5)
                continue

            if not results:
                self.stdout.write(f"ℹ️  No data for {current.date()} → {to_date.date()}")
                current += delta
                continue

            objs = []
            for row in results:
                ts = make_aware(datetime.utcfromtimestamp(row["t"] / 1000))
                objs.append(CoinAPIPrice(
                    coin=DB_COIN,
                    timestamp=ts,
                    open=row["o"],
                    high=row["h"],
                    low=row["l"],
                    close=row["c"],
                    volume=row["v"]
                ))

            CoinAPIPrice.objects.bulk_create(objs, ignore_conflicts=True)
            self.stdout.write(f"✅ {len(objs)} rows inserted for {current.date()} → {to_date.date()}")
            total_inserted += len(objs)

            current += delta
            time.sleep(0.2)

        self.stdout.write(self.style.SUCCESS(f"🎉 Done. Total inserted: {total_inserted}"))
