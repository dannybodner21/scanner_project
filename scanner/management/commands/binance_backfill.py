import time
from datetime import datetime, timezone
import math
import requests
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from scanner.models import CoinAPIPrice

BINANCE_SYMBOL = "GRTUSDT"     # e.g., GRTUSDT
DB_COIN = "GRTUSDT"
INTERVAL = "5m"

MAX_LIMIT = 1000
INTERVAL_MS = 5 * 60 * 1000
EPS = 1e-9  # numeric tolerance for "different"

START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime.utcnow().replace(tzinfo=timezone.utc)

class Command(BaseCommand):
    help = f"Upsert {BINANCE_SYMBOL} {INTERVAL} OHLCV from Binance into CoinAPIPrice"

    def _ts_ms(self, dt: datetime) -> int:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    def _from_ms(self, ms: int) -> datetime:
        return datetime.utcfromtimestamp(ms / 1000).replace(tzinfo=timezone.utc)

    def _fetch_klines(self, symbol: str, start_ms: int, end_ms: int, limit: int = MAX_LIMIT):
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": min(limit, MAX_LIMIT),
        }
        for attempt in range(5):
            try:
                r = requests.get(url, params=params, timeout=15)
                ctype = r.headers.get("Content-Type", "")
                if "application/json" in ctype and r.text.startswith("{"):
                    j = r.json()
                    if isinstance(j, dict) and j.get("code") == 0 and "restricted" in j.get("msg", "").lower():
                        raise RuntimeError("Binance geo restriction detected. Use a non-US IP.")
                r.raise_for_status()
                return r.json()
            except requests.HTTPError:
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(1 + attempt)
        raise RuntimeError("Failed to fetch klines after retries.")

    def _diff(self, a, b):
        # treat NaNs as unequal; otherwise compare with tolerance
        if a is None or b is None:
            return True
        return not math.isclose(float(a), float(b), rel_tol=0, abs_tol=EPS)

    def handle(self, *args, **kwargs):
        start_ms = self._ts_ms(START_DATE)
        end_ms = self._ts_ms(END_DATE)
        total_created = 0
        total_updated = 0

        self.stdout.write(f"▶️  Upserting {BINANCE_SYMBOL} {INTERVAL} from {START_DATE.date()} to {END_DATE.date()}")

        while start_ms <= end_ms:
            batch_end_ms = min(start_ms + (MAX_LIMIT * INTERVAL_MS) - 1, end_ms)

            try:
                klines = self._fetch_klines(BINANCE_SYMBOL, start_ms, batch_end_ms)
            except RuntimeError as e:
                self.stderr.write(f"❌ {e}")
                return
            except requests.RequestException as e:
                self.stderr.write(f"❌ Request error: {e}")
                start_ms = batch_end_ms + 1
                time.sleep(0.2)
                continue

            if not klines:
                s_dt = self._from_ms(start_ms)
                e_dt = self._from_ms(batch_end_ms)
                self.stdout.write(f"ℹ️  No data for {s_dt} → {e_dt}")
                start_ms = batch_end_ms + 1
                time.sleep(0.12)
                continue

            # Build desired rows
            desired = []
            ts_list = []
            for row in klines:
                open_time_ms = int(row[0])
                ts = make_aware(datetime.utcfromtimestamp(open_time_ms / 1000))
                desired.append((
                    ts,
                    float(row[1]),  # open
                    float(row[2]),  # high
                    float(row[3]),  # low
                    float(row[4]),  # close
                    float(row[5]),  # volume
                ))
                ts_list.append(ts)

            # Fetch existing rows for these timestamps
            existing_qs = CoinAPIPrice.objects.filter(coin=DB_COIN, timestamp__in=ts_list)
            existing_by_ts = {obj.timestamp: obj for obj in existing_qs}

            to_create = []
            to_update = []

            for ts, o, h, l, c, v in desired:
                obj = existing_by_ts.get(ts)
                if obj is None:
                    to_create.append(CoinAPIPrice(
                        coin=DB_COIN,
                        timestamp=ts,
                        open=o, high=h, low=l, close=c, volume=v
                    ))
                else:
                    changed = (
                        self._diff(obj.open, o) or
                        self._diff(obj.high, h) or
                        self._diff(obj.low, l) or
                        self._diff(obj.close, c) or
                        self._diff(obj.volume, v)
                    )
                    if changed:
                        obj.open = o
                        obj.high = h
                        obj.low = l
                        obj.close = c
                        obj.volume = v
                        to_update.append(obj)

            if to_create:
                CoinAPIPrice.objects.bulk_create(to_create, ignore_conflicts=True)
                total_created += len(to_create)

            if to_update:
                CoinAPIPrice.objects.bulk_update(
                    to_update, fields=["open", "high", "low", "close", "volume"]
                )
                total_updated += len(to_update)

            first_dt = self._from_ms(int(klines[0][0]))
            last_dt = self._from_ms(int(klines[-1][0]))
            self.stdout.write(
                f"✅ {len(desired)} candles | +{len(to_create)} new, ~{len(to_update)} updated | {first_dt} → {last_dt}"
            )

            # Next window starts at next candle after the last returned
            start_ms = int(klines[-1][0]) + INTERVAL_MS
            time.sleep(0.1)

        self.stdout.write(self.style.SUCCESS(
            f"🎉 Done. Created: {total_created} | Updated: {total_updated}"
        ))
