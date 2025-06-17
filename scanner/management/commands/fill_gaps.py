

import datetime
import time
import requests
from decimal import Decimal
from django.core.management.base import BaseCommand
from django.utils import timezone
from scanner.models import CoinAPIPrice

class Command(BaseCommand):
    help = "Patch missing 5min candles using Polygon.io"

    def handle(self, *args, **kwargs):
        coin = "TRXUSDT"
        polygon_api_key = "qq9Sptr4VfkonQimqFJEgc3oyXoaJ54L"

        start_time = datetime.datetime(2020, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        end_time = datetime.datetime(2025, 6, 13, 23, 55, 0, tzinfo=datetime.timezone.utc)

        self.stdout.write("Loading all existing timestamps into memory...")
        existing = set(
            CoinAPIPrice.objects
            .filter(coin=coin)
            .values_list("timestamp", flat=True)
        )

        self.stdout.write(f"Loaded {len(existing)} existing candles.")

        missing_timestamps = []

        current = start_time
        while current <= end_time:
            if current not in existing:
                missing_timestamps.append(current)
            current += datetime.timedelta(minutes=5)

        self.stdout.write(f"Found {len(missing_timestamps)} missing 5-min candles.")

        # Group by day
        missing_by_day = {}
        for ts in missing_timestamps:
            day = ts.date()
            if day not in missing_by_day:
                missing_by_day[day] = []
            missing_by_day[day].append(ts)

        for day, timestamps in missing_by_day.items():
            day_start = datetime.datetime.combine(day, datetime.time.min, tzinfo=datetime.timezone.utc)
            day_end = datetime.datetime.combine(day, datetime.time.max, tzinfo=datetime.timezone.utc)

            url = (
                f"https://api.polygon.io/v2/aggs/ticker/X:TRXUSD/range/5/minute/"
                f"{int(day_start.timestamp())}/{int(day_end.timestamp())}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}"
            )

            self.stdout.write(f"Fetching data for {day} from Polygon...")

            try:
                resp = requests.get(url)
                if resp.status_code != 200:
                    self.stdout.write(f"❌ Polygon API error {resp.status_code}: {resp.text}")
                    time.sleep(1)
                    continue

                data = resp.json()
                results = data.get("results", [])
                polygon_data = {}

                for item in results:
                    ts = datetime.datetime.utcfromtimestamp(item["t"] / 1000).replace(tzinfo=datetime.timezone.utc)
                    polygon_data[ts] = item

                for ts in timestamps:
                    candle = polygon_data.get(ts)
                    if candle:
                        CoinAPIPrice.objects.create(
                            coin=coin,
                            timestamp=ts,
                            open=Decimal(candle["o"]),
                            high=Decimal(candle["h"]),
                            low=Decimal(candle["l"]),
                            close=Decimal(candle["c"]),
                            volume=Decimal(candle["v"]),
                        )
                        self.stdout.write(f"✅ Filled {ts}")
                    else:
                        self.stdout.write(f"⚠ No Polygon data for {ts} (still missing)")

                time.sleep(0.5)

            except Exception as e:
                self.stdout.write(f"❌ Exception during Polygon pull: {e}")
                time.sleep(1)

        self.stdout.write("🚀 Gap patch complete.")
