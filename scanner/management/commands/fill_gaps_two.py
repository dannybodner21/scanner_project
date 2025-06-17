import os
import time
import requests
import datetime
from decimal import Decimal
from django.core.management.base import BaseCommand
from django.utils import timezone
from scanner.models import CoinAPIPrice

class Command(BaseCommand):
    help = "Final gap filler using multiple CoinAPI sources"

    def handle(self, *args, **kwargs):
        db_coin = "ADAUSDT"
        fallback_symbols = [
            "COINBASE_SPOT_ADA_USD",
            "KRAKEN_SPOT_ADA_USD",
            "KUCOIN_SPOT_ADA_USDT",
            "BITFINEX_SPOT_ADA_USD"
        ]

        api_key = "01293e2a-dcf1-4e81-8310-c6aa9d0cb743"
        headers = {"X-CoinAPI-Key": api_key}

        start_time = datetime.datetime(2020, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        end_time = datetime.datetime(2025, 6, 13, 23, 55, 0, tzinfo=datetime.timezone.utc)

        self.stdout.write("Loading existing timestamps...")
        existing_timestamps = set(
            CoinAPIPrice.objects.filter(coin=db_coin).values_list("timestamp", flat=True)
        )

        total_expected = set()
        current = start_time
        while current <= end_time:
            total_expected.add(current)
            current += datetime.timedelta(minutes=5)

        gaps = sorted(total_expected - existing_timestamps)
        self.stdout.write(f"Found {len(gaps)} missing timestamps.")

        for ts in gaps:
            filled = False
            for coinapi_symbol in fallback_symbols:
                if self.fetch_single_candle(ts, coinapi_symbol, db_coin, headers):
                    filled = True
                    break

            if not filled:
                self.stdout.write(f"❌ Still missing after all sources: {ts}")

            time.sleep(0.2)  # short delay to avoid API hammering

        self.stdout.write("✅ Gap filling complete.")

    def fetch_single_candle(self, ts, coinapi_symbol, db_coin, headers):
        ts_start = ts.strftime('%Y-%m-%dT%H:%M:%S')
        ts_end = (ts + datetime.timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%S')

        url = (
            f"https://rest.coinapi.io/v1/ohlcv/{coinapi_symbol}/history?"
            f"period_id=5MIN"
            f"&time_start={ts_start}"
            f"&time_end={ts_end}"
            f"&limit=1"
        )

        try:
            resp = requests.get(url, headers=headers)
            if resp.status_code != 200:
                self.stdout.write(f"❌ API error {resp.status_code} for {coinapi_symbol}: {resp.text}")
                return False

            data = resp.json()
            if not data:
                self.stdout.write(f"⚠ No data for {coinapi_symbol} at {ts_start}")
                return False

            item = data[0]
            record = CoinAPIPrice(
                coin=db_coin,
                timestamp=ts,
                open=Decimal(item['price_open']),
                high=Decimal(item['price_high']),
                low=Decimal(item['price_low']),
                close=Decimal(item['price_close']),
                volume=Decimal(item['volume_traded']),
            )
            record.save()
            self.stdout.write(f"✅ Filled {ts_start} from {coinapi_symbol}")
            return True

        except Exception as e:
            self.stdout.write(f"❌ Exception during fetch: {e}")
            return False
