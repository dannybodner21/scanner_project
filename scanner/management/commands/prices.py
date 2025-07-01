# management/commands/strict_backfill.py

import os
import time
import requests
import datetime
from decimal import Decimal
from django.core.management.base import BaseCommand
from django.utils import timezone
from scanner.models import CoinAPIPrice

class Command(BaseCommand):
    help = "Strict 5-min backfill for multiple coins using CoinAPI"

    COINS = [
        ("BINANCE_SPOT_BTC_USDT", "BTCUSDT"),
        ("BINANCE_SPOT_ETH_USDT", "ETHUSDT"),
        ("BINANCE_SPOT_XRP_USDT", "XRPUSDT"),
        ("BINANCE_SPOT_LTC_USDT", "LTCUSDT"),
        ("BINANCE_SPOT_SOL_USDT", "SOLUSDT"),
        ("BINANCE_SPOT_DOGE_USDT", "DOGEUSDT"),
        ("BINANCE_SPOT_LINK_USDT", "LINKUSDT"),
        ("BINANCE_SPOT_DOT_USDT", "DOTUSDT"),
        ("BINANCE_SPOT_SHIB_USDT", "SHIBUSDT"),
        ("BINANCE_SPOT_ADA_USDT", "ADAUSDT"),
        ("BINANCE_SPOT_UNI_USDT", "UNIUSDT"),
        ("BINANCE_SPOT_AVAX_USDT", "AVAXUSDT"),
        ("BINANCE_SPOT_XLM_USDT", "XLMUSDT"),
    ]

    def handle(self, *args, **kwargs):
        api_key = "01293e2a-dcf1-4e81-8310-c6aa9d0cb743"
        headers = {"X-CoinAPI-Key": api_key}

        start_time = datetime.datetime(2025, 6, 30, 23, 0, 0, tzinfo=datetime.timezone.utc)
        end_time = datetime.datetime(2025, 7, 1, 19, 00, 0, tzinfo=datetime.timezone.utc)

        for coinapi_symbol, db_coin in self.COINS:
            self.stdout.write(f"\n🔁 Starting {db_coin}...")

            existing_timestamps = set(
                CoinAPIPrice.objects.filter(coin=db_coin).values_list("timestamp", flat=True)
            )
            self.stdout.write(f"📦 Loaded {len(existing_timestamps)} existing rows.")

            current_day = start_time.date()

            while current_day <= end_time.date():
                day_start = datetime.datetime.combine(current_day, datetime.time.min, tzinfo=datetime.timezone.utc)
                day_end = day_start + datetime.timedelta(days=1)

                batch_start_str = day_start.strftime('%Y-%m-%dT%H:%M:%S')
                batch_end_str = day_end.strftime('%Y-%m-%dT%H:%M:%S')

                url = (
                    f"https://rest.coinapi.io/v1/ohlcv/{coinapi_symbol}/history?"
                    f"period_id=5MIN"
                    f"&time_start={batch_start_str}"
                    f"&time_end={batch_end_str}"
                    f"&limit=10000"
                )

                self.stdout.write(f"📆 Fetching {current_day} for {db_coin}...")

                try:
                    resp = requests.get(url, headers=headers)
                    if resp.status_code != 200:
                        self.stdout.write(f"❌ API error {resp.status_code}: {resp.text}")
                        time.sleep(2)
                        current_day += datetime.timedelta(days=1)
                        continue

                    data = resp.json()
                    returned_timestamps = set()
                    records_to_insert = []

                    for item in data:
                        raw_ts = item['time_period_start'][:19]
                        ts = datetime.datetime.fromisoformat(raw_ts).replace(tzinfo=datetime.timezone.utc)
                        returned_timestamps.add(ts)

                        if ts not in existing_timestamps:
                            record = CoinAPIPrice(
                                coin=db_coin,
                                timestamp=ts,
                                open=Decimal(item['price_open']),
                                high=Decimal(item['price_high']),
                                low=Decimal(item['price_low']),
                                close=Decimal(item['price_close']),
                                volume=Decimal(item['volume_traded']),
                            )
                            records_to_insert.append(record)
                            existing_timestamps.add(ts)

                    if records_to_insert:
                        CoinAPIPrice.objects.bulk_create(records_to_insert, batch_size=500)
                        self.stdout.write(f"✅ Inserted {len(records_to_insert)} candles for {current_day}")
                    else:
                        self.stdout.write(f"✅ All candles already exist for {current_day}")

                    # Gap check
                    expected_timestamps = set(
                        day_start + datetime.timedelta(minutes=5 * i)
                        for i in range(288)
                    )
                    missing = expected_timestamps - returned_timestamps

                    if missing:
                        self.stdout.write(f"⚠ {len(missing)} missing candles. Attempting gap fills...")
                        for ts in sorted(missing):
                            self.fetch_single_candle(ts, coinapi_symbol, db_coin, headers, existing_timestamps)

                    time.sleep(1)

                except Exception as e:
                    self.stdout.write(f"❌ Exception: {e}")
                    time.sleep(2)

                current_day += datetime.timedelta(days=1)

        self.stdout.write("🚀 Backfill for all coins complete.")

    def fetch_single_candle(self, ts, coinapi_symbol, db_coin, headers, existing_timestamps):
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
                self.stdout.write(f"❌ Single pull API error {resp.status_code}: {resp.text}")
                time.sleep(1)
                return

            data = resp.json()
            if not data:
                self.stdout.write(f"⚠ No data for {ts_start}")
                return

            item = data[0]
            if ts not in existing_timestamps:
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
                existing_timestamps.add(ts)
                self.stdout.write(f"✅ Filled missing candle for {ts_start}")

        except Exception as e:
            self.stdout.write(f"❌ Exception during gap fill: {e}")
            time.sleep(1)
