from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
import requests
import time
from scanner.models import CoinAPIPrice

class Command(BaseCommand):
    help = 'Efficiently backfills 5-minute candles for GRT-USD from Coinbase from Jan 1, 2022 to present'



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




    def handle(self, *args, **options):
        product_id = 'HBAR-USD'
        granularity = 300  # 5 minutes in seconds
        max_candles = 300  # max Coinbase returns per request
        base_url = 'https://api.exchange.coinbase.com/products'
        
        start_time = datetime(2022, 1, 1)
        #start_time = datetime(2024, 7, 1)
        end_time = datetime.utcnow()

        total_inserted = 0
        total_iterations = int(((end_time - start_time).total_seconds()) // (granularity * max_candles)) + 1
        loop_start = start_time
        iteration = 1

        while loop_start < end_time:
            loop_end = min(loop_start + timedelta(seconds=granularity * max_candles), end_time)
            iso_start = loop_start.isoformat() + 'Z'
            iso_end = loop_end.isoformat() + 'Z'

            url = f"{base_url}/{product_id}/candles?granularity={granularity}&start={iso_start}&end={iso_end}"
            response = requests.get(url)

            if response.status_code != 200:
                self.stderr.write(f"Error: {response.status_code} {response.text}")
                time.sleep(1)
                continue

            candles = response.json()
            objs = []
            existing_timestamps = set(
                CoinAPIPrice.objects.filter(
                    coin='HBARUSDT', timestamp__range=(make_aware(loop_start), make_aware(loop_end))
                ).values_list('timestamp', flat=True)
            )

            for candle in candles:
                timestamp = make_aware(datetime.utcfromtimestamp(candle[0]))
                if timestamp in existing_timestamps:
                    continue

                low = candle[1]
                high = candle[2]
                open_price = candle[3]
                close_price = candle[4]
                volume = candle[5]

                objs.append(CoinAPIPrice(
                    coin='HBARUSDT',
                    timestamp=timestamp,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close_price,
                    volume=volume,
                ))

            if objs:
                CoinAPIPrice.objects.bulk_create(objs, ignore_conflicts=True)
                total_inserted += len(objs)

            progress = (iteration / total_iterations) * 100
            self.stdout.write(f"[{iteration}/{total_iterations}] {loop_start.date()} to {loop_end.date()} - inserted: {total_inserted} ({progress:.2f}% done)")
            iteration += 1
            loop_start = loop_end
            time.sleep(0.1)  # minimal rate limit protection

        self.stdout.write(self.style.SUCCESS(f"Backfill complete. Total candles inserted: {total_inserted}"))
