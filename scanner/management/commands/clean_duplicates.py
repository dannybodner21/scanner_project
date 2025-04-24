from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
from scanner.models import Coin, ShortIntervalData

class Command(BaseCommand):
    help = 'Clean up ShortIntervalData for a specific coin (starting with BTC), ensuring exactly 288 entries per day with 5-minute aligned timestamps.'

    def handle(self, *args, **kwargs):
        start_date = make_aware(datetime(2025, 3, 20))
        end_date = make_aware(datetime(2025, 4, 23))  # exclusive

        def round_to_five_minutes(dt):
            return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)

        # Limit to BTC only for now
        try:
            coin = Coin.objects.get(symbol="BTC")
        except Coin.DoesNotExist:
            self.stdout.write("❌ BTC coin not found in the database.")
            return

        self.stdout.write(f"\n🔍 Cleaning data for: {coin.symbol}")
        current = start_date

        while current < end_date:
            next_day = current + timedelta(days=1)
            entries = ShortIntervalData.objects.filter(
                coin=coin,
                timestamp__gte=current,
                timestamp__lt=next_day
            )

            grouped = {}
            to_delete = []

            for entry in entries:
                original_ts = entry.timestamp
                rounded_ts = round_to_five_minutes(original_ts)

                if original_ts != rounded_ts:
                    if not ShortIntervalData.objects.filter(coin=coin, timestamp=rounded_ts).exists():
                        entry.timestamp = rounded_ts
                        entry.save()
                        self.stdout.write(f"🕒 Updated {coin.symbol} timestamp from {original_ts} -> {rounded_ts}")
                    else:
                        to_delete.append(entry)
                        self.stdout.write(f"❌ Deleted misaligned duplicate at {original_ts}")
                    continue

                if rounded_ts not in grouped:
                    grouped[rounded_ts] = entry
                else:
                    to_delete.append(entry)
                    self.stdout.write(f"❌ Deleted duplicate for {coin.symbol} at {rounded_ts}")

            aligned_entries = list(grouped.values())
            if len(aligned_entries) > 288:
                overflow = aligned_entries[288:]
                to_delete += overflow
                self.stdout.write(f"⚠️  {len(overflow)} overflow entries deleted for {coin.symbol} on {current.date()}")

            for d in to_delete:
                d.delete()

            current = next_day

        self.stdout.write("\n✅ Cleanup complete for BTC.")
