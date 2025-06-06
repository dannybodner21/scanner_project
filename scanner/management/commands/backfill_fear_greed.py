import requests
import pandas as pd
from datetime import datetime
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics

# nohup python manage.py backfill_fear_greed > output.log 2>&1 &
# tail -f output.log

class Command(BaseCommand):
    help = "Backfill Fear & Greed Index for RickisMetrics from March 23, 2025 to May 23, 2025."

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("🚀 Starting Fear & Greed backfill..."))

        # Step 1: Pull full historical Fear & Greed data
        url = "https://api.alternative.me/fng/?limit=0&format=json"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()['data']
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ Error fetching Fear & Greed Index: {e}"))
            return

        # Step 2: Convert to DataFrame
        fg_df = pd.DataFrame(data)
        fg_df['timestamp'] = pd.to_datetime(fg_df['timestamp'], unit='s').dt.date
        fg_df['value'] = fg_df['value'].astype(int)

        # Step 3: Filter for date range
        start_date = datetime(2025, 3, 23).date()
        end_date = datetime(2025, 5, 23).date()
        fg_df = fg_df[(fg_df['timestamp'] >= start_date) & (fg_df['timestamp'] <= end_date)]

        # Step 4: Build map {date: value}
        fg_map = dict(zip(fg_df['timestamp'], fg_df['value']))

        # Step 5: Query RickisMetrics in date range
        metrics = RickisMetrics.objects.filter(
            timestamp__date__gte=start_date,
            timestamp__date__lte=end_date
        )

        total = metrics.count()
        self.stdout.write(f"🔍 Found {total} RickisMetrics records to update.")

        # Step 6: Update records
        updated_count = 0
        for metric in metrics.iterator(chunk_size=1000):
            fg_value = fg_map.get(metric.timestamp.date())
            if fg_value is not None:
                metric.fear_greed = fg_value
                metric.save(update_fields=['fear_greed_index'])
                updated_count += 1
            print(f'updated: {updated_count}')

        self.stdout.write(self.style.SUCCESS(f"✅ Successfully updated {updated_count} RickisMetrics with Fear & Greed Index."))
