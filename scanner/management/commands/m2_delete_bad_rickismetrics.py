from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics

class Command(BaseCommand):
    help = "Delete RickisMetrics entries not aligned to 5-minute timestamps"

    def handle(self, *args, **kwargs):
        bad_entries = RickisMetrics.objects.exclude(
            timestamp__minute__in=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
        ).union(
            RickisMetrics.objects.exclude(timestamp__second=0),
            RickisMetrics.objects.exclude(timestamp__microsecond=0)
        )

        total = bad_entries.count()
        print(f"🗑️ Found {total} RickisMetrics entries on incorrect timestamps.")

        confirm = input("⚠️ Are you sure you want to delete them? Type YES to confirm: ")
        if confirm == "YES":
            bad_entries.delete()
            print("✅ Deleted.")
        else:
            print("❌ Aborted.")
