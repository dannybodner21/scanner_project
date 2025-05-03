from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from datetime import timedelta

class Command(BaseCommand):
    
    help = "Delete RickisMetrics entries not aligned to exact 5-minute timestamps"

    def handle(self, *args, **kwargs):
        print("🔍 Checking for RickisMetrics on invalid timestamps...")

        bad_entries = RickisMetrics.objects.all().iterator()
        to_delete = []

        for entry in bad_entries:
            ts = entry.timestamp
            if (
                ts.minute % 5 != 0
                or ts.second != 0
                or ts.microsecond != 0
            ):
                to_delete.append(entry.id)

        total = len(to_delete)
        print(f"🗑️ Found {total} bad entries to delete.")

        confirm = input("⚠️ Type YES to confirm deletion: ")
        if confirm == "YES":
            RickisMetrics.objects.filter(id__in=to_delete).delete()
            print("✅ Deleted.")
        else:
            print("❌ Deletion canceled.")
