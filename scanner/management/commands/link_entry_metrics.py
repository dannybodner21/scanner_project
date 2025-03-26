from django.core.management.base import BaseCommand
from scanner.models import SuccessfulMove, Metrics
from datetime import timedelta

class Command(BaseCommand):
    help = "Link SuccessfulMove entries to the closest matching Metrics entry"

    def handle(self, *args, **kwargs):
        linked = 0

        for move in SuccessfulMove.objects.all():
            if not move.timestamp:
                continue

            closest = Metrics.objects.filter(
                coin=move.coin,
                timestamp__range=(
                    move.timestamp - timedelta(minutes=10),
                    move.timestamp + timedelta(minutes=10)
                )
            ).order_by('timestamp').first()

            if closest:
                move.entry_metrics = closest
                move.save()
                linked += 1

        print(f"✅ Linked {linked} SuccessfulMove entries to Metrics.")
