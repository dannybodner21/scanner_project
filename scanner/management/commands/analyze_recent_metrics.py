from django.core.management.base import BaseCommand
from datetime import datetime
import logging
from scanner.views import analyze_recent_metrics


logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Run the scheduled task manually"

    def handle(self, *args, **kwargs):

        logger.info(f"Analyzing data...")
        analyze_recent_metrics()
        logger.info("Task completed.")
