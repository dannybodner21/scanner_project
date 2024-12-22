from django.core.management.base import BaseCommand
from datetime import datetime
import logging


logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Run the scheduled task manually"

    def handle(self, *args, **kwargs):

        logger.info(f"Doing the task...")

        from scanner.tasks import scheduled_task_function
        from scanner.views import load_coins, gather_historical_data, fetch_short_interval_data
        #scheduled_task_function()
        load_coins()
        #gather_historical_data()
        fetch_short_interval_data()

        logger.info("Task completed.")
