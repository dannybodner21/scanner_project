from django.core.management.base import BaseCommand
from datetime import datetime
import logging


logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Run the scheduled task manually"

    def handle(self, *args, **kwargs):

        logger.info(f"Doing the task...")

        from scanner.views import update_historical_data, load_coins, fetch_short_interval_data, analyze_historical_metrics, gather_daily_historical_data

        #load_coins()
        #fetch_short_interval_data()
        #gather_daily_historical_data()

        #analyze_historical_metrics()

        update_historical_data()

        logger.info("Task completed.")
