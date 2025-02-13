from django.core.management.base import BaseCommand
from datetime import datetime
import logging
from scanner.views import load_coins, fetch_short_interval_data, gather_daily_historical_data


logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Run the scheduled task manually"

    def handle(self, *args, **kwargs):

        logger.info(f"Doing the task...")
        #load_coins()
        #logger.info(f"Coins loaded.")
        fetch_short_interval_data()
        logger.info(f"Step two completed.")
        #gather_daily_historical_data()
        #analyze_historical_metrics()
        logger.info("Task completed.")
