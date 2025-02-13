from django.core.management.base import BaseCommand
from datetime import datetime
import logging
from scanner.views import initial_setup_one, initial_setup_two, initial_setup_three, initial_setup_four, initial_setup_five, initial_setup_six, initial_setup_seven, load_coins, fetch_short_interval_data, gather_daily_historical_data

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Run the scheduled task manually"

    def handle(self, *args, **kwargs):

        logger.info(f"Doing the task...")
        #load_coins()
        #logger.info(f"Coins loaded.")

        initial_setup_one()
        #initial_setup_two()
        #initial_setup_three()
        #initial_setup_four()
        #initial_setup_five()
        #initial_setup_six()
        #initial_setup_seven()

        #fetch_short_interval_data()
        logger.info(f"Short Interval Data and Metrics completed.")

        #gather_daily_historical_data()
        #analyze_historical_metrics()
        logger.info("Task completed.")
