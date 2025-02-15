from django.core.management.base import BaseCommand
from datetime import datetime
import logging
from scanner.views import calculate_all_metrics, initial_setup_one, initial_setup_two, initial_setup_three, initial_setup_four, initial_setup_five, initial_setup_six, initial_setup_final, load_coins, fetch_short_interval_data, gather_daily_historical_data

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Run the scheduled task manually"

    def handle(self, *args, **kwargs):

        logger.info(f"Doing the task...")
        #load_coins()
        #logger.info(f"Coins loaded.")

        #initial_setup_one()
        #logger.info(f"Doing setup two...")
        #initial_setup_two()
        #logger.info(f"Doing setup three...")
        #initial_setup_three()
        #logger.info(f"Doing setup four...")
        #initial_setup_four()
        #logger.info(f"Doing setup five...")
        #initial_setup_five()
        #logger.info(f"Doing setup six...")
        #initial_setup_six()

        logger.info(f"Doing setup final...")
        #initial_setup_final()

        calculate_all_metrics()

        #fetch_short_interval_data()
        logger.info(f"Short Interval Data and Metrics completed.")

        #gather_daily_historical_data()
        #analyze_historical_metrics()
        logger.info("Task completed.")
