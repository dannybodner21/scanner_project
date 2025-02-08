from django.core.management.base import BaseCommand
from datetime import datetime
import logging

from scanner.views import finn, finn_test, five_min_pattern_check, thirty_min_pattern_check, pattern_recognition, daily_high_low_data, create_main_csv, find_best_trigger, brute_force, brute_force_one, brute_force_short

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Run the scheduled task manually"

    def handle(self, *args, **kwargs):

        logger.info(f"Working...")
        #find_best_trigger()
        #brute_force()
        #brute_force_one()
        #brute_force_short()
        #create_main_csv()
        #finn()
        #daily_high_low_data()
        #pattern_recognition()
        #thirty_min_pattern_check()
        #five_min_pattern_check()
        finn_test()
        logger.info("Done.")
