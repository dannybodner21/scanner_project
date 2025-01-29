from django.core.management.base import BaseCommand
from datetime import datetime
import logging

from scanner.views import find_best_trigger, brute_force, brute_force_one

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Run the scheduled task manually"

    def handle(self, *args, **kwargs):

        logger.info(f"Working...")
        #find_best_trigger()
        brute_force()
        #brute_force_one()
        logger.info("Done.")
