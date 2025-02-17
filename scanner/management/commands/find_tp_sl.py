from django.core.management.base import BaseCommand
from datetime import datetime
import logging

from scanner.views import find_tp_sl, hourly_candles

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Run the scheduled task manually"

    def handle(self, *args, **kwargs):

        logger.info(f"Working...")
        #find_tp_sl()
        hourly_candles()
        logger.info("Done.")
