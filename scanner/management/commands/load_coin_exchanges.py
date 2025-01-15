from django.core.management.base import BaseCommand
from datetime import datetime
import logging
from scanner.views import load_coin_exchanges, test_message

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Run the scheduled task manually"

    def handle(self, *args, **kwargs):

        logger.info(f"Doing the task...")
        #load_coin_exchanges()
        test_message()
        logger.info("Task completed.")
