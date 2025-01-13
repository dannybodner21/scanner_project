from django.core.management.base import BaseCommand
from datetime import datetime
import logging

from scanner.views import check_new_solana_listings

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Run the scheduled task manually"

    def handle(self, *args, **kwargs):

        logger.info(f"Fetching meme coins...")
        check_new_solana_listings()
        logger.info("Task completed.")
