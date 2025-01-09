from django.core.management.base import BaseCommand
from datetime import datetime
import logging

from scanner.views import fetch_solana_meme_coins

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Run the scheduled task manually"

    def handle(self, *args, **kwargs):

        logger.info(f"Fetching meme coins...")
        fetch_solana_meme_coins()
        logger.info("Task completed.")
