from django.core.management.base import BaseCommand
from datetime import datetime
import logging

from scanner.views import do_market_research

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Run the scheduled task manually"

    def handle(self, *args, **kwargs):

        logger.info(f"Finding the metrics...")
        do_market_research()
        logger.info("Task completed.")
