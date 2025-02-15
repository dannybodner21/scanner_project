from django.core.management.base import BaseCommand
from datetime import datetime
import logging

from scanner.views import trigger_combination

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Run the scheduled task manually"

    def handle(self, *args, **kwargs):

        logger.info(f"Working...")
        trigger_combination()
        logger.info("Done.")
