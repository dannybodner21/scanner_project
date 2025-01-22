from django.core.management.base import BaseCommand
from datetime import datetime
import logging

from scanner.views import check_trigger_two

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Run the scheduled task manually"

    def handle(self, *args, **kwargs):

        logger.info(f"Working...")
        check_trigger_two()
        logger.info("Done.")
