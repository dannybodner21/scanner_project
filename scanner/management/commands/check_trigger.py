from django.core.management.base import BaseCommand
from datetime import datetime
import logging

from scanner.views import check_trigger

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Run the scheduled task manually"

    def handle(self, *args, **kwargs):

        logger.info(f"Checking trigger combo...")
        check_trigger()
        logger.info("Done.")
