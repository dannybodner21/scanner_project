from django.core.management.base import BaseCommand
from datetime import datetime
import logging
from scanner.views import delete_old_data_custom

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Run the scheduled task manually"

    def handle(self, *args, **kwargs):

        logger.info(f"doing shit...")
        delete_old_data_custom()
        logger.info("done.")
