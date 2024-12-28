from django.core.management.base import BaseCommand
from datetime import datetime
import logging
from scanner.views import manually_clean_database

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Run the scheduled task manually"

    def handle(self, *args, **kwargs):

        logger.info(f"Cleaning the database...")
        manually_clean_database()
        logger.info("Database cleaning completed.")
