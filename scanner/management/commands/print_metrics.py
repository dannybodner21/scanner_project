from django.core.management.base import BaseCommand
from datetime import datetime, timedelta, timezone
import logging
from scanner.views import print_metrics


logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Run the scheduled task manually"


    def add_arguments(self, parser):

        parser.add_argument(
            '--coin_symbol',
            type=str,
            required=True,
            help="The name of the coin symbol to analyze."
        )


    def handle(self, *args, **kwargs):

        coin_symbol = kwargs['coin_symbol']

        logger.info(f"Analyzing data...")
        print_metrics(coin_symbol)
        logger.info("Task completed.")
