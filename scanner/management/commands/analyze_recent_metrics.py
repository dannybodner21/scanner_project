from django.core.management.base import BaseCommand
from datetime import datetime
import logging
from scanner.views import analyze_recent_metrics


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
        parser.add_argument(
            '--event_time',
            type=str,
            required=True,
            help="The event time in the format 'YYYY-MM-DD HH:MM:SS'."
        )


    def handle(self, *args, **kwargs):

        coin_symbol = kwargs['coin_symbol']
        event_time_str = kwargs['event_time']

        try:
            event_time = datetime.strptime(event_time_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
        except ValueError:
            self.stdout.write(self.style.ERROR("Invalid event_time format. Use 'YYYY-MM-DD HH:MM:SS'."))
            return

        logger.info(f"Analyzing data...")
        analyze_recent_metrics(event_time, coin_symbol)
        logger.info("Task completed.")
