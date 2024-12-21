from django.core.management.base import BaseCommand
from datetime import datetime
import logging


logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Run the scheduled task manually'

    def handle(self, *args, **kwargs):
        logger.info(f'Running scheduled task at {datetime.now()}')
        from scanner.tasks import scheduled_task_function
        scheduled_task_function()
        logger.info("Scheduled task completed.")
