from datetime import timedelta, datetime
import logging
from django.test import RequestFactory
from django_q.models import Schedule


logger = logging.getLogger(__name__)


def scheduled_task_function():

    logger.info("Scheduled task is running...")

    from scanner.views import five_min_update, index

    logger.info("Trying to run function...")
    five_min_update()
    logger.info("Function completed successfully.")

    return "Task completed"


def setup_schedule():

    # Calculate the next 5-minute mark
    now = datetime.now()
    minutes_to_next_five = (5 - now.minute % 5) % 5
    next_run = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_next_five)

    Schedule.objects.update_or_create(
        name='scheduled_task_function',
        defaults={
            'func': 'scanner.tasks.scheduled_task_function',
            'schedule_type': Schedule.MINUTES,
            'minutes': 5,
            'next_run': next_run,
            'repeats': -1,
        }
    )
