from django_q.tasks import schedule
from datetime import timedelta
import logging
from django.test import RequestFactory


logger = logging.getLogger(__name__)


def scheduled_task_function():
    logger.info("Scheduled task is running...")


    # DO STUFF HERE

    from scanner.views import create_temporary_data, five_min_update, index

    logger.info("Trying to run function...")
    #create_temporary_data()
    five_min_update()

    logger.info("Function completed successfully.")

    return "Task completed"


def setup_schedule():
    schedule(
        func='scanner.tasks.scheduled_task_function',  # Path to the function in this file
        schedule_type='I',  # Interval-based scheduling
        minutes=5,         # Run every 5 minutes
        repeats=-1,         # Run indefinitely
    )
