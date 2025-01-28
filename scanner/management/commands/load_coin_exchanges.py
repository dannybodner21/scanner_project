from django.core.management.base import BaseCommand
from datetime import datetime
import logging
from scanner.views import load_coin_exchanges, test_message

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Run the scheduled task manually"

    def handle(self, *args, **kwargs):

        logger.info(f"Doing the task...")
        #load_coin_exchanges()
        test_message()
        logger.info("Task completed.")




'''
# check for new high or low and update the daily
# make sure there is historical data
existing_data = HistoricalData.objects.filter(coin=coin, date=date).exists()

if (existing_data):
    daily_data = HistoricalData.objects.get(coin=coin, date=date)
    previous_high = daily_data.daily_high
    previous_low = daily_data.daily_low
    if (current_price > previous_high):
        # update daily high
        daily_data.daily_high = current_price
        daily_data.save()

    if (current_price < previous_low):
        # update daily low
        daily_data.daily_low = current_price
        daily_data.save()
'''
