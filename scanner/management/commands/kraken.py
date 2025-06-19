from django.core.management.base import BaseCommand
import krakenex
from pykrakenapi import KrakenAPI

class Command(BaseCommand):
    help = "Test Kraken API connection and fetch balance"

    def handle(self, *args, **kwargs):

        api = krakenex.API()
        api.key = API_KEY
        api.secret = API_SECRET

        kraken = KrakenAPI(api)

        try:
            #balance = kraken.get_account_balance()
            #self.stdout.write(self.style.SUCCESS("Connected to Kraken API"))
            #for asset, amount in balance.items():
                #self.stdout.write(f"{asset}: {amount}")

            # Margin trade settings
            pair = 'XBTUSD'         # Kraken uses 'XBT' not 'BTC'
            volume = '0.001'        # Small size for safety
            leverage = '5'          # Max for BTC/USD

            order_payload = {
                'pair': pair,
                'type': 'buy',
                'ordertype': 'market',
                'volume': volume,
                'leverage': leverage,
            }

            self.stdout.write("🔁 Simulated Kraken Margin Order:")
            for k, v in order_payload.items():
                self.stdout.write(f"{k}: {v}")





        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Failed to connect: {e}"))
