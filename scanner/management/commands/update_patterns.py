import os
import requests
import time
from django.core.management.base import BaseCommand
from scanner.models import Coin, Pattern

FINNHUB_API_KEY = "cuf7nohr01qno7m552hgcuf7nohr01qno7m552i0"

class Command(BaseCommand):
    help = "Fetch pattern data from Finnhub and update existing Pattern entries"

    def handle(self, *args, **kwargs):
        coins = Coin.objects.filter(symbol__in=[
            "BTC", "ETH", "XRP", "LTC", "SOL", "DOGE", "PEPE", "ADA",
            "XLM", "SUI", "LINK", "AVAX", "DOT", "SHIB", "HBAR", "UNI"
        ])

        resolutions = [5, 15, 60]

        for coin in coins:
            symbol = f"BINANCE:{coin.symbol}USDT"
            for resolution in resolutions:
                url = f"https://finnhub.io/api/v1/scan/pattern?symbol={symbol}&resolution={resolution}&token={FINNHUB_API_KEY}"

                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    data = response.json()

                    # Load the existing pre-created row
                    pattern_obj = Pattern.objects.get(coin=coin, resolution=resolution)

                    if data.get("points"):
                        latest = data["points"][0]

                        pattern_obj.symbol = symbol
                        pattern_obj.patterntype = latest.get("patterntype")
                        pattern_obj.patternname = latest.get("patternname")
                        pattern_obj.status = latest.get("status")
                        pattern_obj.entry = latest.get("entry")
                        pattern_obj.takeprofit = latest.get("profit1")
                        pattern_obj.stoploss = latest.get("stoploss")
                        pattern_obj.adx = latest.get("adx")

                        pattern_obj.save()
                        self.stdout.write(f"✅ {coin.symbol} {resolution}min pattern updated: {pattern_obj.patternname}")
                    else:
                        self.stdout.write(f"⚠ No pattern found for {coin.symbol} at {resolution}min (left unchanged)")

                except Pattern.DoesNotExist:
                    self.stdout.write(f"❌ Template row missing for {coin.symbol} {resolution}min — create template rows first!")

                except Exception as e:
                    self.stdout.write(f"❌ Error for {coin.symbol} at {resolution}min: {e}")

                time.sleep(0.25)

        self.stdout.write("🚀 FINISHED updating patterns")
