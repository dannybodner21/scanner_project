import os
import requests
import time
from django.core.management.base import BaseCommand
from scanner.models import Coin, Pattern

FINNHUB_API_KEY = "YOUR_FINNHUB_API_KEY"

class Command(BaseCommand):
    help = "Fetch pattern data from Finnhub and store to Pattern model"

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

                    if data.get("points"):
                        latest = data["points"][0]

                        patterntype = latest.get("patterntype")
                        patternname = latest.get("patternname")
                        status = latest.get("status")
                        entry = latest.get("entry")
                        takeprofit = latest.get("profit1")
                        stoploss = latest.get("stoploss")
                        adx = latest.get("adx")

                        Pattern.objects.update_or_create(
                            coin=coin,
                            symbol=symbol,
                            resolution=resolution,
                            defaults={
                                "patterntype": patterntype,
                                "patternname": patternname,
                                "status": status,
                                "entry": entry,
                                "takeprofit": takeprofit,
                                "stoploss": stoploss,
                                "adx": adx,
                            }
                        )

                        self.stdout.write(f"✅ {coin.symbol} {resolution}min pattern updated: {patternname}")
                    else:
                        self.stdout.write(f"⚠ No pattern found for {coin.symbol} at {resolution}min")

                except Exception as e:
                    self.stdout.write(f"❌ Error for {coin.symbol} at {resolution}min: {e}")
                    self.stdout.write(data)

                time.sleep(0.25)  # tiny delay to avoid rate limits

        self.stdout.write("🚀 FINISHED updating patterns")
