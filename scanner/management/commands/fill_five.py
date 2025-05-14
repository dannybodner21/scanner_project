from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from scanner.models import RickisMetrics, Coin
import requests
import time

CMC_API_KEY = '6520549c-03bb-41cd-86e3-30355ece87ba'
HEADERS = {"Accepts": "application/json", "X-CMC_PRO_API_KEY": CMC_API_KEY}
CMC_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"

RATE_LIMIT_CALLS_PER_MIN = 25
SECONDS_BETWEEN_CALLS = 60.0 / RATE_LIMIT_CALLS_PER_MIN

class Command(BaseCommand):
    help = 'Fill missing OHLCV values in RickisMetrics using daily data from March 22 to May 12'

    def handle(self, *args, **kwargs):
        start = datetime(2025, 4, 20)
        end = datetime(2025, 5, 12)

        symbols = [
            "BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LINK",
            "AVAX", "XLM", "TON", "SHIB", "SUI", "HBAR", "BCH", "DOT", "LTC",
            "XMR", "UNI", "PEPE", "APT", "NEAR", "ONDO", "TAO", "ICP", "ETC",
            "RENDER", "MNT", "KAS", "CRO", "AAVE", "POL", "VET", "FIL", "ALGO",
            "ENA", "ATOM", "TIA", "ARB", "DEXE", "OP", "JUP", "MKR", "STX",
            "EOS", "WLD", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT",
            "PAXG", "CRV", "JASMY", "SAND", "GALA", "CORE", "KAIA", "LDO",
            "THETA", "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI",
        ]

        total_updates = 0

        for symbol in symbols:
            try:
                coin = Coin.objects.get(symbol=symbol)
            except Coin.DoesNotExist:
                print(f"❌ Coin not found: {symbol}")
                continue

            print(f"\n🔍 Checking {symbol}")
            day = start
            while day <= end:
                day_start = day.strftime("%Y-%m-%d")
                day_end = (day + timedelta(days=1)).strftime("%Y-%m-%d")

                params = {
                    "symbol": symbol,
                    "time_start": day_start,
                    "time_end": day_end,
                    "convert": "USD"
                }

                try:
                    response = requests.get(CMC_URL, headers=HEADERS, params=params)
                    response.raise_for_status()
                    data = response.json()

                    if not data.get("data") or not data["data"].get("quotes"):
                        print(f"⚠️ No data for {symbol} on {day_start}")
                        day += timedelta(days=1)
                        continue

                    daily_quote = data["data"]["quotes"][0]["quote"]["USD"]
                    high = daily_quote.get("high")
                    low = daily_quote.get("low")
                    open_ = daily_quote.get("open")
                    close = daily_quote.get("close")
                    volume = daily_quote.get("volume")

                    # Update all 5-min intervals for that day
                    start_ts = make_aware(datetime.combine(day, datetime.min.time()))
                    end_ts = start_ts + timedelta(days=1)

                    metrics = RickisMetrics.objects.filter(
                        coin=coin,
                        timestamp__gte=start_ts,
                        timestamp__lt=end_ts
                    )

                    for metric in metrics:
                        updated = False
                        if not metric.high_24h or metric.high_24h == 0:
                            metric.high_24h = high
                            updated = True
                        if not metric.low_24h or metric.low_24h == 0:
                            metric.low_24h = low
                            updated = True
                        if not metric.open or metric.open == 0:
                            metric.open = open_
                            updated = True
                        if not metric.close or metric.close == 0:
                            metric.close = close
                            updated = True
                        if not metric.volume or metric.volume == 0:
                            metric.volume = volume
                            updated = True

                        if updated:
                            metric.save()
                            total_updates += 1
                            print(f"✅ Updated {symbol} — {metric.timestamp}")

                except Exception as e:
                    print(f"❌ Failed {symbol} on {day_start}: {e}")

                time.sleep(SECONDS_BETWEEN_CALLS)
                day += timedelta(days=1)

        print(f"\n🎉 Done. Total RickisMetrics updated: {total_updates}")
