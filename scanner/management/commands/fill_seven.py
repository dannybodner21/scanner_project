from datetime import datetime, timedelta
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics, Coin
import requests
import time

CMC_API_KEY = '6520549c-03bb-41cd-86e3-30355ece87ba'
HEADERS = {"Accepts": "application/json", "X-CMC_PRO_API_KEY": CMC_API_KEY}
CMC_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"

RATE_LIMIT_CALLS_PER_MIN = 25
SECONDS_BETWEEN_CALLS = 60.0 / RATE_LIMIT_CALLS_PER_MIN

class Command(BaseCommand):
    help = 'Efficiently fill missing change_1h and change_24h from CMC for RickisMetrics (April 22–May 13)'

    def handle(self, *args, **kwargs):
        symbols = [
            "BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LINK",
            "AVAX", "XLM", "TON", "SHIB", "SUI", "HBAR", "BCH", "DOT", "LTC",
            "XMR", "UNI", "PEPE", "APT", "NEAR", "ONDO", "TAO", "ICP", "ETC",
            "RENDER", "MNT", "KAS", "CRO", "AAVE", "POL", "VET", "FIL", "ALGO",
            "ENA", "ATOM", "TIA", "ARB", "DEXE", "OP", "JUP", "MKR", "STX",
            "EOS", "WLD", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT",
            "PAXG", "CRV", "JASMY", "SAND", "GALA", "CORE", "KAIA", "LDO",
            "THETA", "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI"
        ]

        start_date = datetime(2025, 4, 13)
        end_date = datetime(2025, 5, 13)

        for symbol in symbols:
            try:
                coin = Coin.objects.get(symbol=symbol)
            except Coin.DoesNotExist:
                print(f"❌ Coin not found: {symbol}")
                continue

            print(f"\n🚀 Updating {symbol}")

            current_day = start_date
            while current_day <= end_date:
                start_ts = int(make_aware(datetime.combine(current_day, datetime.min.time())).timestamp())
                end_ts = start_ts + 86400

                params = {
                    "symbol": symbol,
                    "time_start": start_ts,
                    "time_end": end_ts,
                    "interval": "5m",
                    "convert": "USD"
                }

                try:
                    response = requests.get(CMC_URL, headers=HEADERS, params=params)
                    data = response.json()
                    quotes = data.get("data", {}).get("quotes", [])

                    if not quotes:
                        print(f"⚠️ No data for {symbol} on {current_day.date()}")
                        current_day += timedelta(days=1)
                        time.sleep(SECONDS_BETWEEN_CALLS)
                        continue

                    # Build list of timestamp:quote
                    quote_map = []
                    for q in quotes:
                        try:
                            ts_str = q["timestamp"]
                            ts = make_aware(datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%fZ"))
                            quote_map.append((ts, q["quote"]["USD"]))
                        except Exception:
                            continue

                    # Get RickisMetrics for the day
                    metrics = list(RickisMetrics.objects.filter(
                        coin=coin,
                        timestamp__gte=make_aware(datetime.combine(current_day, datetime.min.time())),
                        timestamp__lt=make_aware(datetime.combine(current_day + timedelta(days=1), datetime.min.time())),
                    ))

                    updated = 0
                    for metric in metrics:
                        if metric.change_1h not in [None, 0] and metric.change_24h not in [None, 0]:
                            continue

                        # Find closest quote (within 5 min)
                        best_quote = None
                        min_diff = timedelta(minutes=6)
                        for ts, quote in quote_map:
                            diff = abs(ts - metric.timestamp)
                            if diff < min_diff:
                                min_diff = diff
                                best_quote = quote

                        if not best_quote:
                            continue

                        change_1h = best_quote.get("percent_change_1h")
                        change_24h = best_quote.get("percent_change_24h")

                        if change_1h is not None and metric.change_1h in [None, 0]:
                            metric.change_1h = change_1h
                        if change_24h is not None and metric.change_24h in [None, 0]:
                            metric.change_24h = change_24h

                        updated += 1

                    # Save all at once
                    RickisMetrics.objects.bulk_update(metrics, ["change_1h", "change_24h"])
                    print(f"✅ {symbol} on {current_day.date()} — {updated} updated")

                except Exception as e:
                    print(f"❌ Error for {symbol} on {current_day.date()}: {e}")

                current_day += timedelta(days=1)
                time.sleep(SECONDS_BETWEEN_CALLS)

        print("\n🎉 All updates complete.")
