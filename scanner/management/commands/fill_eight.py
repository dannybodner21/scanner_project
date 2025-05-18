from datetime import datetime, timedelta
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics, Coin
import requests
import time
from django.db.models import Q

CMC_API_KEY = '6520549c-03bb-41cd-86e3-30355ece87ba'
HEADERS = {"Accepts": "application/json", "X-CMC_PRO_API_KEY": CMC_API_KEY}
CMC_HISTORICAL_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"
CMC_LATEST_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"

RATE_LIMIT_CALLS_PER_MIN = 25
SECONDS_BETWEEN_CALLS = 60.0 / RATE_LIMIT_CALLS_PER_MIN

class Command(BaseCommand):
    help = 'Fill missing change_1h and change_24h for RickisMetrics using CMC.'

    def handle(self, *args, **kwargs):
        symbols = [
            "UNI", "PEPE", "APT", "NEAR", "ONDO", "TAO", "ICP", "ETC",
            "RENDER", "MNT", "KAS", "CRO", "AAVE", "POL", "VET", "FIL", "ALGO",
            "ENA", "ATOM", "TIA", "ARB", "DEXE", "OP", "JUP", "MKR", "STX",
            "EOS", "WLD", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT",
            "PAXG", "CRV", "JASMY", "SAND", "GALA", "CORE", "KAIA", "LDO",
            "THETA", "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI"
        ]

        start_date = datetime(2025, 5, 1)
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

                quote_map = []
                try:
                    response = requests.get(CMC_HISTORICAL_URL, headers=HEADERS, params=params)
                    data = response.json()
                    quotes = data.get("data", {}).get("quotes", [])

                    for q in quotes:
                        try:
                            ts = make_aware(datetime.strptime(q["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ")).replace(second=0, microsecond=0)
                            quote_map.append((ts, q["quote"]["USD"]))
                        except Exception:
                            continue

                except Exception as e:
                    print(f"❌ Failed to fetch historical quotes for {symbol} on {current_day.date()}: {e}")

                metrics = list(RickisMetrics.objects.filter(
                    coin=coin,
                    timestamp__gte=make_aware(datetime.combine(current_day, datetime.min.time())),
                    timestamp__lt=make_aware(datetime.combine(current_day + timedelta(days=1), datetime.min.time())),
                ).filter(
                    Q(change_1h__in=[None, 0]) | Q(change_24h__in=[None, 0])
                ))

                updated = 0
                for metric in metrics:
                    best_quote = None
                    min_diff = timedelta(minutes=30)

                    for ts, quote in quote_map:
                        diff = abs(ts - metric.timestamp)
                        if diff < min_diff:
                            min_diff = diff
                            best_quote = quote

                    # Fallback to quotes/latest if nothing matched
                    if not best_quote:
                        try:
                            fallback_resp = requests.get(CMC_LATEST_URL, headers=HEADERS, params={"symbol": symbol, "convert": "USD"})
                            fallback_data = fallback_resp.json()
                            best_quote = fallback_data["data"][symbol]["quote"]["USD"]
                            print(f"🔁 Fallback used for {symbol} at {metric.timestamp}")
                        except Exception as e:
                            print(f"❌ Fallback failed for {symbol} at {metric.timestamp}: {e}")
                            continue

                    change_1h = best_quote.get("percent_change_1h")
                    change_24h = best_quote.get("percent_change_24h")

                    if change_1h is not None and metric.change_1h in [None, 0]:
                        metric.change_1h = change_1h
                    if change_24h is not None and metric.change_24h in [None, 0]:
                        metric.change_24h = change_24h

                    updated += 1

                if updated > 0:
                    RickisMetrics.objects.bulk_update(metrics, ["change_1h", "change_24h"])
                    print(f"✅ {symbol} on {current_day.date()} — {updated} entries filled")

                current_day += timedelta(days=1)
                time.sleep(SECONDS_BETWEEN_CALLS)

        print("\n🎉 All missing change_1h and change_24h values are now filled.")
