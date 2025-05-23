from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics, Coin
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
import requests
import time
from dateutil import parser

CMC_API_KEY = "6520549c-03bb-41cd-86e3-30355ece87ba"
HEADERS = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
BASE_URL = "https://pro-api.coinmarketcap.com/v2"

class Command(BaseCommand):
    help = "Check and fill in missing 5-minute RickisMetrics entries from May 10 to May 22, 2025."

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 5, 10))
        end = make_aware(datetime(2025, 5, 23))

        rickisSymbols = [
            "BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LINK",
            "AVAX", "XLM", "TON", "SHIB", "SUI", "HBAR", "BCH", "DOT", "LTC",
            "XMR", "UNI", "PEPE", "APT", "NEAR", "ONDO", "TAO", "ICP", "ETC",
            "RENDER", "MNT", "KAS", "CRO", "AAVE", "POL", "VET", "FIL", "ALGO",
            "ENA", "ATOM", "TIA", "ARB", "DEXE", "OP", "JUP", "MKR", "STX",
            "EOS", "WLD", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT",
            "PAXG", "CRV", "JASMY", "SAND", "GALA", "CORE", "KAIA", "LDO",
            "THETA", "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI"
        ]

        timestamps = []
        current = start
        while current < end:
            timestamps.append(current)
            current += timedelta(minutes=5)
        expected_ts = set(timestamps)

        for symbol in rickisSymbols:
            try:
                coin = Coin.objects.get(symbol=symbol)
            except Coin.DoesNotExist:
                self.stdout.write(f"❌ Coin not found: {symbol}")
                continue

            existing_ts = set(
                RickisMetrics.objects.filter(
                    coin=coin,
                    timestamp__gte=start,
                    timestamp__lt=end
                ).values_list('timestamp', flat=True)
            )

            missing_ts = sorted(expected_ts - existing_ts)
            if not missing_ts:
                self.stdout.write(f"✅ {symbol} is complete")
                continue

            self.stdout.write(f"🔧 {symbol} is missing {len(missing_ts)} entries, fetching...")

            date_chunks = sorted(set(ts.date() for ts in missing_ts))
            for day in date_chunks:
                day_start = make_aware(datetime.combine(day, datetime.min.time()))
                try:
                    quotes = self.get_quotes(symbol, day_start)
                except Exception as e:
                    self.stdout.write(f"❌ Error fetching data for {symbol} on {day}: {e}")
                    continue

                for ts in missing_ts:
                    if ts.date() != day:
                        continue
                    ts_epoch = int(ts.timestamp())
                    candidates = [ts_epoch, ts_epoch - 60, ts_epoch + 60, ts_epoch - 120, ts_epoch + 120]
                    q = next((quotes.get(c) for c in candidates if quotes.get(c)), None)
                    if not q:
                        self.stdout.write(f"⚠️ No quote for {symbol} at {ts} (epoch: {ts_epoch})")
                        continue
                    try:
                        RickisMetrics.objects.create(
                            coin=coin,
                            timestamp=ts,
                            price=q.get("price"),
                            volume=q.get("volume_24h"),
                            change_1h=q.get("percent_change_1h"),
                            change_24h=q.get("percent_change_24h")
                        )
                        self.stdout.write(f"✅ Inserted {symbol} at {ts}")
                    except Exception as e:
                        self.stdout.write(f"❌ Failed insert {symbol} at {ts}: {e}")
                time.sleep(1.2)

        self.stdout.write("\n🎉 Backfill complete.")

    def get_quotes(self, symbol, date):
        url = f"{BASE_URL}/cryptocurrency/quotes/historical"
        params = {
            "symbol": symbol,
            "interval": "5m",
            "time_start": date.strftime("%Y-%m-%d"),
            "time_end": (date + timedelta(days=1)).strftime("%Y-%m-%d"),
        }

        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json().get("data", {}).get("quotes", [])

        return {
            int(parser.isoparse(item["timestamp"]).timestamp()): {
                "price": item["quote"]["USD"]["price"],
                "volume_24h": item["quote"]["USD"]["volume_24h"],
                "percent_change_1h": item["quote"]["USD"].get("percent_change_1h"),
                "percent_change_24h": item["quote"]["USD"].get("percent_change_24h"),
            }
            for item in data
        }
