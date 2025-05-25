from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from scanner.models import Coin, RickisMetrics
from dateutil.parser import isoparse
import requests
import time

# nohup python manage.py backfill_check_prices > output.log 2>&1 &
# tail -f output.log

API_KEY = '6520549c-03bb-41cd-86e3-30355ece87ba'
CMC_QUOTES_URL = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical'

class Command(BaseCommand):
    help = 'Verify and correct RickisMetrics from historical CMC quotes.'

    def handle(self, *args, **kwargs):
        start_date = make_aware(datetime(2025, 3, 22))
        end_date = make_aware(datetime(2025, 4, 23))

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

        coins = Coin.objects.filter(symbol__in=symbols)
        coin_map = {coin.symbol: coin for coin in coins}
        corrections = []

        for symbol in symbols:
            coin = coin_map.get(symbol)
            if not coin:
                self.stdout.write(self.style.ERROR(f"❌ {symbol} coin not found."))
                continue

            self.stdout.write(self.style.NOTICE(f"\n🔍 Checking {symbol}"))

            current_day = start_date
            while current_day < end_date:
                quotes = self.fetch_cmc_quotes(symbol, current_day)
                if not quotes:
                    self.stdout.write(self.style.WARNING(f"⚠️  No data for {symbol} on {current_day.date()}"))
                    current_day += timedelta(days=1)
                    continue

                for quote in quotes:
                    try:
                        ts = datetime.strptime(quote["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ")
                        ts = ts.replace(second=0, microsecond=0)
                        ts = make_aware(ts)
                        rounded_minute = ts.minute - (ts.minute % 5)
                        ts = ts.replace(minute=rounded_minute)

                        metric = RickisMetrics.objects.filter(coin=coin, timestamp=ts).first()
                        if not metric:
                            continue

                        cmc_price = float(quote["quote"]["USD"]["price"])
                        db_price = float(metric.price)

                        if abs(db_price - cmc_price) / cmc_price > 0.01:
                            self.stdout.write(
                                self.style.WARNING(
                                    f"❌ {symbol} at {ts} — DB: {db_price}, CMC: {cmc_price}"
                                )
                            )
                            metric.price = cmc_price
                            metric.save()
                            corrections.append((symbol, ts))
                    except Exception as e:
                        self.stdout.write(self.style.ERROR(f"💥 Error at {quote.get('timestamp')}: {e}"))

                self.stdout.write(self.style.SUCCESS(f"✅ {symbol} on {current_day.date()} checked"))
                current_day += timedelta(days=1)
                time.sleep(2)

        self.stdout.write(self.style.SUCCESS(f"\n🎯 Corrections made: {len(corrections)}"))
        for sym, ts in corrections:
            print(f"✔️ Fixed {sym} at {ts}")

    def fetch_cmc_quotes(self, symbol, date):
        headers = {"X-CMC_PRO_API_KEY": API_KEY}
        params = {
            "symbol": symbol,
            "interval": "5m",
            "time_start": date.strftime("%Y-%m-%d"),
            "time_end": (date + timedelta(days=1)).strftime("%Y-%m-%d"),
            "convert": "USD"
        }
        try:
            response = requests.get(CMC_QUOTES_URL, headers=headers, params=params)
            response.raise_for_status()
            return response.json().get("data", {}).get("quotes", [])
        except Exception as e:
            print(f"❌ Error fetching data for {symbol} on {date.date()}: {e}")
            return []
