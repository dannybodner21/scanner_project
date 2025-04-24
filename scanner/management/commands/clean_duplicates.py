from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
from scanner.models import Coin, ShortIntervalData

class Command(BaseCommand):
    help = 'Clean up ShortIntervalData for a list of specific coins, ensuring exactly 288 entries per day with 5-minute aligned timestamps.'

    def handle(self, *args, **kwargs):
        start_date = make_aware(datetime(2025, 3, 20))
        end_date = make_aware(datetime(2025, 4, 23))  # exclusive

        def round_to_five_minutes(dt):
            return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)


        rickisCoins = ["BTC","ETH","XRP","BNB","SOL","TRX","DOGE","ADA","LEO","LINK",
                        "AVAX","XLM","TON","SHIB","SUI","HBAR","BCH","DOT","LTC",
                        "HYPE","BGB","DAI","PI","XMR","UNI","PEPE","OKB","APT","GT",
                        "NEAR","ONDO","TAO","ICP","ETC","RENDER","MNT","KAS","CRO",
                        "AAVE","POL","VET","FIL","TRUMP","ALGO","ENA","ATOM","TIA",
                        "FET","ARB","S","KCS","DEXE","OP","JUP","MKR","XDC","STX",
                        "FLR","EOS","WLD","IP","BONK","FARTCOIN","SEI","INJ","IMX",
                        "GRT","FORM","QNT","PAXG","CRV","JASMY","SAND","GALA",
                        "NEXO","CORE","RAY","KAIA","LDO","THETA","IOTA","HNT",
                        "MANA","FLOW","CAKE","MOVE","FLOKI","XCN"]

        symbols = ["BTC","ETH","XRP","BNB","SOL","TRX","DOGE","ADA","LEO","LINK"]

        for symbol in symbols:
            try:
                coin = Coin.objects.get(symbol=symbol)
            except Coin.DoesNotExist:
                self.stdout.write(f"❌ {symbol} coin not found in the database.")
                continue

            self.stdout.write(f"\n🔍 Cleaning data for: {coin.symbol}")
            current = start_date

            while current < end_date:
                next_day = current + timedelta(days=1)
                entries = ShortIntervalData.objects.filter(
                    coin=coin,
                    timestamp__gte=current,
                    timestamp__lt=next_day
                )

                grouped = {}
                to_delete = []

                for entry in entries:
                    original_ts = entry.timestamp
                    rounded_ts = round_to_five_minutes(original_ts)

                    if original_ts != rounded_ts:
                        if not ShortIntervalData.objects.filter(coin=coin, timestamp=rounded_ts).exists():
                            entry.timestamp = rounded_ts
                            entry.save()
                            self.stdout.write(f"🕒 Updated {coin.symbol} timestamp from {original_ts} -> {rounded_ts}")
                        else:
                            to_delete.append(entry)
                            self.stdout.write(f"❌ Deleted misaligned duplicate at {original_ts}")
                        continue

                    if rounded_ts not in grouped:
                        grouped[rounded_ts] = entry
                    else:
                        to_delete.append(entry)
                        self.stdout.write(f"❌ Deleted duplicate for {coin.symbol} at {rounded_ts}")

                aligned_entries = list(grouped.values())
                aligned_entries.sort(key=lambda x: x.timestamp)  # ✅ Ensure deterministic ordering

                if len(aligned_entries) > 288:
                    overflow = aligned_entries[288:]
                    to_delete += overflow
                    self.stdout.write(f"⚠️  {len(overflow)} overflow entries deleted for {coin.symbol} on {current.date()}")

                for d in to_delete:
                    d.delete()

                current = next_day

            self.stdout.write(f"\n✅ Cleanup complete for {coin.symbol}.")
