from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime, timedelta
from scanner.models import Coin, ShortIntervalData, RickisMetrics
from decimal import Decimal

class Command(BaseCommand):
    help = 'Label RickisMetrics with long/short results based on ShortIntervalData for BTC.'

    def handle(self, *args, **kwargs):
        start_date = make_aware(datetime(2025, 3, 22))
        end_date = make_aware(datetime(2025, 4, 23))  # exclusive

        def round_to_five_minutes(dt):
            return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)

        symbolsTwo = [
            "BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LINK",
            "AVAX", "XLM", "TON", "SHIB", "SUI", "HBAR", "BCH", "DOT", "LTC",
            "XMR", "UNI", "PEPE", "APT", "NEAR", "ONDO", "TAO", "ICP", "ETC",
            "RENDER", "MNT", "KAS", "CRO", "AAVE", "POL", "VET", "FIL", "ALGO",
            "ENA", "ATOM", "TIA", "ARB", "DEXE", "OP", "JUP", "MKR", "STX",
            "EOS", "WLD", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT",
            "PAXG", "CRV", "JASMY", "SAND", "GALA", "CORE", "KAIA", "LDO",
            "THETA", "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI"
        ]

        symbols = [
            "STX",
            "EOS", "WLD", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT",
            "PAXG", "CRV", "JASMY", "SAND", "GALA", "CORE", "KAIA", "LDO",
            "THETA", "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI"
        ]

        for symbol in symbols:

            try:
                coin = Coin.objects.get(symbol=symbol)
            except Coin.DoesNotExist:
                self.stdout.write(self.style.ERROR("❌ BTC coin not found."))
                return

            self.stdout.write(self.style.NOTICE(f"\n🚀 Labeling RickisMetrics for {coin.symbol}"))

            # Get all ShortIntervalData entries ordered by timestamp
            short_data = list(
                ShortIntervalData.objects.filter(
                    coin=coin,
                    timestamp__gte=start_date,
                    timestamp__lt=end_date
                ).order_by('timestamp')
            )

            short_data_map = {round_to_five_minutes(entry.timestamp): entry for entry in short_data}
            rounded_timestamps = sorted(short_data_map.keys())

            for idx, ts in enumerate(rounded_timestamps):
                entry = short_data_map[ts]
                price_now = entry.price

                # Check if RickisMetrics already exists, else create basic one
                rickis, created = RickisMetrics.objects.get_or_create(
                    coin=coin,
                    timestamp=ts,
                    defaults={
                        'price': price_now,
                        'volume': entry.volume_5min,
                        'high_24h': 0,
                        'change_5m': 0,
                        'change_1h': 0,
                        'change_24h': 0,
                        'avg_volume_1h': 0,
                    }
                )

                if rickis.timestamp != ts:
                    rickis.timestamp = ts
                    rickis.save()

                # Only label if long_result/short_result not already set
                if rickis.long_result is not None and rickis.short_result is not None:
                    continue

                long_take_profit = price_now * Decimal('1.04')
                long_stop_loss = price_now * Decimal('0.98')
                short_take_profit = price_now * Decimal('0.96')
                short_stop_loss = price_now * Decimal('1.02')

                future_prices = [short_data_map[t] for t in rounded_timestamps[idx+1:] if t in short_data_map]

                long_outcome = None
                short_outcome = None

                for future_entry in future_prices:
                    future_price = future_entry.price

                    if long_outcome is None:
                        if future_price >= long_take_profit:
                            long_outcome = True
                        elif future_price <= long_stop_loss:
                            long_outcome = False

                    if short_outcome is None:
                        if future_price <= short_take_profit:
                            short_outcome = True
                        elif future_price >= short_stop_loss:
                            short_outcome = False

                    # Stop early if both outcomes are decided
                    if long_outcome is not None and short_outcome is not None:
                        break

                rickis.long_result = long_outcome
                rickis.short_result = short_outcome
                rickis.save()

                self.stdout.write(self.style.SUCCESS(f"✅ Labeled {ts}: Long {long_outcome}, Short {short_outcome}"))

            self.stdout.write(self.style.SUCCESS("\n✅ RickisMetrics labeling complete for BTC."))
