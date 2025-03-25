from django.core.management.base import BaseCommand
from django.utils.timezone import now, timedelta
from scanner.models import Metrics, Coin, SuccessfulMove
from decimal import Decimal

class Command(BaseCommand):
    help = "Scan historical Metrics data for successful long entries"

    def handle(self, *args, **kwargs):
        coins = Coin.objects.all()
        total_saved = 0

        for coin in coins:
            all_metrics = Metrics.objects.filter(coin=coin).order_by("timestamp")
            metric_list = list(all_metrics)

            for i in range(len(metric_list) - 60):  # check next 12 intervals (1 hour)
                entry = metric_list[i]
                entry_price = entry.last_price
                if entry_price is None:
                    continue
                entry_price = Decimal(entry_price)

                tp_price = entry_price * Decimal("1.03")  # +3%
                sl_price = entry_price * Decimal("0.98")  # -2%

                found = False
                hit_tp = False
                hit_sl = False

                for future in metric_list[i+1 : i+13]:
                    if not future.last_price:
                        continue
                    price = Decimal(future.last_price)

                    if price >= tp_price:
                        hit_tp = True
                    if price <= sl_price:
                        hit_sl = True

                    if hit_tp and not hit_sl:
                        found = True
                        break
                    if hit_sl and not hit_tp:
                        break  # invalid move, abort

                if found:
                    already_logged = SuccessfulMove.objects.filter(
                        coin=coin,
                        timestamp=entry.timestamp,
                        move_type="long"
                    ).exists()

                    if not already_logged:
                        SuccessfulMove.objects.create(
                            coin=coin,
                            timestamp=entry.timestamp,
                            entry_price=entry_price,
                            move_type="long",
                            metrics={
                                "price_change_5min": entry.price_change_5min,
                                "five_min_relative_volume": entry.five_min_relative_volume,
                                "price_change_1hr": entry.price_change_1hr,
                                "market_cap": float(entry.market_cap or 0),
                                "price_change_10min": entry.price_change_10min or 0,
                                "price_change_24hr": entry.price_change_24hr or 0,
                                "price_change_7d": entry.price_change_7d or 0,
                                "rolling_relative_volume": entry.rolling_relative_volume or 0,
                                "twenty_min_relative_volume": entry.twenty_min_relative_volume or 0,
                                "volume_24h": entry.volume_24h or 0,
                            }
                        )
                        print(f"✅ Long success: {coin.symbol} at {entry.timestamp}")
                        total_saved += 1

        print(f"\n✅ Done. Logged {total_saved} successful long setups.")
