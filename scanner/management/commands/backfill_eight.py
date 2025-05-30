from datetime import datetime
from django.utils.timezone import make_aware
from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
from scanner.helpers import (
    calculate_price_change_five_min, calculate_avg_volume_1h, calculate_rsi,
    calculate_macd, calculate_stochastic, calculate_support_resistance,
    calculate_relative_volume, calculate_sma, calculate_stddev_1h,
    calculate_atr_1h, calculate_obv, calculate_change_since_high,
    calculate_change_since_low, calculate_fib_distances
)
from django.db import close_old_connections

class Command(BaseCommand):
    help = 'Full backfill of missing metrics from March 23 to May 23, 2025.'

    def handle(self, *args, **kwargs):
        start = make_aware(datetime(2025, 3, 23))
        end = make_aware(datetime(2025, 4, 23))

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

        metrics = RickisMetrics.objects.filter(
            timestamp__gte=start,
            timestamp__lt=end,
            coin__symbol__in=symbols
        ).select_related('coin')

        count = 0

        for metric in metrics:
            updated = False
            close_old_connections()

            coin = metric.coin
            timestamp = metric.timestamp
            price = metric.price
            high = metric.high_24h
            low = metric.low_24h

            # change_5m
            if metric.change_5m is None:
                change_5m = calculate_price_change_five_min(coin, timestamp)
                if change_5m is not None:
                    metric.change_5m = change_5m
                    updated = True

            # avg_volume_1h
            if metric.avg_volume_1h is None:
                avg_vol = calculate_avg_volume_1h(coin, timestamp)
                if avg_vol is not None:
                    metric.avg_volume_1h = avg_vol
                    updated = True

            # rsi
            if metric.rsi is None:
                rsi = calculate_rsi(coin, timestamp)
                if rsi is not None:
                    metric.rsi = rsi
                    updated = True

            # macd, macd_signal
            if metric.macd is None or metric.macd_signal is None:
                macd, signal = calculate_macd(coin, timestamp)
                if macd is not None:
                    metric.macd = macd
                    updated = True
                if signal is not None:
                    metric.macd_signal = signal
                    updated = True

            # stochastic_k, stochastic_d
            if metric.stochastic_k is None or metric.stochastic_d is None:
                k, d = calculate_stochastic(coin, timestamp)
                if k is not None:
                    metric.stochastic_k = k
                    updated = True
                if d is not None:
                    metric.stochastic_d = d
                    updated = True

            # support, resistance
            if metric.support_level is None or metric.resistance_level is None:
                support, resistance = calculate_support_resistance(coin, timestamp)
                if support is not None:
                    metric.support_level = support
                    updated = True
                if resistance is not None:
                    metric.resistance_level = resistance
                    updated = True

            # relative_volume
            if metric.relative_volume is None:
                rvol = calculate_relative_volume(coin, timestamp)
                if rvol is not None:
                    metric.relative_volume = rvol
                    updated = True

            # sma_5, sma_20
            if metric.sma_5 is None:
                sma_5 = calculate_sma(coin, timestamp, 5)
                if sma_5 is not None:
                    metric.sma_5 = sma_5
                    updated = True
            if metric.sma_20 is None:
                sma_20 = calculate_sma(coin, timestamp, 20)
                if sma_20 is not None:
                    metric.sma_20 = sma_20
                    updated = True

            # stddev_1h
            if metric.stddev_1h is None:
                stddev = calculate_stddev_1h(coin, timestamp)
                if stddev is not None:
                    metric.stddev_1h = stddev
                    updated = True

            # atr_1h
            if metric.atr_1h is None:
                atr = calculate_atr_1h(coin, timestamp)
                if atr is not None:
                    metric.atr_1h = atr
                    updated = True

            # obv
            if metric.obv is None:
                obv = calculate_obv(coin, timestamp)
                if obv is not None:
                    metric.obv = obv
                    updated = True

            # change_since_high / change_since_low
            if metric.change_since_high is None:
                csh = calculate_change_since_high(price, high)
                if csh is not None:
                    metric.change_since_high = csh
                    updated = True
            if metric.change_since_low is None:
                csl = calculate_change_since_low(price, low)
                if csl is not None:
                    metric.change_since_low = csl
                    updated = True

            # fib distances
            fibs = calculate_fib_distances(high, low, price)
            for field, value in fibs.items():
                if getattr(metric, field) is None and value is not None:
                    setattr(metric, field, value)
                    updated = True

            if updated:
                metric.save()
                count += 1
                if count % 100 == 0:
                    print(f"✅ Updated {count} metrics")

        print(f"🎯 Done. Total updated: {count}")
