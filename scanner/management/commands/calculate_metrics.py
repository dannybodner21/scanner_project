from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from datetime import datetime
from scanner.models import Coin, RickisMetrics
import numpy as np
from decimal import Decimal

class Command(BaseCommand):
    help = 'Calculate local metrics for RickisMetrics including stochastic indicators, support/resistance, and ATR'

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

        for symbol in symbols:
            coin = coin_map.get(symbol)
            if not coin:
                self.stdout.write(self.style.ERROR(f"❌ {symbol} coin not found."))
                continue

            self.stdout.write(self.style.NOTICE(f"\n🚀 Calculating local metrics for {symbol}"))

            metrics = list(RickisMetrics.objects.filter(
                coin=coin,
                timestamp__gte=start_date,
                timestamp__lt=end_date
            ).order_by('timestamp'))

            price_window = []
            volume_window = []
            high_window = []
            low_window = []
            ema_12 = None
            ema_26 = None
            ema_alpha_12 = 2 / (12 + 1)
            ema_alpha_26 = 2 / (26 + 1)

            to_update = []

            for idx, entry in enumerate(metrics):
                price = float(entry.price)
                price_window.append(price)
                volume_window.append(float(entry.volume))
                high_window.append(float(entry.high_24h or 0))
                low_window.append(float(entry.low_24h or 0))

                if len(volume_window) >= 12:
                    entry.avg_volume_1h = np.mean(volume_window[-12:])
                    entry.relative_volume = float(entry.volume) / entry.avg_volume_1h if entry.avg_volume_1h else 0

                if len(price_window) >= 5:
                    entry.sma_5 = np.mean(price_window[-5:])
                if len(price_window) >= 20:
                    entry.sma_20 = np.mean(price_window[-20:])

                if len(price_window) >= 12:
                    if ema_12 is None:
                        ema_12 = np.mean(price_window[-12:])
                    else:
                        ema_12 = (price_window[-1] - ema_12) * ema_alpha_12 + ema_12
                    entry.ema_12 = ema_12

                if len(price_window) >= 26:
                    if ema_26 is None:
                        ema_26 = np.mean(price_window[-26:])
                    else:
                        ema_26 = (price_window[-1] - ema_26) * ema_alpha_26 + ema_26
                    entry.ema_26 = ema_26

                if entry.ema_12 is not None and entry.ema_26 is not None:
                    entry.macd = entry.ema_12 - entry.ema_26

                if entry.macd is not None:
                    if hasattr(entry, 'macd_signal') and entry.macd_signal is not None:
                        entry.macd_signal = (entry.macd - entry.macd_signal) * (2 / (9 + 1)) + entry.macd_signal
                    else:
                        entry.macd_signal = entry.macd

                if len(price_window) >= 12:
                    entry.stddev_1h = np.std(price_window[-12:])

                if len(price_window) >= 14:
                    deltas = np.diff(price_window[-15:])
                    seed = deltas[:14]
                    up = seed[seed > 0].sum() / 14
                    down = -seed[seed < 0].sum() / 14
                    rs = up / down if down != 0 else 0
                    entry.rsi = 100. - (100. / (1. + rs))

                if len(price_window) >= 12:
                    x = np.arange(len(price_window[-12:]))
                    y = np.array(price_window[-12:])
                    A = np.vstack([x, np.ones(len(x))]).T
                    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
                    entry.price_slope_1h = m

                if len(price_window) >= 14:
                    high14 = max(high_window[-14:])
                    low14 = min(low_window[-14:])
                    if high14 != low14:
                        entry.stochastic_k = (price_window[-1] - low14) / (high14 - low14) * 100
                    else:
                        entry.stochastic_k = 0

                if len(price_window) >= 16:
                    ks = []
                    for offset in range(-3, 0):
                        if idx + offset - 13 >= 0:
                            h14 = max(high_window[idx + offset - 13:idx + offset + 1])
                            l14 = min(low_window[idx + offset - 13:idx + offset + 1])
                            if h14 != l14:
                                ks.append((price_window[idx + offset] - l14) / (h14 - l14) * 100)
                    entry.stochastic_d = np.mean(ks) if ks else 0

                if len(high_window) >= 12 and len(low_window) >= 12:
                    entry.support_level = min(low_window[-12:])
                    entry.resistance_level = max(high_window[-12:])

                if len(high_window) >= 2 and len(low_window) >= 2 and len(price_window) >= 2:
                    trs = [
                        max(high_window[i] - low_window[i], abs(high_window[i] - price_window[i-1]), abs(low_window[i] - price_window[i-1]))
                        for i in range(1, len(high_window))
                    ]
                    if len(trs) >= 12:
                        entry.atr_1h = np.mean(trs[-12:])

                to_update.append(entry)

            RickisMetrics.objects.bulk_update(to_update, fields=[
                'avg_volume_1h', 'relative_volume',
                'sma_5', 'sma_20', 'ema_12', 'ema_26',
                'macd', 'macd_signal', 'stddev_1h',
                'rsi', 'price_slope_1h',
                'stochastic_k', 'stochastic_d',
                'support_level', 'resistance_level',
                'atr_1h'
            ], batch_size=100)

            self.stdout.write(self.style.SUCCESS(f"✅ Local metrics calculated for {symbol}"))
