from datetime import datetime, timedelta
from django.utils.timezone import make_aware
from scanner.models import ShortIntervalData
import numpy as np
from decimal import Decimal


def round_to_five_minutes(dt):
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)


def get_recent_prices(coin, timestamp, window):
    queryset = ShortIntervalData.objects.filter(
        coin=coin,
        timestamp__lte=timestamp
    ).order_by('-timestamp')[:window]
    prices = list(queryset.values_list('price', flat=True))
    return prices[::-1]  # return oldest to newest


def get_recent_volumes(coin, timestamp, window):
    queryset = ShortIntervalData.objects.filter(
        coin=coin,
        timestamp__lte=timestamp
    ).order_by('-timestamp')[:window]
    volumes = list(queryset.values_list('volume_5min', flat=True))
    return volumes[::-1]


def calculate_rsi(coin, timestamp, period=14):
    prices = get_recent_prices(coin, timestamp, period + 1)
    if len(prices) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, len(prices)):
        delta = prices[i] - prices[i - 1]
        if delta >= 0:
            gains.append(delta)
        else:
            losses.append(abs(delta))
    avg_gain = np.mean(gains) if gains else 0
    avg_loss = np.mean(losses) if losses else 0
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_ema_from_prices(prices, window):
    """
    Calculate the Exponential Moving Average (EMA) of the given prices for a specified window.
    """
    if len(prices) < window:
        return None

    prices = np.array(prices)
    weights = np.exp(np.linspace(-1., 0., window))  # Exponential weights
    weights /= weights.sum()  # Normalize the weights

    # Apply the weighted moving average
    ema = np.convolve(prices, weights, mode='valid')  # 'valid' ensures the result is the right size
    return float(ema[-1])  # Return the latest EMA value


def calculate_macd(coin, timestamp):
    prices = get_recent_prices(coin, timestamp, window=50)  # get more history
    if len(prices) < 26:
        return None, None

    prices = np.array(prices)

    # Calculate EMA 12 and EMA 26
    ema12 = calculate_ema_from_prices(prices, 12)
    ema26 = calculate_ema_from_prices(prices, 26)

    if ema12 is None or ema26 is None:
        return None, None

    # MACD Line = EMA12 - EMA26
    macd_line = ema12 - ema26

    # Now create a short MACD history for Signal Line
    macd_history = []
    for i in range(len(prices) - 26):
        ema12_hist = calculate_ema_from_prices(prices[:26+i], 12)
        ema26_hist = calculate_ema_from_prices(prices[:26+i], 26)
        if ema12_hist is not None and ema26_hist is not None:
            macd_history.append(ema12_hist - ema26_hist)

    if len(macd_history) < 9:
        return macd_line, macd_line  # fallback if not enough history

    # Signal Line = EMA(9) of MACD history
    signal_line = calculate_ema_from_prices(np.array(macd_history), 9)

    return macd_line, signal_line


def calculate_stochastic(coin, timestamp, period=14, smoothing=3):
    prices = get_recent_prices(coin, timestamp, period + smoothing - 1)
    if len(prices) < period + smoothing - 1:
        return None, None

    k_values = []
    for i in range(smoothing):
        window_prices = prices[i:i+period]
        highest_high = max(window_prices)
        lowest_low = min(window_prices)
        current_close = window_prices[-1]
        if highest_high == lowest_low:
            k = 0
        else:
            k = (current_close - lowest_low) / (highest_high - lowest_low) * 100
        k_values.append(k)

    k = k_values[-1]
    d = np.mean(k_values)  # D = SMA(3) of K values

    return k, d


def calculate_support_resistance(coin, timestamp, period=20):
    prices = get_recent_prices(coin, timestamp, period)
    if not prices:
        return None, None
    return min(prices), max(prices)


def calculate_avg_volume_1h(coin, timestamp):
    volumes = get_recent_volumes(coin, timestamp, 12)
    if not volumes:
        return None
    return np.mean(volumes)


def calculate_relative_volume(coin, timestamp):
    last_volume = ShortIntervalData.objects.filter(
        coin=coin,
        timestamp=timestamp
    ).values_list('volume_5min', flat=True).first()
    avg_volume = calculate_avg_volume_1h(coin, timestamp)
    if avg_volume and avg_volume != 0:
        return float(last_volume) / float(avg_volume)
    return None


def calculate_sma(coin, timestamp, window):
    prices = get_recent_prices(coin, timestamp, window)
    if not prices:
        return None
    return np.mean(prices)


def calculate_ema(coin, timestamp, window):
    prices = get_recent_prices(coin, timestamp, window * 2)  # Get more to stabilize EMA
    if len(prices) < window:
        return None
    prices = np.array(prices)
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    ema = np.convolve(prices, weights, mode='valid')
    return float(ema[-1]) if len(ema) > 0 else None


def calculate_stddev_1h(coin, timestamp):
    prices = get_recent_prices(coin, timestamp, 12)
    if not prices:
        return None
    return np.std(prices)


def calculate_price_slope_1h(coin, timestamp):
    prices = get_recent_prices(coin, timestamp, 12)
    if len(prices) < 2:
        return None
    x = np.arange(len(prices))
    slope = np.polyfit(x, prices, 1)[0]
    return slope


def calculate_atr_1h(coin, timestamp):
    queryset = ShortIntervalData.objects.filter(
        coin=coin,
        timestamp__lte=timestamp
    ).order_by('-timestamp')[:12]
    prices = list(queryset.values_list('price', flat=True))
    if len(prices) < 2:
        return None
    trs = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
    return np.mean(trs)


def calculate_price_change_five_min(coin, timestamp):
    current = ShortIntervalData.objects.filter(
        coin=coin,
        timestamp=timestamp
    ).values_list('price', flat=True).first()
    prev_timestamp = timestamp - timedelta(minutes=5)
    previous = ShortIntervalData.objects.filter(
        coin=coin,
        timestamp=prev_timestamp
    ).values_list('price', flat=True).first()
    if current is None or previous is None or previous == 0:
        return None
    return ((current - previous) / previous) * 100
