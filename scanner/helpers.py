from datetime import datetime, timedelta
from django.utils.timezone import make_aware
from scanner.models import ShortIntervalData
from scanner.models import RickisMetrics
import numpy as np
import requests
from decimal import Decimal
import pandas as pd
from ta.trend import ADXIndicator
from ta.volatility import BollingerBands


# take a timestamp and round it do be perfectly on the 5 min mark
def round_to_five_minutes(dt):
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)

# get recent prices for a coin in a given window - oldest to newest
def get_recent_prices(coin, timestamp, window):
    queryset = ShortIntervalData.objects.filter(
        coin=coin,
        timestamp__lte=timestamp
    ).order_by('-timestamp')[:window]
    prices = list(queryset.values_list('price', flat=True))
    return prices[::-1]

# get recent volumes for a coin in a given window - oldest to newest
def get_recent_volumes(coin, timestamp, window):
    queryset = ShortIntervalData.objects.filter(
        coin=coin,
        timestamp__lte=timestamp
    ).order_by('-timestamp')[:window]
    volumes = list(queryset.values_list('volume_5min', flat=True))
    return volumes[::-1]

# take a coin and calculate RSI based on 14 time periods
# RSI = 100 - (100 / (1 + (average_gain / average_loss)))
def calculate_rsi(coin, timestamp, period=14):
    try:
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

    except Exception as e:
        print(f"error in calculate_rsi: {e}")

# exponential moving average
# EMA = (current_price x multiplier_k) + (previous_EMA x (1 - multiplier_k))
def calculate_ema_from_prices(prices, window):

    try:

        if prices is None or len(prices) < window:
            return None

        prices = [float(p) for p in prices]
        k = 2 / (window + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price * k) + (ema * (1 - k))
        return ema

    except Exception as e:
        print(f"error in calculate_ema_from_prices: {e}")

# Moving Average Convergence Divergence
# MACD and Signal Line
# MACD = EMA(12) - EMA(26)
# Signal Line = EMA(9) of MACD
def calculate_macd(coin, timestamp):

    try:

        prices = get_recent_prices(coin, timestamp, window=50)
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
            return macd_line, macd_line

        # Signal Line = EMA(9) of MACD history
        signal_line = calculate_ema_from_prices(np.array(macd_history), 9)

        return macd_line, signal_line

    except Exception as e:
        print(f"error in calculate_macd: {e}")

# Stochastic K and Stochastic D
# K shows current position of the price relative to recent high/low range
# D shows the moving average of K and is the signal line
# K = (C - Ln) / (Hn - Ln) x 100
# C = current closing price
# Ln = lowest low over last n periods (I am using 14)
# Hn = highest high over the last n periods (14)
def calculate_stochastic(coin, timestamp, period=14, smoothing=3):

    try:

        # get recent prices -> can't be lower than 16 time periods
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

    except Exception as e:
        print(f"error in calculate_stochastic: {e}")

# get recent support / resistance levels on a 20 time period window
def calculate_support_resistance(coin, timestamp, period=20):

    try:

        # get recent price action
        prices = get_recent_prices(coin, timestamp, period)

        if not prices:
            return None, None

        # return the recent high and low as resistance and support
        return min(prices), max(prices)

    except Exception as e:
        print(f"error in calculate_support_resistance: {e}")

# Average volume over 1 hour
# Average_volme = mean of volumes over 12 time periods
def calculate_avg_volume_1h(coin, timestamp):

    try:
        # get volumes
        volumes = get_recent_volumes(coin, timestamp, 12)

        if not volumes:
            return None

        # calculate the mean
        return np.mean(volumes)

    except Exception as e:
        print(f"error in calculate_avg_volume_1h: {e}")

# Relative volume = recent volume / average volume
def calculate_relative_volume(coin, timestamp):

    try:

        last_volume = ShortIntervalData.objects.filter(
            coin=coin,
            timestamp=timestamp
        ).values_list('volume_5min', flat=True).first()
        avg_volume = calculate_avg_volume_1h(coin, timestamp)

        if avg_volume and avg_volume != 0:
            return float(last_volume) / float(avg_volume)

        return None

    except Exception as e:
        print(f"error in calculate_relative_volume: {e}")


def calculate_sma(coin, timestamp, window):

    try:

        prices = get_recent_prices(coin, timestamp, window)
        if not prices:
            return None
        return np.mean(prices)

    except Exception as e:
        print(f"error in calculate_sma: {e}")


def calculate_ema(coin, timestamp, window):

    try:

        prices = ShortIntervalData.objects.filter(
            coin=coin, timestamp__lte=timestamp
        ).order_by('-timestamp').values_list('price', flat=True)[:window]

        prices = list(prices)
        if len(prices) < window:
            return None

        prices = [float(p) for p in prices]
        return calculate_ema_from_prices(prices, window)

    except Exception as e:
        print(f"error in calculate_ema: {e}")


def calculate_stddev_1h(coin, timestamp):

    try:

        prices = get_recent_prices(coin, timestamp, 12)
        if not prices:
            return None
        return np.std(prices)

    except Exception as e:
        print(f"error in calculate_stddev_1h: {e}")


def calculate_price_slope_1h(coin, timestamp):

    try:

        prices = ShortIntervalData.objects.filter(
            coin=coin, timestamp__lte=timestamp
        ).order_by('-timestamp').values_list('price', flat=True)[:12]

        prices = list(prices)
        if len(prices) < 2:
            return None

        prices = [float(p) for p in prices]  # <<< ADD THIS
        x = list(range(len(prices)))
        y = prices

        # Simple linear regression
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xx = sum(xi * xi for xi in x)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        denominator = n * sum_xx - sum_x ** 2
        if denominator == 0:
            return None

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope

    except Exception as e:
        print(f"error in calculate_price_slope_1h: {e}")


# Average True Rating over 1 hour
# Used to measure how much an asset moves on average over a time period
# ATR_1h = mean of price differences over 1 hour
def calculate_atr_1h(coin, timestamp):

    try:

        # get coin data over an hour period
        queryset = ShortIntervalData.objects.filter(
            coin=coin,
            timestamp__lte=timestamp
        ).order_by('-timestamp')[:12]

        prices = list(queryset.values_list('price', flat=True))

        if len(prices) < 2:
            return None

        # calculate absolute difference over our range
        true_range = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]

        # return the mean
        return np.mean(true_range)

    except Exception as e:
        print(f"error in calculate_atr_1h: {e}")


# function to calculate the price change over a 5 min period
# % change = ((current price - price 5 min ago) / price 5 min ago) x 100
def calculate_price_change_five_min(coin, timestamp):

    try:

        # get current price
        current = ShortIntervalData.objects.filter(
            coin=coin,
            timestamp=timestamp
        ).values_list('price', flat=True).first()

        if current is None:
            return None

        current = float(current)

        # get closest recent price between 4–6 minutes ago
        target_time = timestamp - timedelta(minutes=5)
        window_start = target_time - timedelta(minutes=1)
        window_end = target_time + timedelta(minutes=1)

        previous = (
            ShortIntervalData.objects.filter(
                coin=coin,
                timestamp__gte=window_start,
                timestamp__lte=window_end
            )
            .order_by('timestamp')
            .values_list('price', flat=True)
            .first()
        )

        if previous is None or previous == 0:
            return None

        previous = float(previous)

        # calculate and return 5 min price change
        return ((current - previous) / previous) * 100

    except Exception as e:
        print(f"error in calculate_price_change_five_min: {e}")
        return None


# function to calculate the current price change from recent high
# % change = ((current price - high) / high) x 100
def calculate_change_since_high(price, high_24h):

    try:
        if price is None or high_24h is None:
            return None

        price = float(price)
        high_24h = float(high_24h)

        # return percentage change from recent high
        if high_24h > 0:
            return ((price - high_24h) / high_24h) * 100

    except:
        pass

    return None


# function to calculate the current price change from recent low
# % change = ((current price - low) / low) x 100
def calculate_change_since_low(price, low_24h):

    try:
        if price is None or low_24h is None:
            return None

        price = float(price)
        low_24h = float(low_24h)

        # return percentage change from recent low
        if low_24h > 0:
            return ((price - low_24h) / low_24h) * 100

    except:
        pass

    return None


# GOING TO DELETE THIS FUNCTION - NOT WORKING RIGHT
def calculate_volume_to_market_cap(volume, market_cap):
    try:
        return float(volume) / float(market_cap) if market_cap > 0 else None
    except:
        return None

# function to get market sentiment from alternative API
def fetch_fear_and_greed_index():

    try:
        res = requests.get("https://api.alternative.me/fng/?limit=1").json()
        data = res["data"][0]
        score = int(data["value"])
        label = data["value_classification"]
        return score, label

    except Exception as e:
        print(f"Error fetching Fear & Greed Index: {e}")
        return None, None


# OBV -> On Balance Volume
# A momentum indicator that is used to detect trend strength or reversals
# Shows if volume is flowing in or out of an asset
def calculate_obv(coin, timestamp):

    # get current and previous coin data
    current = ShortIntervalData.objects.filter(coin=coin, timestamp=timestamp).first()
    prev_time = timestamp - timedelta(minutes=5)
    previous = ShortIntervalData.objects.filter(coin=coin, timestamp=prev_time).first()

    if not current or not previous:
        return None

    try:
        current_price = float(current.price)
        previous_price = float(previous.price)
        current_volume = float(current.volume_5min)

    except:
        return None

    # get previous obv
    previous_obv = RickisMetrics.objects.filter(
        coin=coin,
        timestamp=prev_time
    ).values_list('obv', flat=True).first() or 0.0

    # price is rising -> return previous obv + current volume
    if current_price > previous_price:
        return previous_obv + current_volume

    # price is falling -> return previous obv - current volume
    elif current_price < previous_price:
        return previous_obv - current_volume

    return previous_obv


def calculate_adx(coin, timestamp):
    candles = RickisMetrics.objects.filter(
        coin=coin,
        timestamp__lte=timestamp
    ).order_by('-timestamp')[:30]

    if len(candles) < 14:
        return None

    data = list(candles)[::-1]
    rows = []
    for c in data:
        try:
            if c.high_24h is None or c.low_24h is None or c.close is None:
                continue
            rows.append({
                "high": float(c.high_24h),
                "low": float(c.low_24h),
                "close": float(c.close)
            })
        except:
            continue

    if len(rows) < 14:
        return None

    df = pd.DataFrame(rows)
    indicator = ADXIndicator(df['high'], df['low'], df['close'], window=14)
    adx_series = indicator.adx()

    if adx_series.empty or len(adx_series) < 14:
        return None

    return adx_series.iloc[-1]

# Bollinger bands
# used to measure volatility and identify overbought / oversold assets
# upper band = moving average + 2 x standard deviation
# middle band = SMA - Simple Moving Average
# lower band = moving average - 2 x standard deviation
def calculate_bollinger_bands(coin, timestamp):

    # get recent metrics on 20 5 min time window
    candles = RickisMetrics.objects.filter(
        coin=coin,
        timestamp__lte=timestamp
    ).order_by('-timestamp')[:20]

    if len(candles) < 20:
        return None, None, None

    # get the recent candle closing prices
    closes = []
    for c in candles[::-1]:
        if c.close is not None:
            try:
                closes.append(float(c.close))
            except:
                continue

    # make sure we have at least 20 closing prices
    if len(closes) < 20:
        return None, None, None

    # use ta library to calculate bollinger bands
    df = pd.DataFrame({"close": closes})
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    return (
        bb.bollinger_hband().iloc[-1],
        bb.bollinger_mavg().iloc[-1],
        bb.bollinger_lband().iloc[-1],
    )

# Fibonacci retracement levels based on recent highs / lows
# technical indicators used to determine support and resistance levels
def calculate_fib_distances(high, low, current_price):

    try:
        if high is None or low is None or current_price is None:
            return {}

        # get recent high / low
        high = float(high)
        low = float(low)
        current_price = float(current_price)
        diff = high - low

        if diff == 0:
            return {}

        # calculate the Fibonacci retracement levels
        levels = {
            "fib_distance_0_236": low + 0.236 * diff,
            "fib_distance_0_382": low + 0.382 * diff,
            "fib_distance_0_5":   low + 0.5 * diff,
            "fib_distance_0_618": low + 0.618 * diff,
            "fib_distance_0_786": low + 0.786 * diff,
        }

        # return the percentage distance between current price and each level
        return {
            key: ((current_price - val) / val) * 100 if val != 0 else None
            for key, val in levels.items()
        }
    except Exception as e:
        print(f"error in calculate_fib_distances: {e}")
        return {}
