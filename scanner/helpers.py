from datetime import datetime, timedelta
from django.utils.timezone import is_naive, make_aware
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


def get_recent_prices(coin, timestamp, window):
    # Fetch more than needed to ensure we can filter out garbage
    queryset = RickisMetrics.objects.filter(
        coin=coin,
        timestamp__lte=timestamp,
        price__gt=0  # filter out missing/zero prices
    ).order_by('-timestamp')[:window + 10]

    prices = list(queryset.values_list('price', flat=True))

    # Defensive: convert and reverse
    prices = [float(p) for p in prices if p is not None][:window]
    prices = prices[::-1]

    if len(prices) < window:
        print(f"⚠️ {coin.symbol} only has {len(prices)} valid prices (needed {window}) at {timestamp}")
        return []

    return prices


# get recent volumes for a coin in a given window - oldest to newest
def get_recent_volumes(coin, timestamp, window):
    # Fetch more than needed to ensure enough clean data
    queryset = RickisMetrics.objects.filter(
        coin=coin,
        timestamp__lte=timestamp,
        volume__isnull=False
    ).order_by('-timestamp')[:window + 20]

    volumes = []
    for v in queryset:
        try:
            vol = float(v.volume)
            if vol > 0:
                volumes.append(vol)
        except:
            continue

    if len(volumes) < window:
        print(f"⚠️ {coin.symbol} only has {len(volumes)} valid volumes (needed {window}) at {timestamp}")
        return []

    return volumes[:window][::-1]  # oldest to newest


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
            if delta > 0:
                gains.append(delta)
            elif delta < 0:
                losses.append(abs(delta))

        if not gains and not losses:
            return None  # Flat price, no movement

        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0

        if avg_loss == 0:
            return 100
        if avg_gain == 0:
            return 0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    except Exception as e:
        print(f"❌ Error in calculate_rsi for {coin.symbol} at {timestamp}: {e}")
        return None



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

        # calculate EMA 12 and EMA 26
        ema12 = calculate_ema_from_prices(prices, 12)
        ema26 = calculate_ema_from_prices(prices, 26)

        if ema12 is None or ema26 is None:
            return None, None

        macd_line = ema12 - ema26

        # calculate signal line
        macd_history = []
        for i in range(len(prices) - 26):
            ema12_hist = calculate_ema_from_prices(prices[:26+i], 12)
            ema26_hist = calculate_ema_from_prices(prices[:26+i], 26)
            if ema12_hist is not None and ema26_hist is not None:
                macd_history.append(ema12_hist - ema26_hist)

        if len(macd_history) < 9:
            return macd_line, macd_line

        signal_line = calculate_ema_from_prices(np.array(macd_history), 9)
        return macd_line, signal_line

    except Exception as e:
        print(f"error in calculate_macd: {e}")


# Stochastic K and Stochastic D
# K shows current position of the price relative to recent high/low range
# D shows the moving average of K and is the signal line
# D = SMA(3) of K values
# K = (C - Ln) / (Hn - Ln) x 100
# C = current closing price
# Ln = lowest low over last n periods (I am using 14)
# Hn = highest high over the last n periods (14)
def calculate_stochastic(coin, timestamp, period=14, smoothing=3):
    try:
        # Pull enough candles to calculate smoothed K
        candles = (
            RickisMetrics.objects
            .filter(coin=coin, timestamp__lte=timestamp)
            .order_by('-timestamp')[:period + smoothing - 1]
        )

        if len(candles) < period + smoothing - 1:
            print(f"❌ Not enough data for stochastic: {coin.symbol} at {timestamp}")
            return None, None

        candles = list(candles)[::-1]  # oldest to newest
        k_values = []

        for i in range(smoothing):
            window = candles[i:i+period]
            highs = [c.high_24h for c in window if c.high_24h]
            lows = [c.low_24h for c in window if c.low_24h]
            closes = [c.close for c in window if c.close]

            if len(highs) < period or len(lows) < period or not closes:
                continue

            highest_high = max(highs)
            lowest_low = min(lows)
            current_close = closes[-1]

            if highest_high == lowest_low:
                k = 0
            else:
                k = (current_close - lowest_low) / (highest_high - lowest_low) * 100

            k_values.append(k)

        if not k_values:
            return None, None

        k = k_values[-1]
        d = sum(k_values) / len(k_values)

        return k, d

    except Exception as e:
        print(f"error in calculate_stochastic: {e}")
        return None, None


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


# average volume over 1 hour
# average_volme = mean of volumes over 12 time periods
def calculate_avg_volume_1h(coin, timestamp):

    try:
        # get volumes
        volumes = get_recent_volumes(coin, timestamp, 12)

        if not volumes:
            return 0.0

        # calculate the mean using numpy
        return np.mean(volumes)

    except Exception as e:
        print(f"error in calculate_avg_volume_1h: {e}")


# relative volume = recent volume / average volume
def calculate_relative_volume(coin, timestamp):

    try:
        # get recent volume
        last_volume = RickisMetrics.objects.filter(
            coin=coin,
            timestamp=timestamp
        ).values_list('volume', flat=True).first()

        # get average volume
        avg_volume = calculate_avg_volume_1h(coin, timestamp)

        # calculate relative volume
        if avg_volume and avg_volume != 0:
            return float(last_volume) / float(avg_volume)

        return 0.0

    except Exception as e:
        print(f"error in calculate_relative_volume: {e}")


# simple moving average -> average of prices over a time window
def calculate_sma(coin, timestamp, window):

    try:
        # get recent prices
        prices = get_recent_prices(coin, timestamp, window)

        if not prices:
            return None

        # calculate mean using numpy
        return np.mean(prices)

    except Exception as e:
        print(f"error in calculate_sma: {e}")


# standard deviation -> how much price fluctuates from the average
def calculate_stddev_1h(coin, timestamp):

    try:
        # get recent prices
        prices = get_recent_prices(coin, timestamp, 20)

        print(f"std prices: {prices}")

        if not prices:
            return None

        # return numpy standard deviation
        return np.std(prices)

    except Exception as e:
        print(f"error in calculate_stddev_1h: {e}")


# Average True Rating over 1 hour
# Used to measure how much an asset moves on average over a time period
# ATR_1h = mean of price differences over 1 hour
def calculate_atr_1h(coin, timestamp):

    try:
        # get an hour of metrics
        candles = RickisMetrics.objects.filter(
            coin=coin,
            timestamp__lte=timestamp
        ).order_by('-timestamp')[:12]

        if len(candles) < 2:
            return None

        # oldest to newest
        candles = list(candles)[::-1]
        true_ranges = []

        # get price differences
        for i in range(1, len(candles)):
            high_val = candles[i].high_24h
            low_val = candles[i].low_24h
            prev_close_val = candles[i-1].close

            if high_val is None or low_val is None or prev_close_val is None:
                continue

            high = float(high_val)
            low = float(low_val)
            prev_close = float(prev_close_val)

            true_range = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(true_range)

        if not true_ranges:
            return None
        # return the numpy mean
        return np.mean(true_ranges)

    except Exception as e:
        print(f"error in calculate_atr_1h: {e}")
        return None


# function to calculate the price change over a 5 min period
# % change = ((current price - price 5 min ago) / price 5 min ago) x 100
def calculate_price_change_five_min(coin, timestamp):
    metrics = (
        RickisMetrics.objects
        .filter(coin=coin, timestamp__lte=timestamp)
        .order_by('-timestamp')[:2]
    )

    if len(metrics) < 2:
        print(f"⚠️ Not enough data to calculate 5-min change for {coin.symbol} at {timestamp}")
        return None

    latest = metrics[0]
    previous = metrics[1]

    try:
        prev_price = float(previous.price)
        latest_price = float(latest.price)

        if prev_price == 0:
            return None

        return ((latest_price - prev_price) / prev_price) * 100
    except Exception as e:
        print(f"❌ Error calculating 5-min change for {coin.symbol} at {timestamp}: {e}")
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


# OBV -> On Balance Volume
# A momentum indicator that is used to detect trend strength or reversals
# Shows if volume is flowing in or out of an asset
# current price > previous = previous obv + volume
# current price < previous = previous obv - volume
def calculate_obv(coin, timestamp):
    metrics = (
        RickisMetrics.objects
        .filter(coin=coin, timestamp__lte=timestamp)
        .order_by('-timestamp')[:2]
    )

    if len(metrics) < 2:
        return None

    current = metrics[0]
    previous = metrics[1]

    try:
        current_price = float(current.price)
        previous_price = float(previous.price)
        current_volume = float(current.volume)
        previous_obv = previous.obv or 0.0

        if current_price > previous_price:
            return previous_obv + current_volume
        elif current_price < previous_price:
            return previous_obv - current_volume
        else:
            return previous_obv

    except Exception as e:
        print(f"❌ Error in calculate_obv for {coin.symbol} at {timestamp}: {e}")
        return None


# Fibonacci retracement levels based on recent highs / lows
# technical indicators used to determine support and resistance levels
def calculate_fib_distances(high, low, current_price):

    try:
        if high is None or low is None or current_price is None:
            print(f"high, low, or current price is NONE")

            return {}

        # get recent high / low
        high = float(high)
        low = float(low)
        current_price = float(current_price)
        diff = high - low

        if diff == 0:
            print(f"high: {high}")
            print(f"low: {low}")
            print(f"price: {current_price}")
            print(f"diff: {diff}")
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







# NOT USING AT THIS MOMENT -----------------------------------------------------

# exponential moving average
# EMA = (current_price x multiplier_k) + (previous_EMA x (1 - multiplier_k))
def calculate_ema_from_prices(prices, window):

    try:
        if prices is None or len(prices) < window:
            return None

        prices = [float(price) for price in prices]
        k = 2 / (window + 1)
        ema = prices[0]

        for priceTwo in prices[1:]:
            ema = (priceTwo * k) + (ema * (1 - k))
        return ema

    except Exception as e:
        print(f"error in calculate_ema_from_prices: {e}")


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


# NEED TO RECALCULATE ALL OF THESE WITH THIS NEW FUNCTION
def calculate_price_slope_1h(coin, timestamp):

    try:
        prices = RickisMetrics.objects.filter(
            coin=coin,
            timestamp__lte=timestamp
        ).order_by('timestamp')
        prices = list(prices.values_list('price', flat=True))[-12:]

        if len(prices) < 2:
            return None

        prices = [float(p) for p in prices]
        x = list(range(len(prices)))
        y = prices

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
        return None


# NEED TO RECALCULATE ALL OF THESE WITH THIS NEW FUNCTION
def calculate_ema(coin, timestamp, window):

    try:
        prices = RickisMetrics.objects.filter(
            coin=coin,
            timestamp__lte=timestamp
        ).order_by('-timestamp').values_list('price', flat=True)[:window]

        prices = list(prices)

        if len(prices) < window:
            return None

        prices = [float(p) for p in prices]

        return calculate_ema_from_prices(prices[::-1], window)

    except Exception as e:

        print(f"error in calculate_ema: {e}")
        return None
