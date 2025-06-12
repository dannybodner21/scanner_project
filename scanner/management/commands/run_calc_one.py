import pandas as pd
import ta
import numpy as np
import os
from scipy.stats import kurtosis, skew

EXPORT_FOLDER = 'exported_data'
OUTPUT_FOLDER = 'enriched_data'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

coins = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT", "SOLUSDT", "DOGEUSDT", "PEPEUSDT", "ADAUSDT",
         "XLMUSDT", "SUIUSDT", "LINKUSDT", "AVAXUSDT", "DOTUSDT", "SHIBUSDT", "HBARUSDT", "UNIUSDT"]

for coin in coins:
    print(f"Processing {coin}...")
    file_path = os.path.join(EXPORT_FOLDER, f"{coin}.csv")
    df = pd.read_csv(file_path)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df.set_index('timestamp', inplace=True)

    df.columns = ['open', 'high', 'low', 'close', 'volume']

    # Core indicators
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.macd(df['close'])
    macd_signal = ta.trend.macd_signal(df['close'])
    df['macd'] = macd - macd_signal
    df['macd_signal'] = macd_signal

    df['stochastic_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
    df['stochastic_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    df['bollinger_upper'] = ta.volatility.bollinger_hband(df['close'])
    df['bollinger_middle'] = ta.volatility.bollinger_mavg(df['close'])
    df['bollinger_lower'] = ta.volatility.bollinger_lband(df['close'])
    df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
    df['stddev_1h'] = df['close'].rolling(window=12).std()
    df['change_5m'] = df['close'].pct_change()
    df['change_1h'] = df['close'].pct_change(periods=12)
    df['change_24h'] = df['close'].pct_change(periods=288)
    df['relative_volume'] = df['volume'] / df['volume'].rolling(window=288).mean()
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])

    # Change since 24h high/low
    df['high_24h'] = df['high'].rolling(window=288).max()
    df['low_24h'] = df['low'].rolling(window=288).min()
    df['change_since_high'] = (df['close'] - df['high_24h']) / df['high_24h']
    df['change_since_low'] = (df['close'] - df['low_24h']) / df['low_24h']

    # 1-hour price slope (linear regression slope over 12 periods)
    df['price_slope_1h'] = df['close'].rolling(window=12).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])

    # Z-Score 1h window
    df['zscore_1h'] = (df['close'] - df['close'].rolling(12).mean()) / df['close'].rolling(12).std()

    # Kurtosis and Skewness (rolling window of 288 / 24h)
    df['kurtosis_24h'] = df['close'].rolling(288).apply(lambda x: kurtosis(x), raw=True)
    df['skew_24h'] = df['close'].rolling(288).apply(lambda x: skew(x), raw=True)

    # Fibonacci levels relative to 24h high/low
    fib_range = df['high_24h'] - df['low_24h']
    df['fib_distance_0_236'] = (df['close'] - (df['high_24h'] - fib_range * 0.236)) / fib_range
    df['fib_distance_0_382'] = (df['close'] - (df['high_24h'] - fib_range * 0.382)) / fib_range
    df['fib_distance_0_5']   = (df['close'] - (df['high_24h'] - fib_range * 0.5)) / fib_range
    df['fib_distance_0_618'] = (df['close'] - (df['high_24h'] - fib_range * 0.618)) / fib_range
    df['fib_distance_0_786'] = (df['close'] - (df['high_24h'] - fib_range * 0.786)) / fib_range

    # Trim warmup period
    df = df[df.index >= pd.Timestamp("2025-01-14 00:00:00", tz="UTC")]

    output_path = os.path.join(OUTPUT_FOLDER, f"{coin}_enriched.csv")
    df.to_csv(output_path)
    print(f"✅ Saved enriched file for {coin}")

print("🚀 All enrichment complete.")
