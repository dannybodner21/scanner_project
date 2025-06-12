import pandas as pd
import numpy as np
import talib
import os

input_folder = 'exported_data'
output_folder = 'final_data_enriched'
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if not file.endswith('.csv'):
        continue

    filepath = os.path.join(input_folder, file)
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Recalculate indicators exactly as before:
    df['sma_5'] = talib.SMA(df['close'], timeperiod=5)
    df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
    df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
    df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
    df['ema_crossover_flag'] = (df['ema_12'] > df['ema_26']).astype(int)
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    macd, macd_signal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'])
    df['stochastic_k'] = slowk
    df['stochastic_d'] = slowd
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
    df['bollinger_upper'] = upper
    df['bollinger_middle'] = middle
    df['bollinger_lower'] = lower
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['atr_1h'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=12)
    df['stddev_1h'] = talib.STDDEV(df['close'], timeperiod=12)
    df['momentum_10'] = talib.MOM(df['close'], timeperiod=10)
    df['momentum_50'] = talib.MOM(df['close'], timeperiod=50)
    df['roc'] = talib.ROC(df['close'], timeperiod=10)

    df['rolling_volatility_5h'] = df['close'].rolling(60).std()
    df['rolling_volatility_24h'] = df['close'].rolling(288).std()
    df['high_low_ratio'] = df['high'] / df['low']
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    df['candle_body_size'] = abs(df['close'] - df['open'])
    df['candle_body_pct'] = df['candle_body_size'] / (df['high'] - df['low'] + 1e-10)
    df['wick_upper'] = df['high'] - np.maximum(df['open'], df['close'])
    df['wick_lower'] = np.minimum(df['open'], df['close']) - df['low']

    df['slope_5h'] = df['close'].diff(60)
    df['slope_24h'] = df['close'].diff(288)
    df['trend_acceleration'] = df['slope_24h'] - df['slope_5h']

    rolling_high = df['high'].rolling(288).max()
    rolling_low = df['low'].rolling(288).min()
    fib_0_236 = rolling_high - (rolling_high - rolling_low) * 0.236
    fib_0_382 = rolling_high - (rolling_high - rolling_low) * 0.382
    fib_0_618 = rolling_high - (rolling_high - rolling_low) * 0.618
    df['fib_distance_0_236'] = df['close'] - fib_0_236
    df['fib_distance_0_382'] = df['close'] - fib_0_382
    df['fib_distance_0_618'] = df['close'] - fib_0_618

    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['volume_price_ratio'] = df['volume'] / df['close']
    df['volume_change_5m'] = df['volume'].pct_change()
    df['volume_surge'] = df['volume'] / df['volume'].rolling(288).mean()

    df['overbought_rsi'] = (df['rsi'] > 70).astype(int)
    df['oversold_rsi'] = (df['rsi'] < 30).astype(int)
    df['upper_bollinger_break'] = (df['close'] > df['bollinger_upper']).astype(int)
    df['lower_bollinger_break'] = (df['close'] < df['bollinger_lower']).astype(int)
    df['atr_normalized'] = df['atr_1h'] / df['close']

    df['short_vs_long_strength'] = (df['sma_5'] / df['sma_20']) - 1

    # Trim first 288 rows
    df_final = df.iloc[288:].copy()

    # Export
    out_path = os.path.join(output_folder, file)
    df_final.to_csv(out_path, index=False)
    print(f"✅ Enriched and exported {file}")
