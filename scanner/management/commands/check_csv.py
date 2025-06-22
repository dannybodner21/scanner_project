# Load your dataset
import pandas as pd

df = pd.read_csv("four_long_training_data.csv")

features_to_check = [
    # Price and return-based
    'returns_5m', 'returns_15m', 'returns_1h', 'returns_4h',
    'momentum', 'price_change_5', 'volume_change_5',

    # Volume and volatility
    'volume_ma_20', 'vol_spike', 'volatility',

    # Trend indicators
    'ema_9', 'ema_21', 'ema_diff', 'ma_200',
    'slope_1h',

    # MACD
    'macd', 'macd_signal', 'macd_hist',

    # RSI
    'rsi_14',

    # Bollinger Bands
    'bb_upper', 'bb_lower',

    # ATR, ADX
    'atr_14', 'adx_14',

    # OBV
    'obv', 'obv_slope',

    # Stochastic Oscillator
    'stoch_k', 'stoch_d',

    # Market regime
    'bull_regime', 'bear_regime', 'sideways_regime',

    # Relative range + VWAP
    'high_1h', 'low_1h', 'pos_in_range_1h',
    'vwap_1h', 'pos_vs_vwap',

    # Distance from 24h high/low
    'dist_from_high_24h', 'dist_from_low_24h'
]


# 1. Nulls
print("🔍 Missing values:")
print(df[features_to_check].isnull().sum())

# 2. All-zero columns
print("\n🧨 All-zero columns:")
zero_cols = df[features_to_check].columns[(df[features_to_check] == 0).all()]
print(zero_cols.tolist())

# 3. Columns with many zeros (>5%)
print("\n⚠️ High zero-percentage columns:")
for col in features_to_check:
    if col in df.columns:
        zero_pct = (df[col] == 0).mean()
        if zero_pct > 0.05:
            print(f"{col}: {zero_pct:.2%} zeros")

# 4. Duplicates
print(f"\n🔁 Duplicate rows: {df.duplicated().sum()}")

# 5. Summary stats
print("\n📊 Summary statistics:")
print(df[features_to_check].describe())
