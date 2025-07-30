#!/usr/bin/env python3
"""
Standalone Improved SHORT Model Training
Trains a better SHORT model without Django dependencies
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def add_short_specific_features(df):
    """Add SHORT-specific technical features"""
    
    # Basic price/volume columns
    close = df['close']
    high = df['high'] 
    low = df['low']
    volume = df['volume']
    open_price = df['open']
    
    # Simple moving averages
    df['sma_9'] = close.rolling(9).mean()
    df['sma_21'] = close.rolling(21).mean()
    df['sma_50'] = close.rolling(50).mean()
    
    # EMA ratios for trend
    df['ema_9'] = close.ewm(span=9).mean()
    df['ema_21'] = close.ewm(span=21).mean()
    df['ema_9_21_ratio'] = df['ema_9'] / df['ema_21']
    
    # RSI 
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['rsi_14'] = calculate_rsi(close)
    df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
    
    # Volume analysis
    df['volume_sma_20'] = volume.rolling(20).mean()
    df['volume_ratio'] = volume / df['volume_sma_20']
    df['volume_spike'] = (df['volume_ratio'] > 1.5).astype(int)
    
    # Price action patterns
    df['body_size'] = abs(close - open_price) / open_price
    df['upper_shadow'] = (high - np.maximum(close, open_price)) / open_price
    
    # Bollinger Bands position
    df['bb_middle'] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # SHORT-specific patterns
    df['distribution_candle'] = (
        (close < open_price) & 
        (df['upper_shadow'] > df['body_size'] * 1.5) &
        (df['volume_ratio'] > 1.2)
    ).astype(int)
    
    df['buying_exhaustion'] = (
        (df['rsi_14'] > 75) &
        (df['bb_position'] > 0.8) &
        (df['volume_ratio'] > 1.3)
    ).astype(int)
    
    df['trend_weakness'] = (
        (df['ema_9_21_ratio'] < 1.0) &
        (df['rsi_14'] < df['rsi_14'].shift(3))
    ).astype(int)
    
    return df


def get_improved_short_labels(df, forward_periods=24):
    """Create improved labels for SHORT trades"""
    labels = pd.Series(index=df.index, dtype="float64")
    
    # Improved SHORT parameters
    tp_pct = 0.015  # 1.5% take profit
    sl_pct = 0.008  # 0.8% stop loss
    
    for i in range(len(df) - forward_periods):
        entry_price = df.iloc[i]['close']
        tp_price = entry_price * (1 - tp_pct)  # SHORT: profit on drop
        sl_price = entry_price * (1 + sl_pct)  # SHORT: loss on rise
        
        future_highs = df['high'].iloc[i+1:i+1+forward_periods].values
        future_lows = df['low'].iloc[i+1:i+1+forward_periods].values
        
        hit_tp = False
        hit_sl = False
        
        for high, low in zip(future_highs, future_lows):
            if high >= sl_price:  # Check SL first
                hit_sl = True
                break
            if low <= tp_price:  # Then check TP
                hit_tp = True
                break
        
        if hit_tp:
            labels.iloc[i] = 1  # WIN
        else:
            labels.iloc[i] = 0  # LOSS
    
    # Mark incomplete data as NaN
    labels.iloc[-forward_periods:] = np.nan
    return labels


def train_simple_short_model():
    """Train a simple SHORT model using existing data"""
    
    print("🔻 TRAINING IMPROVED SHORT MODEL")
    print("=" * 50)
    
    # Load existing SHORT data if available
    data_file = 'short_three_enhanced_predictions.csv'
    
    try:
        df = pd.read_csv(data_file, parse_dates=['timestamp'])
        print(f"📊 Loaded {len(df)} existing predictions")
    except FileNotFoundError:
        print("❌ No existing SHORT data found")
        return None
    
    # Use existing raw OHLCV data to create better features and labels
    baseline_file = 'baseline_ohlcv.csv'
    
    try:
        raw_df = pd.read_csv(baseline_file, parse_dates=['timestamp'])
        print(f"📊 Loaded {len(raw_df)} raw OHLCV records")
    except FileNotFoundError:
        print("❌ No baseline OHLCV data found")
        return None
    
    # Process each coin separately 
    coin_results = []
    
    for coin in raw_df['coin'].unique():
        print(f"\n  Processing {coin}...")
        coin_df = raw_df[raw_df['coin'] == coin].copy()
        coin_df = coin_df.sort_values('timestamp').reset_index(drop=True)
        
        if len(coin_df) < 1000:
            print(f"    ⚠️ Insufficient data for {coin}")
            continue
        
        # Add improved features
        coin_df = add_short_specific_features(coin_df)
        
        # Add improved labels
        coin_df['label'] = get_improved_short_labels(coin_df)
        
        # Remove NaN labels
        coin_df = coin_df.dropna(subset=['label'])
        
        if len(coin_df) < 500:
            print(f"    ⚠️ Insufficient labeled data for {coin}")
            continue
        
        win_rate = coin_df['label'].mean() * 100
        print(f"    ✅ {coin}: {len(coin_df)} samples, {win_rate:.1f}% win rate")
        
        coin_results.append(coin_df)
    
    if not coin_results:
        print("❌ No valid data for any coin")
        return None
    
    # Combine all coins
    full_df = pd.concat(coin_results, ignore_index=True)
    
    # Split by time (80/20)
    split_idx = int(len(full_df) * 0.8)
    train_df = full_df.iloc[:split_idx]
    test_df = full_df.iloc[split_idx:]
    
    print(f"\n📊 DATASET SUMMARY:")
    print(f"   Total samples: {len(full_df)}")
    print(f"   Training: {len(train_df)}")
    print(f"   Testing: {len(test_df)}")
    print(f"   Overall win rate: {full_df['label'].mean()*100:.2f}%")
    print(f"   Training win rate: {train_df['label'].mean()*100:.2f}%")
    
    # Simple feature selection (use most important features)
    feature_cols = [
        'ema_9_21_ratio', 'rsi_14', 'volume_ratio', 'bb_position',
        'distribution_candle', 'buying_exhaustion', 'trend_weakness',
        'rsi_overbought', 'volume_spike', 'body_size', 'upper_shadow'
    ]
    
    # Ensure all features exist
    available_features = [col for col in feature_cols if col in train_df.columns]
    print(f"   Available features: {len(available_features)}")
    
    if len(available_features) < 5:
        print("❌ Not enough features available")
        return None
    
    X_train = train_df[available_features].fillna(0)
    y_train = train_df['label']
    X_test = test_df[available_features].fillna(0)
    y_test = test_df['label']
    
    # Simple model training (using basic algorithms that don't need special libraries)
    print(f"\n🤖 TRAINING SIMPLE MODEL...")
    
    # Calculate feature importance via correlation
    feature_importance = {}
    for feature in available_features:
        corr = np.corrcoef(X_train[feature], y_train)[0, 1]
        if not np.isnan(corr):
            feature_importance[feature] = abs(corr)
    
    # Select top features
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in sorted_features[:8]]
    
    print(f"   Top features: {top_features}")
    
    # Simple threshold-based model
    best_score = 0
    best_thresholds = {}
    
    # Test different threshold combinations for top features
    for feature in top_features[:3]:  # Use top 3 features
        feature_values = X_train[feature]
        
        # Test percentile thresholds
        for percentile in [60, 70, 75, 80, 85, 90]:
            threshold = np.percentile(feature_values, percentile)
            
            if feature in ['rsi_14', 'bb_position']:  # HIGH values indicate SHORT opportunity
                predictions = (X_train[feature] >= threshold).astype(int)
            else:  # Other features might work differently
                predictions = (X_train[feature] >= threshold).astype(int)
            
            if predictions.sum() > 0:  # Must have some predictions
                accuracy = (predictions == y_train).mean()
                if accuracy > best_score:
                    best_score = accuracy
                    best_thresholds[feature] = threshold
    
    print(f"   Best training accuracy: {best_score:.3f}")
    print(f"   Best thresholds: {best_thresholds}")
    
    # Create final predictions
    if best_thresholds:
        # Use the best single feature threshold
        best_feature = list(best_thresholds.keys())[0]
        best_threshold = best_thresholds[best_feature]
        
        test_predictions = (X_test[best_feature] >= best_threshold).astype(int)
        test_accuracy = (test_predictions == y_test).mean()
        
        print(f"\n✅ TEST RESULTS:")
        print(f"   Using feature: {best_feature}")
        print(f"   Threshold: {best_threshold:.4f}")
        print(f"   Test accuracy: {test_accuracy:.3f}")
        print(f"   Test predictions: {test_predictions.sum()}")
        
        # Calculate win rate for positive predictions
        if test_predictions.sum() > 0:
            positive_win_rate = y_test[test_predictions == 1].mean()
            print(f"   Win rate for predictions: {positive_win_rate:.3f}")
            
            # Save model parameters
            model_params = {
                'feature': best_feature,
                'threshold': best_threshold,
                'train_accuracy': best_score,
                'test_accuracy': test_accuracy,
                'test_win_rate': positive_win_rate
            }
            
            # Create test predictions file
            test_results = test_df[['timestamp', 'coin', 'open', 'high', 'low', 'close', 'volume']].copy()
            test_results['prediction'] = test_predictions
            test_results['prediction_prob'] = X_test[best_feature]
            test_results['label'] = y_test
            
            # Add important features
            for feature in available_features[:5]:
                test_results[feature] = X_test[feature].values
            
            test_results.to_csv('improved_short_test_results.csv', index=False)
            print(f"   Saved results: improved_short_test_results.csv")
            
            return model_params
    
    print("❌ No viable model found")
    return None


if __name__ == "__main__":
    result = train_simple_short_model()
    
    if result:
        print(f"\n🎉 SUCCESS! Improved SHORT model trained")
        print(f"📊 Test accuracy: {result['test_accuracy']:.3f}")
        print(f"🎯 Win rate: {result['test_win_rate']:.3f}")
    else:
        print("\n❌ Failed to train improved SHORT model")