#!/usr/bin/env python3
"""
Combine Long and Short Model Predictions for Regime Detection

This script combines predictions from long and short models to determine market regime:
- Bullish: Long confidence > Short confidence + threshold
- Bearish: Short confidence > Long confidence + threshold  
- Neutral: Neither model significantly stronger

Features:
- Weighted averaging over 4 hours (48 candles at 5-min intervals)
- Proper confidence score normalization between models
- Regime detection with configurable thresholds
- Outputs combined prediction CSV with regime column
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import os

def normalize_confidence_scores(conf_scores, target_mean=0.5, target_std=0.2):
    """
    Normalize confidence scores to have consistent distribution across models.
    
    Args:
        conf_scores: Array of confidence scores
        target_mean: Target mean for normalized scores
        target_std: Target standard deviation for normalized scores
    
    Returns:
        Normalized confidence scores
    """
    if len(conf_scores) == 0 or np.std(conf_scores) == 0:
        return conf_scores
    
    # Z-score normalization
    z_scores = (conf_scores - np.mean(conf_scores)) / np.std(conf_scores)
    
    # Rescale to target distribution
    normalized = z_scores * target_std + target_mean
    
    # Clip to [0, 1] range
    return np.clip(normalized, 0, 1)

def calculate_weighted_average(df, value_col, confidence_col, window_hours=4):
    """
    Calculate weighted average over specified time window.
    
    Args:
        df: DataFrame with predictions
        value_col: Column name for values to average
        confidence_col: Column name for confidence weights
        window_hours: Time window in hours (default 4)
    
    Returns:
        Series with weighted averages
    """
    window_candles = window_hours * 12  # 12 candles per hour (5-min intervals)
    
    # Create weights based on confidence and recency
    weights = df[confidence_col].copy()
    
    # Add recency weight (more recent = higher weight)
    recency_weights = np.linspace(0.5, 1.0, window_candles)
    if len(weights) >= window_candles:
        weights.iloc[-window_candles:] *= recency_weights
    else:
        weights *= np.linspace(0.5, 1.0, len(weights))
    
    # Calculate rolling weighted average
    def weighted_avg(group):
        if len(group) == 0:
            return np.nan
        values = group[value_col].values
        w = group[confidence_col].values
        if np.sum(w) == 0:
            return np.mean(values)
        return np.average(values, weights=w)
    
    # Use expanding window for early periods, then rolling window
    result = []
    for i in range(len(df)):
        start_idx = max(0, i - window_candles + 1)
        window_data = df.iloc[start_idx:i+1]
        result.append(weighted_avg(window_data))
    
    return pd.Series(result, index=df.index)

def determine_regime(long_conf, short_conf, long_threshold=0.6, short_threshold=0.6, neutral_buffer=0.1):
    """
    Determine market regime based on long and short confidence scores.
    
    Args:
        long_conf: Long model confidence score
        short_conf: Short model confidence score
        long_threshold: Minimum confidence for bullish regime
        short_threshold: Minimum confidence for bearish regime
        neutral_buffer: Buffer zone for neutral regime
    
    Returns:
        String: 'bullish', 'bearish', or 'neutral'
    """
    if pd.isna(long_conf) or pd.isna(short_conf):
        return 'neutral'
    
    # Check if either model meets minimum confidence threshold
    long_strong = long_conf >= long_threshold
    short_strong = short_conf >= short_threshold
    
    # Calculate relative strength
    diff = long_conf - short_conf
    
    if long_strong and not short_strong and diff > neutral_buffer:
        return 'bullish'
    elif short_strong and not long_strong and diff < -neutral_buffer:
        return 'bearish'
    elif long_strong and short_strong:
        # Both models confident - use relative strength
        if diff > neutral_buffer:
            return 'bullish'
        elif diff < -neutral_buffer:
            return 'bearish'
        else:
            return 'neutral'
    else:
        return 'neutral'

def combine_predictions(long_file, short_file, output_file, 
                       window_hours=4, long_threshold=0.6, short_threshold=0.6,
                       neutral_buffer=0.1, normalize_confidence=True):
    """
    Combine long and short prediction files into a single regime-aware prediction file.
    
    Args:
        long_file: Path to long model predictions CSV
        short_file: Path to short model predictions CSV  
        output_file: Path for output combined predictions CSV
        window_hours: Hours to average over (default 4)
        long_threshold: Minimum confidence for bullish regime
        short_threshold: Minimum confidence for bearish regime
        neutral_buffer: Buffer zone for neutral regime
        normalize_confidence: Whether to normalize confidence scores between models
    """
    
    print(f"📊 Loading prediction files...")
    
    # Load prediction files
    long_df = pd.read_csv(long_file)
    short_df = pd.read_csv(short_file)
    
    print(f"   Long predictions: {len(long_df)} rows")
    print(f"   Short predictions: {len(short_df)} rows")
    
    # Convert timestamps
    long_df['timestamp'] = pd.to_datetime(long_df['timestamp'])
    short_df['timestamp'] = pd.to_datetime(short_df['timestamp'])
    
    print(f"   Long time range: {long_df['timestamp'].min()} to {long_df['timestamp'].max()}")
    print(f"   Short time range: {short_df['timestamp'].min()} to {short_df['timestamp'].max()}")
    
    # Set timestamp as index for easier merging
    long_df = long_df.set_index('timestamp')
    short_df = short_df.set_index('timestamp')
    
    # Create full time range (5-minute intervals)
    start_time = min(long_df.index.min(), short_df.index.min())
    end_time = max(long_df.index.max(), short_df.index.max())
    full_range = pd.date_range(start=start_time, end=end_time, freq='5min')
    
    print(f"   Combined time range: {start_time} to {end_time}")
    print(f"   Total candles: {len(full_range)}")
    
    # Reindex both dataframes to full time range
    long_df = long_df.reindex(full_range)
    short_df = short_df.reindex(full_range)
    
    # Forward fill missing values (assume last prediction holds)
    long_df = long_df.ffill()
    short_df = short_df.ffill()
    
    # Handle confidence scores
    if 'confidence' in long_df.columns:
        long_conf = long_df['confidence'].fillna(0.0)
    else:
        # Calculate confidence from prediction probability
        long_conf = np.abs(long_df['pred_prob'] - 0.5) * 2
        long_conf = long_conf.fillna(0.0)
    
    if 'confidence' in short_df.columns:
        short_conf = short_df['confidence'].fillna(0.0)
    else:
        # Calculate confidence from prediction probability  
        short_conf = np.abs(short_df['pred_prob'] - 0.5) * 2
        short_conf = short_conf.fillna(0.0)
    
    # Normalize confidence scores if requested
    if normalize_confidence:
        print("🔧 Normalizing confidence scores between models...")
        long_conf = normalize_confidence_scores(long_conf.values)
        short_conf = normalize_confidence_scores(short_conf.values)
    
    # Calculate weighted averages
    print(f"📈 Calculating weighted averages over {window_hours} hours...")
    
    # Create temporary dataframe for weighted average calculation
    temp_df = pd.DataFrame({
        'long_prob': long_df['pred_prob'],
        'short_prob': short_df['pred_prob'],
        'long_conf': long_conf,
        'short_conf': short_conf
    }, index=full_range)
    
    # Calculate weighted averages
    long_avg = calculate_weighted_average(temp_df, 'long_prob', 'long_conf', window_hours)
    short_avg = calculate_weighted_average(temp_df, 'short_prob', 'short_conf', window_hours)
    long_conf_avg = calculate_weighted_average(temp_df, 'long_conf', 'long_conf', window_hours)
    short_conf_avg = calculate_weighted_average(temp_df, 'short_conf', 'short_conf', window_hours)
    
    # Determine regime for each timestamp
    print("🎯 Determining market regime...")
    regimes = []
    for i in range(len(full_range)):
        regime = determine_regime(
            long_conf_avg.iloc[i], 
            short_conf_avg.iloc[i],
            long_threshold, 
            short_threshold, 
            neutral_buffer
        )
        regimes.append(regime)
    
    # Create combined dataframe
    combined_df = pd.DataFrame({
        'coin': 'XRPUSDT',
        'timestamp': full_range,
        'long_prob': long_df['pred_prob'].values,
        'short_prob': short_df['pred_prob'].values,
        'long_confidence': long_conf,
        'short_confidence': short_conf,
        'long_confidence_avg': long_conf_avg.values,
        'short_confidence_avg': short_conf_avg.values,
        'regime': regimes
    })
    
    # Add trading signals based on regime
    combined_df['long_signal'] = (combined_df['regime'] == 'bullish').astype(int)
    combined_df['short_signal'] = (combined_df['regime'] == 'bearish').astype(int)
    
    # Save combined predictions
    combined_df.to_csv(output_file, index=False)
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"   Total predictions: {len(combined_df)}")
    print(f"   Bullish periods: {(combined_df['regime'] == 'bullish').sum()}")
    print(f"   Bearish periods: {(combined_df['regime'] == 'bearish').sum()}")
    print(f"   Neutral periods: {(combined_df['regime'] == 'neutral').sum()}")
    print(f"   Long signals: {combined_df['long_signal'].sum()}")
    print(f"   Short signals: {combined_df['short_signal'].sum()}")
    
    print(f"\n📈 Confidence Score Statistics:")
    print(f"   Long confidence - Mean: {combined_df['long_confidence'].mean():.3f}, Std: {combined_df['long_confidence'].std():.3f}")
    print(f"   Short confidence - Mean: {combined_df['short_confidence'].mean():.3f}, Std: {combined_df['short_confidence'].std():.3f}")
    print(f"   Long confidence avg - Mean: {combined_df['long_confidence_avg'].mean():.3f}, Std: {combined_df['long_confidence_avg'].std():.3f}")
    print(f"   Short confidence avg - Mean: {combined_df['short_confidence_avg'].mean():.3f}, Std: {combined_df['short_confidence_avg'].std():.3f}")
    
    print(f"\n✅ Combined predictions saved to: {output_file}")
    
    return combined_df

def main():
    parser = argparse.ArgumentParser(description='Combine long and short model predictions for regime detection')
    parser.add_argument('--long_file', default='xrp_predictions.csv', help='Long model predictions CSV')
    parser.add_argument('--short_file', default='xrp_simple_short_predictions.csv', help='Short model predictions CSV')
    parser.add_argument('--output_file', default='xrp_combined_predictions.csv', help='Output combined predictions CSV')
    parser.add_argument('--window_hours', type=int, default=4, help='Hours to average over (default 4)')
    parser.add_argument('--long_threshold', type=float, default=0.6, help='Long confidence threshold for bullish (default 0.6)')
    parser.add_argument('--short_threshold', type=float, default=0.6, help='Short confidence threshold for bearish (default 0.6)')
    parser.add_argument('--neutral_buffer', type=float, default=0.1, help='Buffer zone for neutral regime (default 0.1)')
    parser.add_argument('--no_normalize', action='store_true', help='Disable confidence score normalization')
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.long_file):
        print(f"❌ Error: Long file not found: {args.long_file}")
        return 1
    
    if not os.path.exists(args.short_file):
        print(f"❌ Error: Short file not found: {args.short_file}")
        return 1
    
    try:
        combined_df = combine_predictions(
            long_file=args.long_file,
            short_file=args.short_file,
            output_file=args.output_file,
            window_hours=args.window_hours,
            long_threshold=args.long_threshold,
            short_threshold=args.short_threshold,
            neutral_buffer=args.neutral_buffer,
            normalize_confidence=not args.no_normalize
        )
        
        print(f"\n🎉 Successfully created combined predictions!")
        print(f"   Output file: {args.output_file}")
        print(f"   Rows: {len(combined_df)}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
