#!/usr/bin/env python3
"""
Improve the regime filter by:
1. Lowering confidence thresholds
2. Adding a "weak" regime for low-confidence trades
3. Using relative strength instead of absolute thresholds
4. Allowing more trading opportunities
"""

import pandas as pd
import numpy as np

def improve_regime_detection(df, long_threshold=0.3, short_threshold=0.3, 
                           relative_strength_threshold=0.1, min_confidence=0.2):
    """
    Improved regime detection that allows more trading opportunities.
    
    Args:
        df: DataFrame with long_confidence_avg and short_confidence_avg
        long_threshold: Minimum confidence for strong bullish
        short_threshold: Minimum confidence for strong bearish  
        relative_strength_threshold: Minimum difference for regime decision
        min_confidence: Minimum confidence to trade at all
    """
    
    regimes = []
    long_signals = []
    short_signals = []
    
    for i, row in df.iterrows():
        long_conf = row['long_confidence_avg']
        short_conf = row['short_confidence_avg']
        
        if pd.isna(long_conf) or pd.isna(short_conf):
            regimes.append('neutral')
            long_signals.append(0)
            short_signals.append(0)
            continue
            
        # Calculate relative strength
        diff = long_conf - short_conf
        
        # Determine regime based on relative strength and minimum confidence
        if long_conf >= min_confidence and diff > relative_strength_threshold:
            if long_conf >= long_threshold:
                regimes.append('bullish_strong')
                long_signals.append(1)
                short_signals.append(0)
            else:
                regimes.append('bullish_weak')
                long_signals.append(1)
                short_signals.append(0)
        elif short_conf >= min_confidence and diff < -relative_strength_threshold:
            if short_conf >= short_threshold:
                regimes.append('bearish_strong')
                long_signals.append(0)
                short_signals.append(1)
            else:
                regimes.append('bearish_weak')
                long_signals.append(0)
                short_signals.append(1)
        else:
            regimes.append('neutral')
            long_signals.append(0)
            short_signals.append(0)
    
    return regimes, long_signals, short_signals

def main():
    # Load the combined predictions
    df = pd.read_csv('xrp_combined_predictions.csv')
    
    print("🔧 Improving regime detection...")
    
    # Test different parameter combinations
    test_configs = [
        {"long_threshold": 0.3, "short_threshold": 0.3, "relative_strength_threshold": 0.05, "min_confidence": 0.2},
        {"long_threshold": 0.4, "short_threshold": 0.4, "relative_strength_threshold": 0.1, "min_confidence": 0.25},
        {"long_threshold": 0.5, "short_threshold": 0.5, "relative_strength_threshold": 0.15, "min_confidence": 0.3},
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n📊 Configuration {i+1}: {config}")
        
        regimes, long_signals, short_signals = improve_regime_detection(df, **config)
        
        # Count regimes
        regime_counts = pd.Series(regimes).value_counts()
        total_signals = sum(long_signals) + sum(short_signals)
        
        print(f"   Regime distribution:")
        for regime, count in regime_counts.items():
            pct = (count / len(regimes)) * 100
            print(f"     {regime}: {count:,} ({pct:.1f}%)")
        
        print(f"   Total signals: {total_signals:,} ({(total_signals/len(regimes)*100):.1f}%)")
        print(f"   Long signals: {sum(long_signals):,}")
        print(f"   Short signals: {sum(short_signals):,}")
        
        # Save improved version
        df_improved = df.copy()
        df_improved['regime'] = regimes
        df_improved['long_signal'] = long_signals
        df_improved['short_signal'] = short_signals
        
        output_file = f'xrp_improved_regime_{i+1}.csv'
        df_improved.to_csv(output_file, index=False)
        print(f"   Saved to: {output_file}")

if __name__ == "__main__":
    main()
