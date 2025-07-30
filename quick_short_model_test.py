#!/usr/bin/env python3
"""
Quick SHORT Model Improvement Test
Test if improved labeling can make SHORT trades profitable
"""

import pandas as pd
import numpy as np

def test_improved_short_labeling():
    """Test improved SHORT labeling on existing data"""
    
    print("🔻 QUICK SHORT MODEL IMPROVEMENT TEST")
    print("=" * 50)
    
    # Load existing SHORT predictions
    try:
        df = pd.read_csv('short_three_enhanced_predictions.csv', parse_dates=['timestamp'])
        print(f"📊 Loaded {len(df)} SHORT predictions")
    except FileNotFoundError:
        print("❌ No SHORT predictions file found")
        return None
    
    # Test different labeling strategies on this data
    print(f"\n📊 TESTING DIFFERENT LABELING STRATEGIES:")
    
    strategies = [
        {'name': 'Original (2.0%/1.5%)', 'tp': 0.020, 'sl': 0.015},
        {'name': 'Improved (1.5%/0.8%)', 'tp': 0.015, 'sl': 0.008},
        {'name': 'Conservative (1.0%/0.6%)', 'tp': 0.010, 'sl': 0.006},
        {'name': 'Aggressive (2.5%/1.0%)', 'tp': 0.025, 'sl': 0.010},
    ]
    
    results = []
    
    for strategy in strategies:
        print(f"\n  Testing {strategy['name']}...")
        tp_pct = strategy['tp']
        sl_pct = strategy['sl']
        
        wins = 0
        losses = 0
        total_pnl = 0
        
        for _, row in df.iterrows():
            entry_price = row['open']
            high = row['high']
            low = row['low']
            
            tp_price = entry_price * (1 - tp_pct)  # SHORT: profit on drop
            sl_price = entry_price * (1 + sl_pct)  # SHORT: loss on rise
            
            # Check what happens first
            hit_tp = low <= tp_price
            hit_sl = high >= sl_price
            
            if hit_tp and not hit_sl:
                wins += 1
                total_pnl += tp_pct  # Profit
            elif hit_sl:
                losses += 1
                total_pnl -= sl_pct  # Loss
            else:
                losses += 1  # No clear direction = loss
        
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        print(f"    Wins: {wins}")
        print(f"    Losses: {losses}")
        print(f"    Win Rate: {win_rate:.2f}%")
        print(f"    Total P&L: {total_pnl:.4f}")
        print(f"    Avg P&L per trade: {total_pnl/total_trades:.6f}")
        
        results.append({
            'strategy': strategy['name'],
            'tp': tp_pct,
            'sl': sl_pct,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl/total_trades if total_trades > 0 else 0
        })
    
    # Find best strategy
    best = max(results, key=lambda x: x['total_pnl'])
    
    print(f"\n🏆 BEST STRATEGY: {best['strategy']}")
    print(f"   TP/SL: {best['tp']*100:.1f}%/{best['sl']*100:.1f}%")
    print(f"   Win Rate: {best['win_rate']:.2f}%")
    print(f"   Total P&L: {best['total_pnl']:.4f}")
    print(f"   Avg P&L: {best['avg_pnl']:.6f}")
    
    if best['total_pnl'] > 0:
        print(f"   ✅ PROFITABLE!")
    else:
        print(f"   ❌ Still losing")
    
    return best


def test_filtering_impact():
    """Test impact of confidence filtering on profitability"""
    
    print(f"\n🔍 TESTING CONFIDENCE FILTERING IMPACT:")
    
    try:
        df = pd.read_csv('short_three_enhanced_predictions.csv', parse_dates=['timestamp'])
    except FileNotFoundError:
        print("❌ No data available")
        return None
    
    # Use best parameters from previous test
    tp_pct = 0.015
    sl_pct = 0.008
    
    confidence_thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    for threshold in confidence_thresholds:
        filtered_df = df[df['prediction_prob'] >= threshold]
        
        if len(filtered_df) == 0:
            print(f"   Threshold {threshold}: No trades")
            continue
        
        wins = 0
        losses = 0
        total_pnl = 0
        
        for _, row in filtered_df.iterrows():
            entry_price = row['open']
            high = row['high']
            low = row['low']
            
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)
            
            hit_tp = low <= tp_price
            hit_sl = high >= sl_price
            
            if hit_tp and not hit_sl:
                wins += 1
                total_pnl += tp_pct
            elif hit_sl:
                losses += 1
                total_pnl -= sl_pct
            else:
                losses += 1
        
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        print(f"   Threshold {threshold}: {total_trades} trades, {win_rate:.1f}% win rate, P&L: {total_pnl:.4f}")
    
    return None


if __name__ == "__main__":
    # Test improved labeling
    best_strategy = test_improved_short_labeling()
    
    if best_strategy:
        # Test filtering impact
        test_filtering_impact()
    
    print(f"\n✅ Analysis complete!")