#!/usr/bin/env python3
"""
Trade Accuracy Verification Script
Verifies that trades are executing correctly with proper timing, pricing, and calculations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def verify_trade_execution():
    """Verify actual trade execution accuracy"""
    
    print("🔍 TRADE EXECUTION ACCURACY VERIFICATION")
    print("=" * 55)
    
    # Load prediction data
    file_path = 'three_enhanced_predictions.csv'
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp')
    
    print(f"📊 Loaded {len(df)} predictions")
    print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"⏰ Time intervals: {df['timestamp'].diff().mode().iloc[0]}")
    print()
    
    # Test parameters
    confidence_threshold = 0.75
    take_profit = 0.025  # 2.5%
    stop_loss = 0.015    # 1.5%
    leverage = 10.0
    
    # Find high-confidence predictions for detailed analysis
    high_conf_trades = df[(df['prediction'] == 1) & (df['prediction_prob'] >= confidence_threshold)].head(10)
    
    print("🎯 DETAILED TRADE ANALYSIS (First 10 High-Confidence Trades):")
    print("=" * 70)
    
    trade_counter = 0
    
    for idx, entry_row in high_conf_trades.iterrows():
        trade_counter += 1
        coin = entry_row['coin']
        entry_time = entry_row['timestamp']
        entry_price = entry_row['open']
        confidence = entry_row['prediction_prob']
        
        # Calculate target prices
        tp_price = entry_price * (1 + take_profit)
        sl_price = entry_price * (1 - stop_loss)
        
        print(f"\n📈 TRADE #{trade_counter}: {coin}")
        print(f"   Entry Time: {entry_time}")
        print(f"   Entry Price: ${entry_price:.6f}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Target TP: ${tp_price:.6f} (+{take_profit*100:.1f}%)")
        print(f"   Target SL: ${sl_price:.6f} (-{stop_loss*100:.1f}%)")
        
        # Look for future candles for this coin to simulate trade execution
        future_candles = df[(df['coin'] == coin) & (df['timestamp'] > entry_time)].head(100)
        
        if future_candles.empty:
            print("   ❌ No future data available")
            continue
            
        trade_closed = False
        exit_time = None
        exit_price = None
        exit_reason = None
        
        for _, candle in future_candles.iterrows():
            high = candle['high']
            low = candle['low']
            close = candle['close']
            candle_time = candle['timestamp']
            
            # Check if TP or SL was hit
            if high >= tp_price:
                exit_time = candle_time
                exit_price = tp_price
                exit_reason = "TAKE_PROFIT"
                trade_closed = True
                break
            elif low <= sl_price:
                exit_time = candle_time
                exit_price = sl_price
                exit_reason = "STOP_LOSS"
                trade_closed = True
                break
        
        if trade_closed:
            duration = exit_time - entry_time
            duration_minutes = duration.total_seconds() / 60
            
            # Calculate actual returns
            price_change_pct = (exit_price - entry_price) / entry_price
            leveraged_return_pct = price_change_pct * leverage
            
            print(f"   ✅ Trade Closed: {exit_reason}")
            print(f"   Exit Time: {exit_time}")
            print(f"   Exit Price: ${exit_price:.6f}")
            print(f"   Duration: {duration_minutes:.0f} minutes ({duration})")
            print(f"   Price Change: {price_change_pct*100:.2f}%")
            print(f"   Leveraged Return: {leveraged_return_pct*100:.2f}%")
            
            # Verify calculations are correct
            expected_change = take_profit if exit_reason == "TAKE_PROFIT" else -stop_loss
            expected_leveraged = expected_change * leverage
            
            print(f"   📊 VERIFICATION:")
            print(f"      Expected Price Change: {expected_change*100:.2f}%")
            print(f"      Actual Price Change: {price_change_pct*100:.2f}%")
            print(f"      Expected Leveraged Return: {expected_leveraged*100:.2f}%")
            print(f"      Actual Leveraged Return: {leveraged_return_pct*100:.2f}%")
            
            # Check if calculations match
            if abs(price_change_pct - expected_change) < 0.0001:
                print(f"      ✅ Calculations CORRECT")
            else:
                print(f"      ❌ Calculations INCORRECT")
                
            # Check if duration is realistic (should be at least 5 minutes apart)
            if duration_minutes >= 5:
                print(f"      ✅ Duration REALISTIC ({duration_minutes:.0f} min)")
            else:
                print(f"      ⚠️ Duration VERY SHORT ({duration_minutes:.0f} min)")
                
        else:
            print(f"   ⏳ Trade would remain open (no TP/SL hit in next 100 candles)")
    
    print("\n" + "=" * 70)
    print("🔍 DATA QUALITY VERIFICATION:")
    print("=" * 70)
    
    # Check for realistic price movements
    sample_data = df.head(1000)
    
    # Calculate 5-minute returns
    for coin in sample_data['coin'].unique()[:5]:
        coin_data = sample_data[sample_data['coin'] == coin].sort_values('timestamp')
        if len(coin_data) < 2:
            continue
            
        coin_data['price_change_5min'] = coin_data['close'].pct_change()
        coin_data['high_low_range'] = (coin_data['high'] - coin_data['low']) / coin_data['open']
        
        avg_5min_change = coin_data['price_change_5min'].abs().mean() * 100
        avg_range = coin_data['high_low_range'].mean() * 100
        max_5min_change = coin_data['price_change_5min'].abs().max() * 100
        
        print(f"\n📊 {coin} Price Movement Analysis:")
        print(f"   Average 5-min price change: {avg_5min_change:.3f}%")
        print(f"   Average high-low range: {avg_range:.3f}%")
        print(f"   Maximum 5-min change: {max_5min_change:.3f}%")
        
        # Check if our TP/SL targets are realistic
        tp_realistic = avg_range > (take_profit * 100)
        sl_realistic = avg_range > (stop_loss * 100)
        
        print(f"   TP target ({take_profit*100:.1f}%): {'✅ REALISTIC' if tp_realistic else '⚠️ AGGRESSIVE'}")
        print(f"   SL target ({stop_loss*100:.1f}%): {'✅ REALISTIC' if sl_realistic else '⚠️ AGGRESSIVE'}")
    
    print("\n" + "=" * 70)
    print("✅ TRADE ACCURACY VERIFICATION COMPLETE")
    

def check_baseline_data_consistency():
    """Check if baseline OHLCV data matches prediction data"""
    
    print("\n🔍 BASELINE DATA CONSISTENCY CHECK")
    print("=" * 55)
    
    try:
        pred_df = pd.read_csv('three_enhanced_predictions.csv', parse_dates=['timestamp'])
        base_df = pd.read_csv('baseline_ohlcv.csv', parse_dates=['timestamp'])
        
        print(f"📊 Predictions: {len(pred_df)} rows")
        print(f"📊 Baseline: {len(base_df)} rows")
        
        # Check for matching timestamps and coins
        pred_sample = pred_df.head(100)
        consistency_issues = 0
        
        for _, row in pred_sample.iterrows():
            coin = row['coin']
            timestamp = row['timestamp']
            
            # Find matching baseline row
            baseline_match = base_df[(base_df['coin'] == coin) & (base_df['timestamp'] == timestamp)]
            
            if baseline_match.empty:
                consistency_issues += 1
                if consistency_issues <= 3:  # Show first 3 issues
                    print(f"⚠️ Missing baseline data: {coin} at {timestamp}")
            else:
                # Compare OHLC values
                pred_open = row['open']
                base_open = baseline_match.iloc[0]['open']
                
                if abs(pred_open - base_open) > 0.001:
                    print(f"⚠️ Price mismatch: {coin} at {timestamp}")
                    print(f"   Prediction open: {pred_open}")
                    print(f"   Baseline open: {base_open}")
        
        if consistency_issues == 0:
            print("✅ Baseline data is consistent with predictions")
        else:
            print(f"⚠️ Found {consistency_issues} consistency issues")
            
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")


if __name__ == "__main__":
    verify_trade_execution()
    check_baseline_data_consistency()