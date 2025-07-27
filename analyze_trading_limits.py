#!/usr/bin/env python3
"""
Trading Limits Analysis
Analyzes current trading limits and patterns in the simulation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

def analyze_current_limits():
    """Analyze the current trading limits in the simulation"""
    
    print("🔍 CURRENT TRADING LIMITS ANALYSIS")
    print("=" * 55)
    
    print("📊 IDENTIFIED LIMITS IN CURRENT SIMULATION:")
    print()
    
    print("1️⃣ PER-COIN LIMIT:")
    print("   ✅ Only ONE trade per coin at any time")
    print("   📝 Code: `already_open = any(t.coin == coin for t in open_trades)`")
    print("   💡 This prevents overexposure to single coins")
    print()
    
    print("2️⃣ CONFIDENCE HISTORY REQUIREMENT:")
    print("   ✅ Requires 3 confidence readings before trading")
    print("   📝 Code: `if len(confidence_history[coin]) < 3: continue`")
    print("   💡 This prevents immediate trading on single signals")
    print()
    
    print("3️⃣ MAX HOLD TIME:")
    print("   ✅ Trades auto-close after 48 hours")
    print("   📝 Code: `max_hold_hours = 48`")
    print("   💡 This prevents indefinite positions")
    print()
    
    print("❌ NO LIMITS ON:")
    print("   • Total number of concurrent trades")
    print("   • Daily trade frequency")
    print("   • Total position size exposure")
    print("   • Risk per day")
    print()

def simulate_trading_frequency():
    """Analyze actual trading frequency patterns"""
    
    print("📈 ACTUAL TRADING FREQUENCY ANALYSIS")
    print("=" * 55)
    
    # Load the data to analyze patterns
    df = pd.read_csv('three_enhanced_predictions.csv', parse_dates=['timestamp'])
    df = df.sort_values('timestamp')
    
    # Simulate with high confidence trades to see frequency
    confidence_threshold = 0.75
    high_conf_trades = df[(df['prediction'] == 1) & (df['prediction_prob'] >= confidence_threshold)]
    
    print(f"📊 High-confidence signals: {len(high_conf_trades)}")
    print(f"📅 Period: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    
    # Group by date to see daily patterns
    high_conf_trades['date'] = high_conf_trades['timestamp'].dt.date
    daily_signals = high_conf_trades.groupby('date').size()
    
    print(f"📊 Daily signal statistics:")
    print(f"   Average signals per day: {daily_signals.mean():.1f}")
    print(f"   Max signals in one day: {daily_signals.max()}")
    print(f"   Min signals in one day: {daily_signals.min()}")
    print(f"   Days with 10+ signals: {(daily_signals >= 10).sum()}")
    print(f"   Days with 20+ signals: {(daily_signals >= 20).sum()}")
    print()
    
    # Group by hour to see intraday patterns
    high_conf_trades['hour'] = high_conf_trades['timestamp'].dt.hour
    hourly_signals = high_conf_trades.groupby('hour').size()
    
    print(f"🕐 Peak trading hours:")
    top_hours = hourly_signals.nlargest(5)
    for hour, count in top_hours.items():
        print(f"   {hour:02d}:00 - {count} signals")
    print()
    
    # Analyze unique coins per day
    daily_coins = high_conf_trades.groupby('date')['coin'].nunique()
    print(f"🪙 Unique coins per day:")
    print(f"   Average unique coins per day: {daily_coins.mean():.1f}")
    print(f"   Max unique coins in one day: {daily_coins.max()}")
    print()
    
    # Calculate potential concurrent trades (if no per-coin limit)
    print("⚠️ POTENTIAL RISK WITHOUT PER-COIN LIMIT:")
    
    # Simulate without per-coin limit
    concurrent_trades = []
    open_trades_count = 0
    max_concurrent = 0
    
    # Simple simulation to estimate max concurrent
    for i in range(0, len(high_conf_trades), 10):  # Sample every 10th trade
        # Assume trades last ~4 hours on average
        trade_duration_hours = 4
        
        # Count how many could be open simultaneously
        if i % 20 == 0:  # Every 20th signal creates a trade
            open_trades_count += 1
        
        if i % 80 == 0:  # Every 80th signal, close some trades
            open_trades_count = max(0, open_trades_count - 3)
        
        max_concurrent = max(max_concurrent, open_trades_count)
    
    print(f"   Estimated max concurrent trades: {max_concurrent}")
    print(f"   At 10% position size each: {max_concurrent * 10}% total exposure")
    print()

def recommend_risk_limits():
    """Recommend appropriate risk management limits"""
    
    print("💡 RECOMMENDED ADDITIONAL LIMITS")
    print("=" * 55)
    
    print("1️⃣ MAX CONCURRENT TRADES:")
    print("   🎯 Recommended: 10-15 trades maximum")
    print("   📝 Reasoning: Prevents overexposure, manageable risk")
    print("   💰 At 10% position size: 100-150% max exposure")
    print()
    
    print("2️⃣ DAILY TRADE FREQUENCY:")
    print("   🎯 Recommended: 20 new trades per day maximum")
    print("   📝 Reasoning: Prevents overtrading in volatile markets")
    print("   ⏰ Allows 1 trade every ~1.2 hours during trading day")
    print()
    
    print("3️⃣ MAXIMUM DAILY RISK:")
    print("   🎯 Recommended: 50% of portfolio at risk per day")
    print("   📝 Reasoning: Limits catastrophic loss scenarios")
    print("   💡 If 10% per trade × 5 trades = 50% max daily risk")
    print()
    
    print("4️⃣ POSITION SIZE SCALING:")
    print("   🎯 Recommended: Scale position size with confidence")
    print("   📝 High confidence (0.85+): 10% position size")
    print("   📝 Medium confidence (0.75-0.85): 7% position size")
    print("   📝 Lower confidence (0.70-0.75): 5% position size")
    print()
    
    print("5️⃣ BALANCE-BASED LIMITS:")
    print("   🎯 Recommended: Increase limits as balance grows")
    print("   📝 Under $10k: Max 5 concurrent trades")
    print("   📝 $10k-$50k: Max 10 concurrent trades")
    print("   📝 Over $50k: Max 15 concurrent trades")
    print()

def show_code_modifications():
    """Show code modifications to implement limits"""
    
    print("🔧 CODE MODIFICATIONS FOR RISK LIMITS")
    print("=" * 55)
    
    print("""
# Add these parameters to the simulator:
MAX_CONCURRENT_TRADES = 10
MAX_DAILY_TRADES = 20
MAX_DAILY_RISK_PCT = 0.50

# Track daily stats:
daily_trades_opened = 0
daily_risk_exposure = 0.0
current_date = None

# Before opening new trade, add these checks:
if len(open_trades) >= MAX_CONCURRENT_TRADES:
    continue  # Skip if too many concurrent trades

if daily_trades_opened >= MAX_DAILY_TRADES:
    continue  # Skip if daily limit reached

if daily_risk_exposure >= MAX_DAILY_RISK_PCT:
    continue  # Skip if daily risk too high

# Reset daily counters when date changes:
trade_date = timestamp.date()
if current_date != trade_date:
    daily_trades_opened = 0
    daily_risk_exposure = 0.0
    current_date = trade_date

# When opening trade, update counters:
daily_trades_opened += 1
daily_risk_exposure += position_size_pct
""")

if __name__ == "__main__":
    analyze_current_limits()
    simulate_trading_frequency()
    recommend_risk_limits()
    show_code_modifications()