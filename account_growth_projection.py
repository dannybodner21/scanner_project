#!/usr/bin/env python3
"""
Account Growth Projection Calculator
Projects daily account balance growth based on LONG trade simulator performance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_projected_growth():
    """Calculate projected account growth based on simulator results"""
    
    print("💰 ACCOUNT GROWTH PROJECTION CALCULATOR")
    print("=" * 60)
    
    # === SIMULATOR RESULTS (LONG TRADES) ===
    print("📊 BASE SIMULATION DATA:")
    initial_balance = 5000
    final_balance = 23771.94
    total_return_pct = 375.44
    trades_executed = 73
    win_rate = 60.27
    simulation_days = 51  # June 1 to July 21, 2025
    
    print(f"   Simulation period: {simulation_days} days")
    print(f"   Initial balance: ${initial_balance:,.2f}")
    print(f"   Final balance: ${final_balance:,.2f}")
    print(f"   Total return: {total_return_pct:.2f}%")
    print(f"   Trades executed: {trades_executed}")
    print(f"   Win rate: {win_rate:.2f}%")
    print()
    
    # === TRADE PARAMETERS ===
    print("🎯 TRADE PARAMETERS:")
    leverage = 10.0
    tp_percent = 2.5  # 2.5% price movement
    sl_percent = 1.5  # 1.5% price movement
    leveraged_win = tp_percent * leverage  # 25% return per win
    leveraged_loss = sl_percent * leverage  # -15% return per loss
    position_size_pct = 0.25  # 25% of account per trade
    max_concurrent_trades = 4
    
    print(f"   Leverage: {leverage}x")
    print(f"   TP/SL: {tp_percent}%/{sl_percent}% price movement")
    print(f"   Leveraged returns: +{leveraged_win}%/-{leveraged_loss}% per trade")
    print(f"   Position size: {position_size_pct*100}% per trade")
    print(f"   Max concurrent trades: {max_concurrent_trades}")
    print()
    
    # === CALCULATE DAILY METRICS ===
    daily_trades = trades_executed / simulation_days
    daily_wins = daily_trades * (win_rate / 100)
    daily_losses = daily_trades * (1 - win_rate / 100)
    
    print("📈 CALCULATED DAILY METRICS:")
    print(f"   Average daily trades: {daily_trades:.2f}")
    print(f"   Average daily wins: {daily_wins:.2f}")
    print(f"   Average daily losses: {daily_losses:.2f}")
    print()
    
    # === PROJECTION SETUP ===
    start_date = datetime(2025, 7, 28)  # Start from today
    end_date = datetime(2025, 11, 27)   # Project to November 27th
    projection_days = (end_date - start_date).days
    
    # New starting parameters
    new_initial_balance = 1000
    
    print("🚀 PROJECTION PARAMETERS:")
    print(f"   New starting balance: ${new_initial_balance:,.2f}")
    print(f"   Projection start: {start_date.strftime('%B %d, %Y')}")
    print(f"   Projection end: {end_date.strftime('%B %d, %Y')}")
    print(f"   Projection period: {projection_days} days")
    print()
    
    # === DAILY PROJECTION ===
    print("📅 DAILY ACCOUNT BALANCE PROJECTION:")
    print("=" * 60)
    
    balance = new_initial_balance
    projection_data = []
    
    current_date = start_date
    
    # Show first 10 days, then weekly summaries, then final days
    day_counter = 0
    
    while current_date <= end_date:
        day_counter += 1
        
        # Simulate daily trading
        wins_today = np.random.poisson(daily_wins)
        losses_today = np.random.poisson(daily_losses)
        
        # Calculate daily P&L
        daily_pnl = 0
        
        # Process wins
        for _ in range(wins_today):
            trade_amount = balance * position_size_pct
            profit = trade_amount * (leveraged_win / 100)
            daily_pnl += profit
        
        # Process losses
        for _ in range(losses_today):
            trade_amount = balance * position_size_pct
            loss = trade_amount * (leveraged_loss / 100)
            daily_pnl -= loss
        
        # Update balance
        balance += daily_pnl
        
        # Store data
        projection_data.append({
            'date': current_date,
            'day': day_counter,
            'balance': balance,
            'daily_pnl': daily_pnl,
            'wins': wins_today,
            'losses': losses_today
        })
        
        # Display logic
        if day_counter <= 10:  # First 10 days
            print(f"Day {day_counter:3d} ({current_date.strftime('%b %d')}): ${balance:12,.2f} (${daily_pnl:+8,.2f}) | W:{wins_today} L:{losses_today}")
        elif day_counter % 7 == 0:  # Weekly updates
            week_num = day_counter // 7
            print(f"Week {week_num:2d} ({current_date.strftime('%b %d')}): ${balance:12,.2f} (${daily_pnl:+8,.2f}) | W:{wins_today} L:{losses_today}")
        elif projection_days - day_counter < 5:  # Final 5 days
            print(f"Day {day_counter:3d} ({current_date.strftime('%b %d')}): ${balance:12,.2f} (${daily_pnl:+8,.2f}) | W:{wins_today} L:{losses_today}")
        
        current_date += timedelta(days=1)
    
    # === FINAL RESULTS ===
    final_projected_balance = balance
    total_projected_return = ((final_projected_balance - new_initial_balance) / new_initial_balance) * 100
    
    print("\n" + "=" * 60)
    print("🎯 FINAL PROJECTION RESULTS:")
    print("=" * 60)
    print(f"Starting Balance:  ${new_initial_balance:15,.2f}")
    print(f"Projected Balance: ${final_projected_balance:15,.2f}")
    print(f"Total Return:      {total_projected_return:15.2f}%")
    print(f"Projection Period: {projection_days:15d} days")
    print(f"Average Daily ROI: {(total_projected_return/projection_days):15.3f}%")
    print()
    
    # === MILESTONES ===
    print("🏆 PROJECTED MILESTONES:")
    milestones = [5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
    
    for milestone in milestones:
        for data in projection_data:
            if data['balance'] >= milestone:
                days_to_milestone = data['day']
                milestone_date = data['date']
                print(f"   ${milestone:7,}: Day {days_to_milestone:3d} ({milestone_date.strftime('%B %d, %Y')})")
                break
    
    print()
    
    # === RISK ANALYSIS ===
    print("⚠️ RISK CONSIDERATIONS:")
    print("=" * 60)
    print("• This projection assumes consistent performance")
    print("• Real trading involves market volatility and drawdowns")
    print("• Results may vary significantly from projections")
    print("• Risk management is crucial for long-term success")
    print(f"• Max exposure per day: {max_concurrent_trades * position_size_pct * 100}% of account")
    print()
    
    # === CONSERVATIVE SCENARIO ===
    print("🛡️ CONSERVATIVE SCENARIO (50% of performance):")
    conservative_balance = new_initial_balance * (1 + (total_projected_return/100) * 0.5)
    conservative_return = ((conservative_balance - new_initial_balance) / new_initial_balance) * 100
    print(f"   Conservative Balance: ${conservative_balance:,.2f}")
    print(f"   Conservative Return:  {conservative_return:.2f}%")
    
    return projection_data

if __name__ == "__main__":
    projection_data = calculate_projected_growth()