#!/usr/bin/env python3
"""
Enhanced Trading Simulator - Standalone Version
Runs both SHORT and LONG trade simulators with all enhanced filtering logic
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from collections import defaultdict, deque

# === ENHANCED FILTERING FUNCTIONS ===

def passes_advanced_short_filters(row, ml_confidence, coin):
    """
    PERMISSIVE filtering for SHORT trades - Allow profitable trades to pass
    Returns True for potentially profitable SHORT setups
    """
    
    # TIER 1: HIGH CONFIDENCE (auto-pass - trust the model)
    if ml_confidence >= 0.85:
        return True  # Trust high-confidence predictions
    
    # TIER 2: MEDIUM-HIGH CONFIDENCE (very light filtering)
    if ml_confidence >= 0.75:
        # Just check for basic volume activity
        volume_ratio = getattr(row, 'volume_ratio', 1.0)
        if pd.isna(volume_ratio): volume_ratio = 1.0
        return volume_ratio >= 1.0  # Any volume is acceptable
    
    # TIER 3: MEDIUM CONFIDENCE (light filtering)
    if ml_confidence >= 0.70:
        # Allow most trades but check basic conditions
        filter_score = 0
        
        # Basic volume check
        volume_ratio = getattr(row, 'volume_ratio', 1.0)
        if pd.isna(volume_ratio): volume_ratio = 1.0
        if volume_ratio >= 1.1:  # Slight volume increase
            filter_score += 1
        
        # Basic momentum check
        rsi_14 = getattr(row, 'rsi_14', 50)
        if pd.isna(rsi_14): rsi_14 = 50
        if rsi_14 > 55 or rsi_14 < 45:  # Any momentum (not neutral)
            filter_score += 1
        
        return filter_score >= 1  # Very permissive - just need one condition
    
    # TIER 4: LOW CONFIDENCE (still allow some trades)
    if ml_confidence >= 0.65:
        # Even for lower confidence, allow some trades with basic checks
        volume_ratio = getattr(row, 'volume_ratio', 1.0)
        if pd.isna(volume_ratio): volume_ratio = 1.0
        rsi_14 = getattr(row, 'rsi_14', 50)
        if pd.isna(rsi_14): rsi_14 = 50
        
        # Allow if volume is reasonable and RSI shows some overbought condition
        return volume_ratio >= 1.0 and rsi_14 > 55
    
    # Only reject very low confidence trades
    return False


def passes_advanced_long_filters(row, ml_confidence, coin):
    """
    Advanced filtering system for LONG trades to maximize profitability
    Returns True if trade should proceed, False to reject
    """
    
    # Start with high-confidence trades to reduce filtering severity
    if ml_confidence >= 0.85:
        return True
    
    # For medium confidence trades, apply selective filtering
    filter_score = 0
    max_score = 4
    
    # FILTER 1: Volume Confirmation (optional for medium confidence)
    volume_ratio = getattr(row, 'volume_ratio', 1.0)
    if pd.isna(volume_ratio):
        volume_ratio = 1.0
    if volume_ratio >= 1.1:  # Slightly above average volume
        filter_score += 1
    
    # FILTER 2: RSI Oversold Check
    rsi_14 = getattr(row, 'rsi_14', 50)
    if pd.isna(rsi_14):
        rsi_14 = 50
    if rsi_14 < 40:  # Reasonably oversold
        filter_score += 1
    
    # FILTER 3: EMA Trend Check
    ema_9_21_ratio = getattr(row, 'ema_9_21_ratio', 1.0)
    if pd.isna(ema_9_21_ratio):
        ema_9_21_ratio = 1.0
    if ema_9_21_ratio > 1.0:  # Short EMA above long EMA (bullish)
        filter_score += 1
    
    # FILTER 4: Volatility Check
    atr_14 = getattr(row, 'atr_14', 0)
    if pd.isna(atr_14):
        atr_14 = 0.001  # Small default instead of 0
    if atr_14 > 0:  # Any volatility is good
        filter_score += 1
    
    # Allow trade if it meets at least 50% of criteria
    return filter_score >= (max_score * 0.5)


class Trade:
    def __init__(self, coin, entry_time, entry_price, direction, confidence, trade_id, leverage):
        self.coin = coin
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.direction = direction
        self.confidence = confidence
        self.trade_id = trade_id
        self.leverage = leverage

        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl_pct = 0.0
        self.pnl = 0.0
        self.duration_minutes = 0

    def close_trade(self, exit_time, exit_price, reason):
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = reason
        self.duration_minutes = (exit_time - self.entry_time).total_seconds() / 60

        if self.direction == 'long':
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price

        self.pnl = self.pnl_pct * self.leverage * 100


def run_short_simulator():
    """Run enhanced SHORT trade simulator"""
    print("🔻 RUNNING ENHANCED SHORT TRADE SIMULATOR")
    print("=" * 55)
    
    file_path = 'short_three_enhanced_predictions.csv'
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp')
    
    # BALANCED SHORT parameters (profitable but reasonable)
    initial_balance = 5000
    confidence_threshold = 0.80  # BALANCED: High but achievable threshold
    position_size_pct = 0.15  # BALANCED: Moderate position size (15%)
    stop_loss = 0.010  # BALANCED: 1.0% SL (tight but reasonable)
    take_profit = 0.015  # BALANCED: 1.5% TP (better risk/reward ratio)
    max_hold_hours = 12  # BALANCED: Medium hold time (12h)
    leverage = 8.0  # BALANCED: Good leverage (8x)
    direction = 'short'
    max_concurrent_trades = 2  # BALANCED: Allow 2 concurrent trades
    
    trades = []
    open_trades = []
    trade_id_counter = 0
    balance = initial_balance
    
    max_hold_minutes = max_hold_hours * 60
    total_filtered = 0
    
    print(f"📊 Processing {len(df)} predictions...")
    print(f"🎯 Enhanced confidence threshold: {confidence_threshold}")
    print(f"💰 Enhanced TP/SL: {take_profit*100}%/{stop_loss*100}%")
    print(f"🛡️ Risk limits: Max {max_concurrent_trades} concurrent trades, {position_size_pct*100}% per trade")
    print(f"⚖️ Total max exposure: {max_concurrent_trades * position_size_pct * 100}% of account")
    print()

    for i, row in df.iterrows():
        timestamp = row['timestamp']
        coin = row['coin']
        pred = row['prediction']
        conf = row['prediction_prob']
        open_price = row['open']
        high = row['high']
        low = row['low']
        close = row['close']

        # Exit logic for open trades
        for trade in open_trades[:]:
            if trade.coin != coin:
                continue

            duration = (timestamp - trade.entry_time).total_seconds() / 60
            sl_price = trade.entry_price * (1 + stop_loss)
            tp_price = trade.entry_price * (1 - take_profit)

            hit_tp = low <= tp_price
            hit_sl = high >= sl_price

            if hit_tp:
                trade.close_trade(timestamp, tp_price, 'take_profit')
            elif hit_sl:
                trade.close_trade(timestamp, sl_price, 'stop_loss')
            elif duration >= max_hold_minutes:
                trade.close_trade(timestamp, close, 'max_hold')
            else:
                continue

            trades.append(trade)
            open_trades.remove(trade)

            position_size = balance * position_size_pct
            balance += (trade.pnl / 100) * position_size

        # Entry logic with enhanced filtering
        if pred == 1 and conf >= confidence_threshold:
            
            # === ENHANCED PROFITABILITY FILTERING ===
            if not passes_advanced_short_filters(row, conf, coin):
                total_filtered += 1
                continue  # Skip this trade
            
            already_open = any(t.coin == coin for t in open_trades)
            
            # Check risk limits before opening new trade
            if not already_open and len(open_trades) < max_concurrent_trades:
                trade_id_counter += 1
                entry_price = open_price
                trade = Trade(coin, timestamp, entry_price, direction, conf, trade_id_counter, leverage)
                open_trades.append(trade)

    # Close remaining trades
    if not df.empty:
        last_time = df['timestamp'].iloc[-1]
        for trade in open_trades:
            last_close = df[df['coin'] == trade.coin]['close'].iloc[-1]
            trade.close_trade(last_time, last_close, 'end_of_data')
            trades.append(trade)

    # Calculate results
    total_trades = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl <= 0)
    win_pct = (wins / total_trades) * 100 if total_trades > 0 else 0
    
    original_potential = len(df[(df['prediction'] == 1) & (df['prediction_prob'] >= confidence_threshold)])
    filtering_pct = (total_filtered / original_potential) * 100 if original_potential > 0 else 0

    print(f"✅ SHORT SIMULATION COMPLETE")
    print(f"📊 Enhanced filtering effectiveness: {total_filtered}/{original_potential} trades filtered ({filtering_pct:.1f}%)")
    print(f"📊 Final results: {total_trades} trades, {wins} wins, {losses} losses")
    print(f"📊 Win rate: {win_pct:.2f}%")
    print(f"💰 Final balance: ${balance:,.2f} (Starting: ${initial_balance:,.2f})")
    print(f"📈 Total return: {((balance - initial_balance) / initial_balance) * 100:.2f}%")
    print()
    
    return {
        'direction': 'SHORT',
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_pct,
        'initial_balance': initial_balance,
        'final_balance': balance,
        'return_pct': ((balance - initial_balance) / initial_balance) * 100,
        'filtered_trades': total_filtered,
        'original_potential': original_potential,
        'filtering_pct': filtering_pct
    }


def run_long_simulator():
    """Run enhanced LONG trade simulator"""
    print("🔺 RUNNING ENHANCED LONG TRADE SIMULATOR")
    print("=" * 54)
    
    pred_path = 'three_enhanced_predictions.csv'
    base_path = 'baseline_ohlcv.csv'
    
    if not os.path.exists(pred_path) or not os.path.exists(base_path):
        print(f"❌ Missing required files: {pred_path} or {base_path}")
        return None

    predictions = pd.read_csv(pred_path, parse_dates=['timestamp'])
    baseline = pd.read_csv(base_path, parse_dates=['timestamp'])

    predictions.sort_values('timestamp', inplace=True)
    baseline.sort_values('timestamp', inplace=True)

    # Speed up baseline lookup
    baseline.set_index(['coin', 'timestamp'], inplace=True)

    # Enhanced parameters with risk limits
    initial_balance = 5000
    confidence_threshold = 0.75  # More moderate threshold
    position_size_pct = 0.25  # 25% of account balance per trade
    stop_loss = 0.015  # 1.5% SL (original working ratio)
    take_profit = 0.025  # 2.5% TP (original working ratio)
    max_hold_hours = 48
    leverage = 10.0
    max_concurrent_trades = 4  # Maximum 4 trades open at once
    
    trades = []
    open_trades = []
    balance = initial_balance
    confidence_history = defaultdict(lambda: deque(maxlen=3))
    trade_id_counter = 0

    max_hold_minutes = max_hold_hours * 60
    total_filtered = 0

    all_timestamps = sorted(predictions['timestamp'].unique())
    
    print(f"📊 Processing {len(predictions)} predictions...")
    print(f"🎯 Enhanced confidence threshold: {confidence_threshold}")
    print(f"💰 Enhanced TP/SL: {take_profit*100}%/{stop_loss*100}%")
    print(f"🛡️ Risk limits: Max {max_concurrent_trades} concurrent trades, {position_size_pct*100}% per trade")
    print(f"⚖️ Total max exposure: {max_concurrent_trades * position_size_pct * 100}% of account")
    print()

    for timestamp in all_timestamps:
        # Exit logic
        still_open = []
        for trade in open_trades:
            key = (trade.coin, timestamp)
            if key not in baseline.index:
                still_open.append(trade)
                continue

            row = baseline.loc[key]
            high = row['high']
            low = row['low']
            close = row['close']
            duration = (timestamp - trade.entry_time).total_seconds() / 60

            tp_price = trade.entry_price * (1 + take_profit)
            sl_price = trade.entry_price * (1 - stop_loss)

            if high >= tp_price:
                trade.close_trade(timestamp, tp_price, 'take_profit')
                trades.append(trade)
            elif low <= sl_price:
                trade.close_trade(timestamp, sl_price, 'stop_loss')
                trades.append(trade)
            elif duration >= max_hold_minutes:
                trade.close_trade(timestamp, close, 'max_hold')
                trades.append(trade)
            else:
                still_open.append(trade)

        open_trades = still_open

        # Entry logic with enhanced filtering
        current_rows = predictions[predictions['timestamp'] == timestamp]
        for _, row in current_rows.iterrows():
            coin = row['coin']
            conf = row['prediction_prob']
            pred = row['prediction']
            entry_price = row['open']

            confidence_history[coin].append(conf)
            if len(confidence_history[coin]) < 3:
                continue

            avg_conf = np.mean(confidence_history[coin])
            already_open = any(t.coin == coin for t in open_trades)

            if pred == 1 and conf >= confidence_threshold and not already_open:
                
                # === ENHANCED PROFITABILITY FILTERING ===
                if not passes_advanced_long_filters(row, conf, coin):
                    total_filtered += 1
                    continue  # Skip this trade

                # Check risk limits before opening new trade
                if len(open_trades) < max_concurrent_trades:
                    trade_id_counter += 1
                    trade = Trade(coin, timestamp, entry_price, 'long', conf, trade_id_counter, leverage)
                    open_trades.append(trade)

    # Close remaining trades
    for trade in open_trades:
        future_rows = baseline.loc[trade.coin]
        future_rows = future_rows[future_rows.index > trade.entry_time]

        if not future_rows.empty:
            last_time = future_rows.index[-1]
            last_price = future_rows.iloc[-1]['close']
            trade.close_trade(last_time, last_price, 'end_of_data')
            trades.append(trade)

    # Calculate results
    total_trades = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = total_trades - wins
    win_pct = (wins / total_trades) * 100 if total_trades > 0 else 0

    for t in trades:
        position_size = balance * position_size_pct
        pl = (t.pnl / 100) * position_size
        balance += pl
    
    original_potential = len(predictions[(predictions['prediction'] == 1) & (predictions['prediction_prob'] >= confidence_threshold)])
    filtering_pct = (total_filtered / original_potential) * 100 if original_potential > 0 else 0

    print(f"✅ LONG SIMULATION COMPLETE")
    print(f"📊 Enhanced filtering effectiveness: {total_filtered}/{original_potential} trades filtered ({filtering_pct:.1f}%)")
    print(f"📊 Final results: {total_trades} trades, {wins} wins, {losses} losses")
    print(f"📊 Win rate: {win_pct:.2f}%")
    print(f"💰 Final balance: ${balance:,.2f} (Starting: ${initial_balance:,.2f})")
    print(f"📈 Total return: {((balance - initial_balance) / initial_balance) * 100:.2f}%")
    print()
    
    return {
        'direction': 'LONG',
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_pct,
        'initial_balance': initial_balance,
        'final_balance': balance,
        'return_pct': ((balance - initial_balance) / initial_balance) * 100,
        'filtered_trades': total_filtered,
        'original_potential': original_potential,
        'filtering_pct': filtering_pct
    }


def main():
    """Run both enhanced simulators and compare results"""
    print("🚀 ENHANCED CRYPTOCURRENCY TRADING PLATFORM")
    print("💎 Running Complete Simulation with All Enhancements")
    print("=" * 65)
    print()
    
    # Run both simulators
    short_results = run_short_simulator()
    long_results = run_long_simulator()
    
    print("🎯 COMPREHENSIVE RESULTS COMPARISON")
    print("=" * 65)
    
    if short_results:
        print(f"🔻 SHORT TRADES:")
        print(f"   Trades: {short_results['total_trades']}")
        print(f"   Win Rate: {short_results['win_rate']:.2f}%")
        print(f"   Return: {short_results['return_pct']:.2f}%")
        print(f"   Final Balance: ${short_results['final_balance']:,.2f}")
        print(f"   Filtering: {short_results['filtering_pct']:.1f}% of potential trades filtered")
        print()
    
    if long_results:
        print(f"🔺 LONG TRADES:")
        print(f"   Trades: {long_results['total_trades']}")
        print(f"   Win Rate: {long_results['win_rate']:.2f}%")
        print(f"   Return: {long_results['return_pct']:.2f}%")
        print(f"   Final Balance: ${long_results['final_balance']:,.2f}")
        print(f"   Filtering: {long_results['filtering_pct']:.1f}% of potential trades filtered")
        print()
    
    if short_results and long_results:
        combined_return = short_results['return_pct'] + long_results['return_pct']
        combined_balance = short_results['final_balance'] + long_results['final_balance'] - 5000  # Subtract one initial balance
        print(f"💎 COMBINED PERFORMANCE:")
        print(f"   Total Return: {combined_return:.2f}%")
        print(f"   Combined Balance: ${combined_balance:,.2f}")
        print(f"   Platform Status: {'🟢 HIGHLY PROFITABLE' if combined_return > 100 else '🟡 PROFITABLE' if combined_return > 0 else '🔴 NEEDS OPTIMIZATION'}")


if __name__ == "__main__":
    main()