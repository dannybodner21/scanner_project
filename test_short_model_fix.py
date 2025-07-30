#!/usr/bin/env python3
"""
Simple test for the improved SHORT model
Tests only the SHORT simulator to validate profitability improvements
"""

import sys
import os

# Add the current directory to Python path to avoid import issues
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from collections import defaultdict, deque

    # Import the enhanced filtering function
    from run_enhanced_simulators import passes_advanced_short_filters, Trade

    def test_short_model_improvements():
        """Test the improved SHORT model with balanced parameters"""
        
        print("🔻 TESTING IMPROVED SHORT MODEL")
        print("=" * 50)
        
        file_path = 'short_three_enhanced_predictions.csv'
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None
        
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df = df.sort_values('timestamp')
        
        # Ultra-selective HIGH confidence approach
        initial_balance = 5000
        confidence_threshold = 0.88  # Very high threshold - only best trades
        position_size_pct = 0.20     # Larger position for high-confidence
        stop_loss = 0.010           # Reasonable SL (1.0%)
        take_profit = 0.020         # Good TP (2.0%) - 2:1 ratio
        max_hold_hours = 8          # Allow more time for development
        leverage = 10.0             # Standard leverage
        max_concurrent_trades = 2   # Limit to 2 best trades
        
        trades = []
        open_trades = []
        trade_id_counter = 0
        balance = initial_balance
        
        max_hold_minutes = max_hold_hours * 60
        total_filtered = 0
        
        print(f"📊 Testing {len(df)} predictions...")
        print(f"🎯 Confidence threshold: {confidence_threshold}")
        print(f"💰 TP/SL: {take_profit*100}%/{stop_loss*100}%")
        print(f"🛡️ Risk: Max {max_concurrent_trades} trades, {position_size_pct*100}% per trade")
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
                
                # Apply balanced filtering
                if not passes_advanced_short_filters(row, conf, coin):
                    total_filtered += 1
                    continue
                
                already_open = any(t.coin == coin for t in open_trades)
                
                if not already_open and len(open_trades) < max_concurrent_trades:
                    trade_id_counter += 1
                    entry_price = open_price
                    trade = Trade(coin, timestamp, entry_price, 'short', conf, trade_id_counter, leverage)
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

        print("✅ IMPROVED SHORT MODEL RESULTS:")
        print("=" * 50)
        print(f"📊 Total trades: {total_trades}")
        print(f"🎯 Wins: {wins} | Losses: {losses}")
        print(f"📈 Win rate: {win_pct:.2f}%")
        print(f"💰 Starting balance: ${initial_balance:,.2f}")
        print(f"💰 Final balance: ${balance:,.2f}")
        print(f"📊 Total return: {((balance - initial_balance) / initial_balance) * 100:.2f}%")
        print(f"🔍 Filtering: {total_filtered}/{original_potential} ({filtering_pct:.1f}%) filtered")
        print()
        
        # Performance assessment
        if balance > initial_balance:
            print("🟢 SHORT MODEL STATUS: PROFITABLE!")
            profit_multiple = balance / initial_balance
            print(f"📈 Profit multiple: {profit_multiple:.2f}x")
        else:
            print("🔴 SHORT MODEL STATUS: Still losing money")
            loss_pct = ((initial_balance - balance) / initial_balance) * 100
            print(f"📉 Loss percentage: -{loss_pct:.2f}%")
        
        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_pct,
            'initial_balance': initial_balance,
            'final_balance': balance,
            'return_pct': ((balance - initial_balance) / initial_balance) * 100,
            'is_profitable': balance > initial_balance
        }

    if __name__ == "__main__":
        result = test_short_model_improvements()
        
        if result and result['is_profitable']:
            print("🎉 SUCCESS: SHORT model is now PROFITABLE!")
        else:
            print("⚠️ Still needs work: SHORT model not yet profitable")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required packages: pip install pandas numpy")
except Exception as e:
    print(f"❌ Error: {e}")