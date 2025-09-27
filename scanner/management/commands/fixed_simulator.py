# scanner/management/commands/fixed_simulator.py
# Master CSV simulator with correct entry logic

from django.core.management.base import BaseCommand
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class RegimeAwareTrade:
    def __init__(self, coin, entry_time, entry_price, leverage, position_size_usd, conf, trade_id, side, regime_score):
        self.coin = coin
        self.side = side.lower()
        self.entry_time = entry_time
        self.entry_price = float(entry_price)
        self.leverage = float(leverage)
        self.position_size_usd = float(position_size_usd)
        self.confidence = float(conf)
        self.trade_id = int(trade_id)
        self.regime_score = float(regime_score)

        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None
        self.gross_return = 0.0
        self.gross_pl_usd = 0.0
        self.fee_usd = 0.0
        self.net_pl_usd = 0.0

    def close(self, exit_time, exit_price, reason, entry_fee_bps, exit_fee_bps):
        self.exit_time = exit_time
        self.exit_price = float(exit_price)
        self.exit_reason = reason

        notional = self.position_size_usd * self.leverage
        if self.side == 'long':
            self.gross_return = (self.exit_price / self.entry_price - 1.0) * self.leverage
            self.gross_pl_usd = notional * (self.exit_price / self.entry_price - 1.0)
        else:  # short
            self.gross_return = (self.entry_price / self.exit_price - 1.0) * self.leverage
            self.gross_pl_usd = notional * (self.entry_price / self.exit_price - 1.0)

        fee_rate = (entry_fee_bps + exit_fee_bps) / 10000.0
        self.fee_usd = notional * fee_rate
        self.net_pl_usd = self.gross_pl_usd - self.fee_usd

class Command(BaseCommand):
    help = 'Master CSV simulator with correct entry logic'

    def add_arguments(self, parser):
        parser.add_argument('--master-file', type=str, default='master_simulation.csv')
        parser.add_argument('--sample-rate', type=int, default=1)
        parser.add_argument('--max-timestamps', type=int, default=999999)
        
        # Trading parameters
        parser.add_argument('--initial-balance', type=float, default=1000.0)
        parser.add_argument('--position-size', type=float, default=1.00)
        parser.add_argument('--leverage', type=float, default=15.0)
        parser.add_argument('--take-profit', type=float, default=0.02)
        parser.add_argument('--stop-loss', type=float, default=0.01)
        parser.add_argument('--max-hold-hours', type=int, default=100)
        parser.add_argument('--min-regime-strength', type=float, default=0.6)
        
        # Fees
        parser.add_argument('--entry-fee-bps', type=float, default=0.0)
        parser.add_argument('--exit-fee-bps', type=float, default=0.0)
        parser.add_argument('--max-concurrent-trades', type=int, default=1)

    def get_coins_from_master(self, master_df):
        """Extract coin list from master CSV columns"""
        coins = []
        for col in master_df.columns:
            if col.endswith('_open'):
                coin = col.replace('_open', '')
                coins.append(coin)
        return sorted(coins)

    def handle(self, *args, **opt):
        start_time = datetime.now()
        
        # Load configuration
        sample_rate = int(opt['sample_rate'])
        max_timestamps = int(opt['max_timestamps'])
        self.min_regime_strength = float(opt['min_regime_strength'])
        
        self.stdout.write("🚀 MASTER CSV SIMULATOR")
        self.stdout.write("📊 Using master CSV with correct entry logic")
        
        # Load master CSV
        master_path = opt['master_file']
        if not os.path.exists(master_path):
            self.stderr.write(f"❌ Master file not found: {master_path}")
            return
        
        self.stdout.write(f"▶ Loading master CSV...")
        master_df = pd.read_csv(master_path)
        master_df['timestamp'] = pd.to_datetime(master_df['timestamp'])
        
        # Sample data if needed
        if sample_rate > 1 or max_timestamps < len(master_df):
            sampled_indices = list(range(0, len(master_df), sample_rate))[:max_timestamps]
            master_df = master_df.iloc[sampled_indices].reset_index(drop=True)
        
        self.stdout.write(f"✅ Loaded {len(master_df)} timestamps from master CSV")
        self.stdout.write(f"📅 Date range: {master_df['timestamp'].min()} to {master_df['timestamp'].max()}")
        
        # Get coins from master CSV
        coins = self.get_coins_from_master(master_df)
        self.stdout.write(f"🪙 Coins: {coins}")
        
        # Trading parameters
        balance = float(opt['initial_balance'])
        position_frac = float(opt['position_size'])
        leverage = float(opt['leverage'])
        tp = float(opt['take_profit'])
        sl = float(opt['stop_loss'])
        max_hold_minutes = int(opt['max_hold_hours']) * 60
        entry_fee_bps = float(opt['entry_fee_bps'])
        exit_fee_bps = float(opt['exit_fee_bps'])
        max_concurrent = int(opt['max_concurrent_trades'])
        
        # Trading state
        open_trades = []
        closed_trades = []
        trade_id_counter = 0
        reserved_margin = 0.0
        equity_points = []
        
        self.stdout.write(f"▶ Starting simulation...")
        self.stdout.write(f"💰 Initial balance: ${balance:,.2f}")
        self.stdout.write("=" * 60)
        
        # Track regime changes
        last_regime = None
        regime_changes = 0
        
        for i, row in master_df.iterrows():
            if i % 100 == 0:
                elapsed = datetime.now() - start_time
                self.stdout.write(f"⏱️  {i+1}/{len(master_df)} ({((i+1)/len(master_df)*100):.1f}%) - {elapsed.total_seconds():.1f}s")
            
            ts = row['timestamp']
            
            # ===== REGIME LOOKUP =====
            current_regime = row.get('market_regime', 'neutral')
            current_bull_strength = row.get('bull_strength', 0.0)
            current_bear_strength = row.get('bear_strength', 0.0)
            
            # Log regime changes
            if last_regime != current_regime:
                regime_changes += 1
                self.stdout.write(f"🔄 REGIME: {last_regime} → {current_regime} at {ts}")
                last_regime = current_regime
            
            # ===== EXITS =====
            still_open = []
            for t in open_trades:
                if ts <= t.entry_time:
                    still_open.append(t)
                    continue
                
                # Get OHLCV data for this coin and timestamp
                high_col = f'{t.coin}_high'
                low_col = f'{t.coin}_low'
                close_col = f'{t.coin}_close'
                
                if high_col not in row or pd.isna(row[high_col]):
                    still_open.append(t)
                    continue
                
                high = float(row[high_col])
                low = float(row[low_col])
                close = float(row[close_col])
                
                # Check TP/SL
                if t.side == 'long':
                    tp_price = t.entry_price * (1.0 + tp)
                    sl_price = t.entry_price * (1.0 - sl)
                    hit_tp = (high >= tp_price)
                    hit_sl = (low <= sl_price)
                else:  # short
                    tp_price = t.entry_price * (1.0 - tp)
                    sl_price = t.entry_price * (1.0 + sl)
                    hit_tp = (low <= tp_price)
                    hit_sl = (high >= sl_price)
                
                def _close_and_update(reason, px):
                    nonlocal reserved_margin, balance
                    t.close(ts, px, reason, entry_fee_bps, exit_fee_bps)
                    closed_trades.append(t)
                    reserved_margin -= t.position_size_usd
                    balance += t.net_pl_usd
                    equity_points.append({'timestamp': ts, 'equity': balance})
                    
                    result = "✅ WIN" if t.net_pl_usd > 0 else "❌ LOSS"
                    self.stdout.write(f"🔚 CLOSED {t.side.upper()} {t.coin} | {result} | PnL: ${t.net_pl_usd:,.2f}")
                
                if hit_tp and hit_sl:
                    _close_and_update('both_hit_tp_first', tp_price)
                    continue
                
                if hit_tp:
                    _close_and_update('take_profit', tp_price)
                    continue
                
                if hit_sl:
                    _close_and_update('stop_loss', sl_price)
                    continue
                
                # Max hold
                dur_min = (ts - t.entry_time).total_seconds() / 60.0
                if dur_min >= max_hold_minutes:
                    _close_and_update('max_hold', close)
                else:
                    still_open.append(t)
            
            open_trades = still_open
            
            # ===== ENTRIES =====
            # Only one trade at a time
            if len(open_trades) == 0:
                available = balance - reserved_margin
                if available > 0:
                    position_size_usd = available * position_frac
                    
                    # Debug: Track why trades aren't opened for each coin
                    if i % 100 == 0:  # Only log every 100th timestamp to avoid spam
                        self.stdout.write(f"🔍 DEBUG: Checking trade opportunities at {ts}")
                        self.stdout.write(f"   Regime: {current_regime} (Bull: {current_bull_strength:.3f}, Bear: {current_bear_strength:.3f})")
                    
                    for coin in coins:
                        # Get confidence scores and thresholds for this coin
                        long_conf_col = f'{coin}_long_confidence'
                        short_conf_col = f'{coin}_short_confidence'
                        long_thresh_col = f'{coin}_long_threshold'
                        short_thresh_col = f'{coin}_short_threshold'
                        open_col = f'{coin}_open'
                        
                        if any(col not in row or pd.isna(row[col]) for col in [long_conf_col, short_conf_col, long_thresh_col, short_thresh_col, open_col]):
                            if i % 100 == 0:
                                self.stdout.write(f"   ⏭️ {coin}: Missing data")
                            continue
                        
                        entry_price = float(row[open_col])
                        long_confidence = float(row[long_conf_col])
                        short_confidence = float(row[short_conf_col])
                        long_threshold = float(row[long_thresh_col])
                        short_threshold = float(row[short_thresh_col])
                        
                        # Debug logging for specific coins
                        if i % 100 == 0 and coin in ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']:
                            self.stdout.write(f"   📊 {coin}: Long {long_confidence:.3f}/{long_threshold:.3f}, Short {short_confidence:.3f}/{short_threshold:.3f}")
                        
                        # Check long signals (only in bullish regime)
                        if current_regime == 'bull' and current_bull_strength >= 0.3 and long_confidence >= long_threshold:
                            trade_id_counter += 1
                            t = RegimeAwareTrade(
                                coin=coin, entry_time=ts, entry_price=entry_price,
                                leverage=leverage, position_size_usd=position_size_usd,
                                conf=long_confidence, trade_id=trade_id_counter,
                                side='long', regime_score=current_bull_strength
                            )
                            open_trades.append(t)
                            reserved_margin += position_size_usd
                            
                            self.stdout.write(f"🟢 OPENED LONG {coin} | ${entry_price:.6f} | Conf: {long_confidence:.3f} | Regime: {current_regime} ({current_bull_strength:.3f})")
                            break
                        
                        # Check short signals (only in bearish regime)
                        elif current_regime == 'bear' and current_bear_strength >= 0.3 and short_confidence >= short_threshold:
                            trade_id_counter += 1
                            t = RegimeAwareTrade(
                                coin=coin, entry_time=ts, entry_price=entry_price,
                                leverage=leverage, position_size_usd=position_size_usd,
                                conf=short_confidence, trade_id=trade_id_counter,
                                side='short', regime_score=current_bear_strength
                            )
                            open_trades.append(t)
                            reserved_margin += position_size_usd
                            
                            self.stdout.write(f"🔴 OPENED SHORT {coin} | ${entry_price:.6f} | Conf: {short_confidence:.3f} | Regime: {current_regime} ({current_bear_strength:.3f})")
                            break
        
        # Force close remaining trades
        if open_trades:
            last_row = master_df.iloc[-1]
            last_ts = last_row['timestamp']
            for t in open_trades:
                close_col = f'{t.coin}_close'
                if close_col in last_row and not pd.isna(last_row[close_col]):
                    last_close = float(last_row[close_col])
                    t.close(last_ts, last_close, 'end_of_data', entry_fee_bps, exit_fee_bps)
                    closed_trades.append(t)
                    reserved_margin -= t.position_size_usd
                    balance += t.net_pl_usd
                    equity_points.append({'timestamp': last_ts, 'equity': balance})
                    self.stdout.write(f"🔚 FORCE CLOSED {t.side.upper()} {t.coin} | PnL: ${t.net_pl_usd:,.2f}")
        
        # Final results
        total_time = datetime.now() - start_time
        total = len(closed_trades)
        wins = sum(1 for t in closed_trades if t.net_pl_usd > 0)
        losses = total - wins
        win_pct = (wins / total * 100.0) if total > 0 else 0.0
        total_net = sum(t.net_pl_usd for t in closed_trades)
        
        bull_trades = [t for t in closed_trades if t.side == 'long']
        bear_trades = [t for t in closed_trades if t.side == 'short']
        bull_wins = sum(1 for t in bull_trades if t.net_pl_usd > 0)
        bear_wins = sum(1 for t in bear_trades if t.net_pl_usd > 0)
        
        self.stdout.write("=" * 60)
        self.stdout.write(self.style.SUCCESS(f"✅ SIMULATION COMPLETE in {total_time.total_seconds():.1f} seconds"))
        self.stdout.write(self.style.SUCCESS(f"📊 Total trades: {total}, Wins: {wins}, Losses: {losses}, Win %: {win_pct:.2f}%"))
        self.stdout.write(self.style.SUCCESS(f"🐂 Bull trades: {len(bull_trades)}, Wins: {bull_wins}, Win %: {(bull_wins/len(bull_trades)*100) if bull_trades else 0:.2f}%"))
        self.stdout.write(self.style.SUCCESS(f"🐻 Bear trades: {len(bear_trades)}, Wins: {bear_wins}, Win %: {(bear_wins/len(bear_trades)*100) if bear_trades else 0:.2f}%"))
        self.stdout.write(self.style.SUCCESS(f"🔄 Regime changes: {regime_changes}"))
        self.stdout.write(self.style.SUCCESS(f"💰 End Balance: ${balance:,.2f} (Net: ${total_net:,.2f})"))
