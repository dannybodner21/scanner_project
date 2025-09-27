# scanner/management/commands/fast_regime_simulator_v2.py
# Fast Trade Simulator with detailed logging and progress tracking

from django.core.management.base import BaseCommand
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from datetime import timezone
import json

class RegimeAwareTrade:
    """
    Trade class for regime-aware trading
    """
    def __init__(self, coin, entry_time, entry_price, leverage, position_size_usd, conf, trade_id, side, regime_score):
        self.coin = coin
        self.side = side.lower()  # 'long' or 'short'
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
    help = 'Fast Regime-Aware Trade Simulator with detailed logging'

    def add_arguments(self, parser):
        # Input files
        parser.add_argument('--baseline-file', type=str, default='baseline_ohlcv.csv')
        parser.add_argument('--regime-file', type=str, default='market_regime.csv')
        
        # Trading parameters
        parser.add_argument('--initial-balance', type=float, default=5000.0)
        parser.add_argument('--position-size', type=float, default=1.00)
        parser.add_argument('--leverage', type=float, default=15.0)
        parser.add_argument('--take-profit', type=float, default=0.02)
        parser.add_argument('--stop-loss', type=float, default=0.01)
        parser.add_argument('--max-hold-hours', type=int, default=100)
        
        # Regime parameters
        parser.add_argument('--min-regime-strength', type=float, default=0.6)
        
        # Performance
        parser.add_argument('--progress-interval', type=int, default=1000, help='Print progress every N timestamps')
        
        # Fees and other
        parser.add_argument('--entry-fee-bps', type=float, default=0.0)
        parser.add_argument('--exit-fee-bps', type=float, default=0.0)
        parser.add_argument('--max-concurrent-trades', type=int, default=1)
        parser.add_argument('--output-dir', type=str, default='.')

    def load_model_predictions_fast(self):
        """Load model predictions with minimal processing"""
        predictions = {}
        
        # Model files and thresholds (hardcoded for speed)
        models = {
            # Long models
            'ADAUSDT_long': ('ada_two_predictions.csv', 0.55),
            'ATOMUSDT_long': ('atom_two_predictions.csv', 0.5),
            'AVAXUSDT_long': ('avax_two_predictions.csv', 0.5),
            'BTCUSDT_long': ('btc_two_predictions.csv', 0.38),
            'DOGEUSDT_long': ('doge_two_predictions.csv', 0.5),
            'DOTUSDT_long': ('dot_two_predictions.csv', 0.55),
            'ETHUSDT_long': ('eth_two_predictions.csv', 0.4),
            'LINKUSDT_long': ('link_two_predictions.csv', 0.45),
            'LTCUSDT_long': ('ltc_two_predictions.csv', 0.55),
            'SOLUSDT_long': ('sol_two_predictions.csv', 0.5),
            'UNIUSDT_long': ('uni_two_predictions.csv', 0.55),
            'XLMUSDT_long': ('xlm_two_predictions.csv', 0.5),
            'XRPUSDT_long': ('xrp_two_predictions.csv', 0.55),
            'SHIBUSDT_long': ('shib_two_predictions.csv', 0.55),
            'TRXUSDT_long': ('trx_two_predictions.csv', 0.1),
            
            # Short models
            'ADAUSDT_short': ('ada_simple_short_predictions.csv', 0.55),
            'ATOMUSDT_short': ('atom_simple_short_predictions.csv', 0.5),
            'AVAXUSDT_short': ('avax_simple_short_predictions.csv', 0.5),
            'DOGEUSDT_short': ('doge_simple_short_predictions.csv', 0.5),
            'DOTUSDT_short': ('dot_simple_short_predictions.csv', 0.55),
            'ETHUSDT_short': ('eth_simple_short_predictions.csv', 0.4),
            'LINKUSDT_short': ('link_simple_short_predictions.csv', 0.45),
            'LTCUSDT_short': ('ltc_simple_short_predictions.csv', 0.55),
            'SOLUSDT_short': ('sol_simple_short_predictions.csv', 0.5),
            'UNIUSDT_short': ('uni_simple_short_predictions.csv', 0.55),
            'XRPUSDT_short': ('xrp_simple_short_predictions.csv', 0.55),
            'SHIBUSDT_short': ('shib_simple_short_predictions.csv', 0.55),
        }
        
        for model_key, (file_path, threshold) in models.items():
            if os.path.exists(file_path):
                try:
                    # Load only necessary columns for speed
                    df = pd.read_csv(file_path, usecols=['timestamp', 'pred_prob'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
                    df = df.set_index('timestamp')
                    predictions[model_key] = {
                        'data': df,
                        'threshold': threshold
                    }
                    self.stdout.write(f"✅ Loaded {model_key}")
                except Exception as e:
                    self.stdout.write(f"❌ Failed to load {model_key}: {e}")
        
        return predictions

    def handle(self, *args, **opt):
        start_time = datetime.now()
        
        # Load configuration
        self.min_regime_strength = float(opt['min_regime_strength'])
        progress_interval = int(opt['progress_interval'])
        
        self.stdout.write("🚀 FAST REGIME-AWARE TRADE SIMULATOR V2")
        self.stdout.write("📊 Using pre-calculated market regime data with detailed logging")
        
        # Load baseline OHLCV data
        baseline_path = opt['baseline_file']
        if not os.path.exists(baseline_path):
            self.stderr.write(f"❌ Baseline file not found: {baseline_path}")
            return
        
        self.stdout.write(f"▶ Loading baseline data from {baseline_path}...")
        baseline = pd.read_csv(baseline_path)
        baseline['timestamp'] = pd.to_datetime(baseline['timestamp'], utc=True).dt.tz_localize(None)
        baseline = baseline.sort_values(['coin', 'timestamp']).reset_index(drop=True)
        self.stdout.write(f"✅ Loaded {len(baseline)} OHLCV records")
        
        # Load pre-calculated regime data
        regime_path = opt['regime_file']
        if not os.path.exists(regime_path):
            self.stderr.write(f"❌ Regime file not found: {regime_path}")
            return
        
        self.stdout.write(f"▶ Loading regime data from {regime_path}...")
        regime_df = pd.read_csv(regime_path)
        regime_df['timestamp'] = pd.to_datetime(regime_df['timestamp'], utc=True).dt.tz_localize(None)
        regime_df = regime_df.set_index('timestamp')
        self.stdout.write(f"✅ Loaded {len(regime_df)} regime records")
        
        # Load model predictions
        self.stdout.write("▶ Loading model predictions...")
        predictions = self.load_model_predictions_fast()
        self.stdout.write(f"✅ Loaded {len(predictions)} model predictions")
        
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
        
        # Get unique timestamps and create lookup structures for speed
        market_ts = baseline['timestamp'].unique()
        market_ts = sorted(market_ts)
        
        # Create fast lookup structures
        baseline_lookup = baseline.set_index(['coin', 'timestamp'])
        
        self.stdout.write(f"▶ Starting simulation of {len(market_ts)} timestamps...")
        self.stdout.write(f"💰 Initial balance: ${balance:,.2f}")
        self.stdout.write(f"📊 Position size: {position_frac*100:.0f}% of balance")
        self.stdout.write(f"⚡ Leverage: {leverage}x")
        self.stdout.write(f"🎯 TP: {tp*100:.1f}%, SL: {sl*100:.1f}%")
        self.stdout.write("=" * 80)
        
        # Track regime changes
        last_regime = None
        regime_changes = 0
        
        for i, ts in enumerate(market_ts):
            if i % progress_interval == 0:
                elapsed = datetime.now() - start_time
                self.stdout.write(f"⏱️  Progress: {i+1}/{len(market_ts)} ({((i+1)/len(market_ts)*100):.1f}%) - {elapsed.total_seconds():.1f}s elapsed")
            
            # ===== REGIME LOOKUP (FAST!) =====
            try:
                regime_row = regime_df.loc[ts]
                current_regime = regime_row['regime']
                current_bull_strength = regime_row['bull_strength']
                current_bear_strength = regime_row['bear_strength']
                
                # Log regime changes
                if last_regime != current_regime:
                    regime_changes += 1
                    self.stdout.write(f"🔄 REGIME CHANGE: {last_regime} → {current_regime} at {ts}")
                    self.stdout.write(f"   Bull: {current_bull_strength:.3f}, Bear: {current_bear_strength:.3f}")
                    last_regime = current_regime
                    
            except KeyError:
                current_regime = 'neutral'
                current_bull_strength = 0.0
                current_bear_strength = 0.0
            
            # ===== EXITS =====
            still_open = []
            for t in open_trades:
                if ts < t.entry_time:
                    still_open.append(t)
                    continue
                
                # Get OHLCV data for this coin and timestamp
                try:
                    row = baseline_lookup.loc[(t.coin, ts)]
                    high = float(row['high'])
                    low = float(row['low'])
                    close = float(row['close'])
                except KeyError:
                    still_open.append(t)
                    continue
                
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
                    
                    # Log trade closure
                    result = "✅ WIN" if t.net_pl_usd > 0 else "❌ LOSS"
                    self.stdout.write(f"🔚 CLOSED {t.side.upper()} {t.coin} | {result} | PnL: ${t.net_pl_usd:,.2f} | {reason}")
                
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
            if len(open_trades) < max_concurrent:
                available = balance - reserved_margin
                if available > 0:
                    position_size_usd = available * position_frac
                    
                    # Check each coin for entry signals
                    for coin in baseline['coin'].unique():
                        if any(ot.coin == coin for ot in open_trades):
                            continue  # Already have position in this coin
                        
                        # Get OHLCV data
                        try:
                            row = baseline_lookup.loc[(coin, ts)]
                            entry_price = float(row['open'])
                        except KeyError:
                            continue
                        
                        # Check long signals (only in bull regime)
                        if current_regime == 'bull' and current_bull_strength >= self.min_regime_strength:
                            long_key = f'{coin}_long'
                            if long_key in predictions:
                                model_data = predictions[long_key]
                                df = model_data['data']
                                threshold = model_data['threshold']
                                
                                # Find prediction for this timestamp
                                try:
                                    confidence = df.loc[ts, 'pred_prob']
                                    if confidence >= threshold:
                                        trade_id_counter += 1
                                        t = RegimeAwareTrade(
                                            coin=coin, entry_time=ts, entry_price=entry_price,
                                            leverage=leverage, position_size_usd=position_size_usd,
                                            conf=confidence, trade_id=trade_id_counter,
                                            side='long', regime_score=current_bull_strength
                                        )
                                        open_trades.append(t)
                                        reserved_margin += position_size_usd
                                        
                                        # Log trade opening
                                        self.stdout.write(f"🟢 OPENED LONG {coin} | Price: ${entry_price:.6f} | Conf: {confidence:.3f} | Regime: {current_bull_strength:.3f}")
                                        break  # Only one trade per timestamp
                                except KeyError:
                                    pass
                        
                        # Check short signals (only in bear regime)
                        elif current_regime == 'bear' and current_bear_strength >= self.min_regime_strength:
                            short_key = f'{coin}_short'
                            if short_key in predictions:
                                model_data = predictions[short_key]
                                df = model_data['data']
                                threshold = model_data['threshold']
                                
                                # Find prediction for this timestamp
                                try:
                                    confidence = df.loc[ts, 'pred_prob']
                                    if confidence >= threshold:
                                        trade_id_counter += 1
                                        t = RegimeAwareTrade(
                                            coin=coin, entry_time=ts, entry_price=entry_price,
                                            leverage=leverage, position_size_usd=position_size_usd,
                                            conf=confidence, trade_id=trade_id_counter,
                                            side='short', regime_score=current_bear_strength
                                        )
                                        open_trades.append(t)
                                        reserved_margin += position_size_usd
                                        
                                        # Log trade opening
                                        self.stdout.write(f"🔴 OPENED SHORT {coin} | Price: ${entry_price:.6f} | Conf: {confidence:.3f} | Regime: {current_bear_strength:.3f}")
                                        break  # Only one trade per timestamp
                                except KeyError:
                                    pass
        
        # Force close remaining trades
        if open_trades:
            last_ts = market_ts[-1]
            for t in open_trades:
                try:
                    row = baseline_lookup.loc[(t.coin, last_ts)]
                    last_close = float(row['close'])
                    t.close(last_ts, last_close, 'end_of_data', entry_fee_bps, exit_fee_bps)
                    closed_trades.append(t)
                    reserved_margin -= t.position_size_usd
                    balance += t.net_pl_usd
                    equity_points.append({'timestamp': last_ts, 'equity': balance})
                    self.stdout.write(f"🔚 FORCE CLOSED {t.side.upper()} {t.coin} | PnL: ${t.net_pl_usd:,.2f}")
                except KeyError:
                    pass
        
        # Sort trades by exit time
        closed_trades.sort(key=lambda x: x.exit_time or datetime.min)
        
        # Save results
        os.makedirs(opt['output_dir'], exist_ok=True)
        out_csv = os.path.join(opt['output_dir'], 'fast_regime_trading_results_v2.csv')
        
        out_df = pd.DataFrame([{
            'trade_id': t.trade_id,
            'side': t.side,
            'coin': t.coin,
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'confidence': t.confidence,
            'regime_score': t.regime_score,
            'position_size_usd': t.position_size_usd,
            'leverage': t.leverage,
            'exit_reason': t.exit_reason,
            'gross_return': t.gross_return,
            'gross_pl_usd': t.gross_pl_usd,
            'fee_usd': t.fee_usd,
            'net_pl_usd': t.net_pl_usd
        } for t in closed_trades])
        out_df.to_csv(out_csv, index=False)
        
        # Equity curve
        eq_csv = os.path.join(opt['output_dir'], 'fast_regime_equity_curve_v2.csv')
        if equity_points:
            pd.DataFrame(equity_points).sort_values('timestamp').to_csv(eq_csv, index=False)
        
        # Print final results
        total_time = datetime.now() - start_time
        total = len(closed_trades)
        wins = sum(1 for t in closed_trades if t.net_pl_usd > 0)
        losses = total - wins
        win_pct = (wins / total * 100.0) if total > 0 else 0.0
        total_net = sum(t.net_pl_usd for t in closed_trades)
        
        # Regime statistics
        bull_trades = [t for t in closed_trades if t.side == 'long']
        bear_trades = [t for t in closed_trades if t.side == 'short']
        bull_wins = sum(1 for t in bull_trades if t.net_pl_usd > 0)
        bear_wins = sum(1 for t in bear_trades if t.net_pl_usd > 0)
        
        self.stdout.write("=" * 80)
        self.stdout.write(self.style.SUCCESS(f"✅ SIMULATION COMPLETE in {total_time.total_seconds():.1f} seconds"))
        self.stdout.write(self.style.SUCCESS(f"📊 Total trades: {total}, Wins: {wins}, Losses: {losses}, Win %: {win_pct:.2f}%"))
        self.stdout.write(self.style.SUCCESS(f"🐂 Bull trades: {len(bull_trades)}, Wins: {bull_wins}, Win %: {(bull_wins/len(bull_trades)*100) if bull_trades else 0:.2f}%"))
        self.stdout.write(self.style.SUCCESS(f"🐻 Bear trades: {len(bear_trades)}, Wins: {bear_wins}, Win %: {(bear_wins/len(bear_trades)*100) if bear_trades else 0:.2f}%"))
        self.stdout.write(self.style.SUCCESS(f"🔄 Regime changes: {regime_changes}"))
        self.stdout.write(self.style.SUCCESS(f"💰 End Balance: ${balance:,.2f} (Net: ${total_net:,.2f} from start ${opt['initial_balance']:,.2f})"))
        self.stdout.write(self.style.SUCCESS(f"📈 Equity curve: {eq_csv}"))
        self.stdout.write(self.style.SUCCESS(f"📊 Results: {out_csv}"))
