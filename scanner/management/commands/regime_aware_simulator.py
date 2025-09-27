# scanner/management/commands/regime_aware_simulator.py
# Market Regime-Aware Trade Simulator
# 
# Strategy:
# 1. Use ALL long models to detect bull market regime
# 2. Use ALL short models to detect bear market regime  
# 3. Only take long trades during bull regimes
# 4. Only take short trades during bear regimes
# 5. Weight models by their historical performance
# 6. Average confidence scores to determine market regime

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
        self.regime_score = float(regime_score)  # Market regime score when trade was opened

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
    help = 'Regime-Aware Trade Simulator - Uses market regime detection to determine long vs short trades'

    def add_arguments(self, parser):
        # Model files (hardcoded for all 15 coins)
        parser.add_argument('--baseline-file', type=str, default='regime_baseline_ohlcv.csv')
        
        # Trading parameters
        parser.add_argument('--initial-balance', type=float, default=5000.0)
        parser.add_argument('--position-size', type=float, default=1.00, help='Fraction of balance per trade')
        parser.add_argument('--leverage', type=float, default=15.0)
        parser.add_argument('--take-profit', type=float, default=0.02, help='2% TP')
        parser.add_argument('--stop-loss', type=float, default=0.01, help='1% SL')
        parser.add_argument('--max-hold-hours', type=int, default=100, help='Max hold time')
        
        # Regime detection parameters
        parser.add_argument('--regime-lookback', type=int, default=12, help='Bars to look back for regime detection')
        parser.add_argument('--regime-threshold', type=float, default=0.1, help='Min difference to switch regimes')
        parser.add_argument('--min-regime-strength', type=float, default=0.6, help='Min regime strength to trade')
        parser.add_argument('--regime-update-frequency', type=int, default=12, help='Update regime every N timestamps (for speed)')
        
        # Fees and other
        parser.add_argument('--entry-fee-bps', type=float, default=0.0)
        parser.add_argument('--exit-fee-bps', type=float, default=0.0)
        parser.add_argument('--max-concurrent-trades', type=int, default=1)
        parser.add_argument('--output-dir', type=str, default='.')

    def load_model_predictions(self):
        """Load all model prediction files"""
        predictions = {}
        
        # Long model files (hardcoded)
        long_models = {
            'ADAUSDT': 'ada_two_predictions.csv',
            'ATOMUSDT': 'atom_two_predictions.csv',
            'AVAXUSDT': 'avax_two_predictions.csv',
            'BTCUSDT': 'btc_two_predictions.csv',
            'DOGEUSDT': 'doge_two_predictions.csv',
            'DOTUSDT': 'dot_two_predictions.csv',
            'ETHUSDT': 'eth_two_predictions.csv',
            'LINKUSDT': 'link_two_predictions.csv',
            'LTCUSDT': 'ltc_two_predictions.csv',
            'SOLUSDT': 'sol_two_predictions.csv',
            'UNIUSDT': 'uni_two_predictions.csv',
            'XLMUSDT': 'xlm_two_predictions.csv',
            'XRPUSDT': 'xrp_two_predictions.csv',
            'SHIBUSDT': 'shib_two_predictions.csv',
            'TRXUSDT': 'trx_two_predictions.csv'
        }
        
        # Short model files (hardcoded)
        short_models = {
            'ADAUSDT': 'ada_simple_short_predictions.csv',
            'ATOMUSDT': 'atom_simple_short_predictions.csv',
            'AVAXUSDT': 'avax_simple_short_predictions.csv',
            'DOGEUSDT': 'doge_simple_short_predictions.csv',
            'DOTUSDT': 'dot_simple_short_predictions.csv',
            'ETHUSDT': 'eth_simple_short_predictions.csv',
            'LINKUSDT': 'link_simple_short_predictions.csv',
            'LTCUSDT': 'ltc_simple_short_predictions.csv',
            'SOLUSDT': 'sol_simple_short_predictions.csv',
            'UNIUSDT': 'uni_simple_short_predictions.csv',
            'XRPUSDT': 'xrp_simple_short_predictions.csv',
            'SHIBUSDT': 'shib_simple_short_predictions.csv'
        }
        
        # Model optimal thresholds (for normalizing confidence scores)
        long_thresholds = {
            'ADAUSDT': 0.55, 'ATOMUSDT': 0.5, 'AVAXUSDT': 0.5, 'BTCUSDT': 0.38,
            'DOGEUSDT': 0.5, 'DOTUSDT': 0.55, 'ETHUSDT': 0.4, 'LINKUSDT': 0.45,
            'LTCUSDT': 0.55, 'SOLUSDT': 0.5, 'UNIUSDT': 0.55, 'XLMUSDT': 0.5,
            'XRPUSDT': 0.55, 'SHIBUSDT': 0.55, 'TRXUSDT': 0.1
        }
        
        short_thresholds = {
            'ADAUSDT': 0.55, 'ATOMUSDT': 0.5, 'AVAXUSDT': 0.6, 'DOGEUSDT': 0.6,
            'DOTUSDT': 0.55, 'ETHUSDT': 0.55, 'LINKUSDT': 0.6, 'LTCUSDT': 0.6,
            'SOLUSDT': 0.65, 'UNIUSDT': 0.5, 'XRPUSDT': 0.55, 'SHIBUSDT': 0.55
        }
        
        # Load long predictions
        for coin, file_path in long_models.items():
            if os.path.exists(file_path) and long_thresholds[coin] > 0:
                try:
                    df = pd.read_csv(file_path)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
                    predictions[f'{coin}_long'] = {
                        'data': df,
                        'threshold': long_thresholds[coin],
                        'optimal_threshold': long_thresholds[coin]
                    }
                    self.stdout.write(f"✅ Loaded long model for {coin} (threshold: {long_thresholds[coin]})")
                except Exception as e:
                    self.stdout.write(f"❌ Failed to load long model for {coin}: {e}")
        
        # Load short predictions
        for coin, file_path in short_models.items():
            if os.path.exists(file_path) and short_thresholds[coin] > 0:
                try:
                    df = pd.read_csv(file_path)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
                    predictions[f'{coin}_short'] = {
                        'data': df,
                        'threshold': short_thresholds[coin],
                        'optimal_threshold': short_thresholds[coin]
                    }
                    self.stdout.write(f"✅ Loaded short model for {coin} (threshold: {short_thresholds[coin]})")
                except Exception as e:
                    self.stdout.write(f"❌ Failed to load short model for {coin}: {e}")
        
        return predictions

    def detect_market_regime(self, predictions, current_time, lookback_bars=12):
        """
        Detect market regime using normalized confidence scores
        Normalizes each model's confidence score by its optimal threshold
        Returns: (regime, bull_strength, bear_strength)
        - regime: 'bull', 'bear', or 'neutral'
        - bull_strength: normalized average of long model scores
        - bear_strength: normalized average of short model scores
        """
        bull_scores = []
        bear_scores = []
        
        # Get recent timestamps
        recent_times = []
        for key, model_data in predictions.items():
            df = model_data['data']
            if not df.empty:
                recent_times.extend(df['timestamp'].tail(lookback_bars).tolist())
        
        if not recent_times:
            return 'neutral', 0.0, 0.0
        
        # Get the most recent timestamps
        recent_times = sorted(set(recent_times))[-lookback_bars:]
        
        for timestamp in recent_times:
            timestamp_bull_scores = []
            timestamp_bear_scores = []
            
            # Collect normalized scores for this timestamp
            for key, model_data in predictions.items():
                df = model_data['data']
                optimal_threshold = model_data['optimal_threshold']
                
                # Find matching timestamp
                mask = df['timestamp'] == timestamp
                if mask.any():
                    row = df[mask].iloc[0]
                    
                    if key.endswith('_long'):
                        if 'pred_prob' in row:
                            raw_score = row['pred_prob']
                            # Normalize: score / optimal_threshold
                            # This makes all models comparable regardless of their threshold
                            normalized_score = raw_score / optimal_threshold
                            timestamp_bull_scores.append(normalized_score)
                    elif key.endswith('_short'):
                        if 'pred_prob' in row:
                            raw_score = row['pred_prob']
                            # Normalize: score / optimal_threshold
                            normalized_score = raw_score / optimal_threshold
                            timestamp_bear_scores.append(normalized_score)
            
            # Average normalized scores for this timestamp
            if timestamp_bull_scores:
                bull_avg = np.mean(timestamp_bull_scores)
                bull_scores.append(bull_avg)
            
            if timestamp_bear_scores:
                bear_avg = np.mean(timestamp_bear_scores)
                bear_scores.append(bear_avg)
        
        # Calculate final regime scores (simple average of normalized scores)
        bull_strength = np.mean(bull_scores) if bull_scores else 0.0
        bear_strength = np.mean(bear_scores) if bear_scores else 0.0
        
        # Determine regime
        regime_diff = abs(bull_strength - bear_strength)
        
        if regime_diff < self.regime_threshold:
            regime = 'neutral'
        elif bull_strength > bear_strength:
            regime = 'bull'
        else:
            regime = 'bear'
        
        return regime, bull_strength, bear_strength

    def handle(self, *args, **opt):
        # Load configuration
        self.regime_threshold = float(opt['regime_threshold'])
        self.min_regime_strength = float(opt['min_regime_strength'])
        self.regime_update_frequency = int(opt['regime_update_frequency'])
        
        self.stdout.write("🚀 REGIME-AWARE TRADE SIMULATOR")
        self.stdout.write("📊 Strategy: Market regime detection using weighted model consensus")
        
        # Load baseline OHLCV data
        baseline_path = opt['baseline_file']
        if not os.path.exists(baseline_path):
            self.stderr.write(f"❌ Baseline file not found: {baseline_path}")
            return
        
        baseline = pd.read_csv(baseline_path)
        baseline['timestamp'] = pd.to_datetime(baseline['timestamp'], utc=True).dt.tz_localize(None)
        baseline = baseline.sort_values(['coin', 'timestamp']).reset_index(drop=True)
        
        # Load all model predictions
        self.stdout.write("▶ Loading model predictions...")
        predictions = self.load_model_predictions()
        
        if not predictions:
            self.stderr.write("❌ No model predictions loaded.")
            return
        
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
        
        # Get unique timestamps
        market_ts = baseline['timestamp'].unique()
        market_ts = sorted(market_ts)
        
        # Initialize regime detection
        current_regime = 'neutral'
        current_bull_strength = 0.0
        current_bear_strength = 0.0
        last_regime_update = 0
        
        self.stdout.write(f"▶ Simulating {len(market_ts)} market timestamps...")
        self.stdout.write(f"📊 Regime update frequency: every {self.regime_update_frequency} timestamps")
        
        for i, ts in enumerate(market_ts):
            if i % 1000 == 0:
                self.stdout.write(f"  Processing timestamp {i+1}/{len(market_ts)}: {ts}")
            
            # ===== REGIME DETECTION (only update periodically for speed) =====
            if i - last_regime_update >= self.regime_update_frequency:
                current_regime, current_bull_strength, current_bear_strength = self.detect_market_regime(predictions, ts)
                last_regime_update = i
                if i % 1000 == 0:  # Only print regime updates occasionally
                    self.stdout.write(f"    📊 Regime: {current_regime} (Bull: {current_bull_strength:.3f}, Bear: {current_bear_strength:.3f})")
            
            # ===== EXITS =====
            still_open = []
            for t in open_trades:
                if ts < t.entry_time:
                    still_open.append(t)
                    continue
                
                # Get OHLCV data for this coin and timestamp
                coin_data = baseline[(baseline['coin'] == t.coin) & (baseline['timestamp'] == ts)]
                if coin_data.empty:
                    still_open.append(t)
                    continue
                
                row = coin_data.iloc[0]
                high = float(row['high'])
                low = float(row['low'])
                close = float(row['close'])
                
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
                        coin_data = baseline[(baseline['coin'] == coin) & (baseline['timestamp'] == ts)]
                        if coin_data.empty:
                            continue
                        
                        row = coin_data.iloc[0]
                        entry_price = float(row['open'])
                        
                        # Check long signals (only in bull regime)
                        if current_regime == 'bull' and current_bull_strength >= self.min_regime_strength:
                            long_key = f'{coin}_long'
                            if long_key in predictions:
                                model_data = predictions[long_key]
                                df = model_data['data']
                                threshold = model_data['threshold']
                                
                                # Find prediction for this timestamp
                                mask = df['timestamp'] == ts
                                if mask.any():
                                    pred_row = df[mask].iloc[0]
                                    if 'pred_prob' in pred_row:
                                        confidence = pred_row['pred_prob']
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
                                            break  # Only one trade per timestamp
                        
                        # Check short signals (only in bear regime)
                        elif current_regime == 'bear' and current_bear_strength >= self.min_regime_strength:
                            short_key = f'{coin}_short'
                            if short_key in predictions:
                                model_data = predictions[short_key]
                                df = model_data['data']
                                threshold = model_data['threshold']
                                
                                # Find prediction for this timestamp
                                mask = df['timestamp'] == ts
                                if mask.any():
                                    pred_row = df[mask].iloc[0]
                                    if 'pred_prob' in pred_row:
                                        confidence = pred_row['pred_prob']
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
                                            break  # Only one trade per timestamp
        
        # Force close remaining trades
        if open_trades:
            last_ts = market_ts[-1]
            for t in open_trades:
                coin_data = baseline[(baseline['coin'] == t.coin) & (baseline['timestamp'] == last_ts)]
                if not coin_data.empty:
                    last_close = float(coin_data.iloc[0]['close'])
                    t.close(last_ts, last_close, 'end_of_data', entry_fee_bps, exit_fee_bps)
                    closed_trades.append(t)
                    reserved_margin -= t.position_size_usd
                    balance += t.net_pl_usd
                    equity_points.append({'timestamp': last_ts, 'equity': balance})
        
        # Sort trades by exit time
        closed_trades.sort(key=lambda x: x.exit_time or datetime.min)
        
        # Save results
        os.makedirs(opt['output_dir'], exist_ok=True)
        out_csv = os.path.join(opt['output_dir'], 'regime_aware_trading_results.csv')
        
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
        eq_csv = os.path.join(opt['output_dir'], 'regime_aware_equity_curve.csv')
        if equity_points:
            pd.DataFrame(equity_points).sort_values('timestamp').to_csv(eq_csv, index=False)
        
        # Print results
        for t in closed_trades:
            result = "WIN" if t.net_pl_usd > 0 else "LOSS"
            self.stdout.write(
                f"[{t.exit_time}] {t.side.upper()} {t.coin} | "
                f"Entry: {t.entry_price:.6f} @ {t.entry_time} | "
                f"Exit: {t.exit_price:.6f} ({t.exit_reason}) | "
                f"Conf: {t.confidence:.3f} | Regime: {t.regime_score:.3f} | "
                f"{result} | Net PnL: ${t.net_pl_usd:,.2f}"
            )
        
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
        
        self.stdout.write(self.style.SUCCESS(f"\n✅ REGIME-AWARE SIMULATION COMPLETE"))
        self.stdout.write(self.style.SUCCESS(f"📈 Equity curve: {eq_csv}"))
        self.stdout.write(self.style.SUCCESS(f"📊 Total trades: {total}, Wins: {wins}, Losses: {losses}, Win %: {win_pct:.2f}%"))
        self.stdout.write(self.style.SUCCESS(f"🐂 Bull trades: {len(bull_trades)}, Wins: {bull_wins}, Win %: {(bull_wins/len(bull_trades)*100) if bull_trades else 0:.2f}%"))
        self.stdout.write(self.style.SUCCESS(f"🐻 Bear trades: {len(bear_trades)}, Wins: {bear_wins}, Win %: {(bear_wins/len(bear_trades)*100) if bear_trades else 0:.2f}%"))
        self.stdout.write(self.style.SUCCESS(f"💰 End Balance: ${balance:,.2f} (Net: ${total_net:,.2f} from start ${opt['initial_balance']:,.2f})"))
