# scanner/management/commands/simulate_trades_leverage.py
from django.core.management.base import BaseCommand
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from datetime import timezone

class Trade:
    """
    Long-only trade with margin sizing (position_size_usd = margin at entry).
    PnL math:
      qty = (margin * lev) / entry
      gross_pl_usd = qty * (exit - entry) = (margin * lev) * (exit/entry - 1)
      fees (bps) charged on notional both sides.
    """
    def __init__(self, coin, entry_time, entry_price, leverage, position_size_usd, conf, trade_id):
        self.coin = coin
        self.entry_time = entry_time # naive UTC
        self.entry_price = float(entry_price)
        self.leverage = float(leverage)
        self.position_size_usd = float(position_size_usd) # margin reserved at entry
        self.confidence = float(conf)
        self.trade_id = int(trade_id)

        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None

        self.gross_return = 0.0
        self.gross_pl_usd = 0.0
        self.fee_usd = 0.0
        self.net_pl_usd = 0.0

        # dynamic/protective stop state
        self.sl_armed = False
        self.dynamic_sl = None # when armed, = entry * (1 + locked_gain)

    def maybe_arm_profit_stop(self, bar_high, arm_mult=1.006, sl_mult=1.003):
        """
        Arm a profit-stop once price reaches arm_mult * entry (e.g., +0.6%).
        After arming, SL becomes sl_mult * entry (e.g., lock +0.3%).
        Updated for leverage trading - smaller profit targets.
        """
        if not self.sl_armed and bar_high >= self.entry_price * arm_mult:
            self.sl_armed = True
            self.dynamic_sl = self.entry_price * sl_mult

    def current_sl_price(self, base_sl_mult):
        # base_sl_mult = (1 - sl), e.g. 0.996 for 0.4% SL
        if self.sl_armed and self.dynamic_sl is not None:
            return self.dynamic_sl
        return self.entry_price * base_sl_mult

    def close(self, exit_time, exit_price, reason, entry_fee_bps, exit_fee_bps):
        self.exit_time = exit_time
        self.exit_price = float(exit_price)
        self.exit_reason = reason

        # Gross leveraged return (long-only)
        self.gross_return = (self.exit_price / self.entry_price - 1.0) * self.leverage
        self.gross_pl_usd = self.position_size_usd * self.gross_return

        # Fees on notional round-trip (entry + exit)
        notional = self.position_size_usd * self.leverage
        fee_rate = (entry_fee_bps + exit_fee_bps) / 10000.0
        self.fee_usd = notional * fee_rate

        self.net_pl_usd = self.gross_pl_usd - self.fee_usd

class Command(BaseCommand):
    help = 'Simulate trading with LEVERAGE-OPTIMIZED model (0.8% TP / 0.4% SL). Compounding ON.'

    def add_arguments(self, parser):
        parser.add_argument('--predictions-file', type=str, required=True,
                            help='CSV with columns: coin,timestamp,pred_prob (from new model)')
        parser.add_argument('--baseline-file', type=str, required=True,
                            help='OHLCV CSV with columns: coin,timestamp,open,high,low,close')

        # UPDATED DEFAULTS for leverage model
        parser.add_argument('--initial-balance', type=float, default=1000.0)
        parser.add_argument('--position-size', type=float, default=1.00,
                            help='Fraction of AVAILABLE balance per trade (0.25 = 25% - conservative for leverage)')
        parser.add_argument('--leverage', type=float, default=15.0,
                            help='Leverage multiplier (12x default for 0.8%/0.4% targets)')

        # LEVERAGE-APPROPRIATE TARGETS (match training)
        parser.add_argument('--confidence-threshold', type=float, default=0.38,
                            help='Minimum confidence for trade entry (calibrated model)')
        parser.add_argument('--take-profit', type=float, default=0.02,
                            help='Take profit % (0.008 = 0.8% - matches training)')
        parser.add_argument('--stop-loss', type=float, default=0.01,
                            help='Stop loss % (0.004 = 0.4% - matches training)')
        parser.add_argument('--max-hold-hours', type=int, default=100,
                            help='Max hold time (matches training - 2 hours)')

        parser.add_argument('--entry-fee-bps', type=float, default=2.0,
                            help='Entry fee in basis points (2 bps = 0.02%)')
        parser.add_argument('--exit-fee-bps', type=float, default=2.0,
                            help='Exit fee in basis points (2 bps = 0.02%)')

        parser.add_argument('--max-concurrent-trades', type=int, default=1,
                            help='Max simultaneous positions')
        parser.add_argument('--output-dir', type=str, default='simulation_results')

        parser.add_argument('--same-bar-policy', type=str, default='sl-first', 
                            choices=['sl-first', 'tp-first'])
        parser.add_argument('--entry-lag-bars', type=int, default=1,
                            help='Bars to delay entry after signal (1=next bar open)')

        # LEVERAGE-APPROPRIATE profit-lock (smaller targets)
        parser.add_argument('--profit-arm-threshold', type=float, default=1.006,
                            help='Arm profit stop at +0.6% unrealized gain')
        parser.add_argument('--profit-stop-level', type=float, default=1.003,
                            help='Lock profit at +0.3% once armed')

        # Time-based filtering
        parser.add_argument('--entry-local-tz', type=str, default='America/Los_Angeles')
        parser.add_argument('--entry-start-hour', type=int, default=6,
                            help='Earliest hour for entries (6 AM PT = active US premarket)')
        parser.add_argument('--entry-end-hour', type=int, default=23,
                            help='Latest hour for entries (10 PM PT)')
        
        # Risk management for leverage
        parser.add_argument('--min-confidence-gap', type=float, default=0.05,
                            help='Minimum gap above threshold for entry (avoid borderline trades)')
        parser.add_argument('--max-daily-trades', type=int, default=80,
                            help='Maximum trades per day (risk management)')

    # ---- Utility methods ----
    def _in_entry_window(self, ts_utc_naive, tz: ZoneInfo, start_hour: int, end_hour: int) -> bool:
        """Return True if ts falls within [start_hour, end_hour) in local timezone."""
        aware_utc = ts_utc_naive.replace(tzinfo=timezone.utc)
        local = aware_utc.astimezone(tz)
        return (start_hour <= local.hour < end_hour)

    @staticmethod
    def _normalize_timestamps(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Normalize timestamps to naive UTC datetime."""
        if pd.api.types.is_datetime64tz_dtype(df[col]):
            df[col] = df[col].dt.tz_convert('UTC').dt.tz_localize(None)
        else:
            df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)
        return df

    def handle(self, *args, **opt):
        pred_path = opt['predictions_file']
        base_path = opt['baseline_file']

        print(f"🎯 LEVERAGE-OPTIMIZED TRADING SIMULATION")
        print(f"   Predictions: {pred_path}")
        print(f"   Baseline OHLCV: {base_path}")
        print(f"   Targets: {opt['take_profit']*100:.1f}% TP / {opt['stop_loss']*100:.1f}% SL")
        print(f"   Leverage: {opt['leverage']:.0f}x")
        print(f"   Confidence threshold: {opt['confidence_threshold']:.1%}")

        if not os.path.exists(pred_path) or not os.path.exists(base_path):
            self.stderr.write("❌ Missing required files.")
            return

        # Load data
        predictions = pd.read_csv(pred_path)
        baseline = pd.read_csv(base_path)

        # Validate required columns
        pred_required = ['coin', 'timestamp', 'pred_prob']
        base_required = ['coin', 'timestamp', 'open', 'high', 'low', 'close']
        
        for c in pred_required:
            if c not in predictions.columns:
                self.stderr.write(f"❌ Predictions missing '{c}' column. Required: {pred_required}")
                return
                
        for c in base_required:
            if c not in baseline.columns:
                self.stderr.write(f"❌ Baseline missing '{c}' column. Required: {base_required}")
                return

        print(f"✅ Data loaded: {len(predictions):,} predictions, {len(baseline):,} OHLCV bars")

        # Normalize timestamps
        predictions = self._normalize_timestamps(predictions, 'timestamp')
        baseline = self._normalize_timestamps(baseline, 'timestamp')

        # Sort data
        predictions.sort_values(['timestamp', 'coin'], inplace=True, kind='mergesort')
        baseline.sort_values(['coin', 'timestamp'], inplace=True, kind='mergesort')

        # Create baseline index for fast lookups
        baseline.set_index(['coin', 'timestamp'], inplace=True)
        if not baseline.index.is_unique:
            dups = baseline.reset_index().duplicated(subset=['coin', 'timestamp'], keep=False).sum()
            raise RuntimeError(f"Baseline has {dups} duplicate (coin,timestamp) rows. Fix baseline file.")

        # Market timeline
        market_ts = (
            pd.Index(baseline.index.get_level_values('timestamp'))
            .unique()
            .sort_values()
            .tolist()
        )
        print(f"📅 Simulation period: {market_ts[0]} to {market_ts[-1]} ({len(market_ts):,} bars)")

        # ---- Simulation parameters ----
        balance = float(opt['initial_balance'])
        position_frac = float(opt['position_size'])
        leverage = float(opt['leverage'])
        conf_thr = float(opt['confidence_threshold'])
        min_conf_gap = float(opt['min_confidence_gap'])
        tp = float(opt['take_profit'])
        sl = float(opt['stop_loss'])
        max_hold_minutes = int(opt['max_hold_hours']) * 60
        entry_fee_bps = float(opt['entry_fee_bps'])
        exit_fee_bps = float(opt['exit_fee_bps'])
        max_concurrent = int(opt['max_concurrent_trades'])
        max_daily_trades = int(opt['max_daily_trades'])
        same_bar_policy = opt['same_bar_policy']
        entry_lag = max(0, int(opt['entry_lag_bars']))

        # Profit-lock parameters (leverage-appropriate)
        arm_thr = float(opt['profit_arm_threshold'])  # 0.6%
        lock_gain = float(opt['profit_stop_level'])   # 0.3%

        # Time window parameters
        entry_tz = ZoneInfo(opt['entry_local_tz'])
        entry_start_hour = int(opt['entry_start_hour'])
        entry_end_hour = int(opt['entry_end_hour'])

        # State tracking
        open_trades = []
        closed_trades = []
        trade_id_counter = 0
        reserved_margin = 0.0
        daily_trade_count = {}  # date -> count
        
        # Performance tracking
        equity_points = []
        ts_to_index = {ts: i for i, ts in enumerate(market_ts)}

        # Filter predictions to only high-confidence with gap
        effective_threshold = conf_thr + min_conf_gap
        high_conf_predictions = predictions[predictions['pred_prob'] >= effective_threshold].copy()
        print(f"📊 High-confidence predictions: {len(high_conf_predictions):,} / {len(predictions):,} "
              f"({len(high_conf_predictions)/len(predictions)*100:.1f}%)")

        # Main simulation loop
        print(f"\n🚀 Starting simulation...")
        
        for i, ts in enumerate(market_ts):

            # ===== PROCESS EXITS =====
            still_open = []
            for t in open_trades:
                # Skip if trade hasn't started yet (respect entry lag)
                if ts < t.entry_time:
                    still_open.append(t)
                    continue

                key = (t.coin, ts)
                if key not in baseline.index:
                    still_open.append(t)
                    continue

                brow = baseline.loc[key]
                high = float(brow['high'])
                low = float(brow['low'])
                close = float(brow['close'])

                # Update dynamic stop loss
                t.maybe_arm_profit_stop(
                    bar_high=high,
                    arm_mult=(1.0 + arm_thr),
                    sl_mult=(1.0 + lock_gain),
                )

                # Calculate exit levels
                tp_price = t.entry_price * (1.0 + tp)
                base_sl_mult = (1.0 - sl)
                sl_price = t.current_sl_price(base_sl_mult=base_sl_mult)

                hit_tp = high >= tp_price
                hit_sl = low <= sl_price

                def _close_and_update(reason, px):
                    nonlocal reserved_margin, balance
                    t.close(ts, px, reason, entry_fee_bps, exit_fee_bps)
                    closed_trades.append(t)
                    # Release margin and compound
                    reserved_margin -= t.position_size_usd
                    balance += t.net_pl_usd
                    equity_points.append({'timestamp': ts, 'equity': balance})
                    
                    # Print trade close details
                    win_loss = "WIN" if t.net_pl_usd > 0 else "LOSS"
                    print(f"📉 Close #{t.trade_id}: {t.coin} @ {t.exit_price:.6f} on {t.exit_time.strftime('%Y-%m-%d %H:%M')} "
                          f"({reason}) - {win_loss}: ${t.net_pl_usd:+.2f} (Balance: ${balance:.2f})")

                # Handle simultaneous TP/SL hits
                if hit_tp and hit_sl:
                    if same_bar_policy == 'sl-first':
                        _close_and_update('both_hit_sl_first', sl_price)
                    else:
                        _close_and_update('both_hit_tp_first', tp_price)
                    continue

                if hit_tp:
                    _close_and_update('take_profit', tp_price)
                    continue

                if hit_sl:
                    stop_reason = 'stop_loss_locked' if t.sl_armed else 'stop_loss'
                    _close_and_update(stop_reason, sl_price)
                    continue

                # Check max hold time
                dur_minutes = (ts - t.entry_time).total_seconds() / 60.0
                if dur_minutes >= max_hold_minutes:
                    _close_and_update('max_hold', close)
                else:
                    still_open.append(t)

            open_trades = still_open

            # ===== PROCESS ENTRIES =====
            if len(open_trades) < max_concurrent:
                # Calculate entry timestamp with lag
                idx = ts_to_index.get(ts)
                if idx is None:
                    continue

                entry_idx = idx + entry_lag
                if entry_idx >= len(market_ts):
                    continue

                entry_ts = market_ts[entry_idx]

                # Check time window
                if not self._in_entry_window(entry_ts, entry_tz, entry_start_hour, entry_end_hour):
                    continue

                # Check daily trade limit
                entry_date = entry_ts.date()
                daily_count = daily_trade_count.get(entry_date, 0)
                if daily_count >= max_daily_trades:
                    continue

                # Get signals for this timestamp
                sig_rows = high_conf_predictions[high_conf_predictions['timestamp'] == ts]
                if sig_rows.empty:
                    continue

                # Sort by confidence (trade highest confidence first)
                sig_rows = sig_rows.sort_values('pred_prob', ascending=False)
                
                for _, row in sig_rows.iterrows():
                    if len(open_trades) >= max_concurrent:
                        break

                    coin = row['coin']
                    prob = float(row['pred_prob'])

                    # Skip if we already have a position in this coin
                    if any(ot.coin == coin for ot in open_trades):
                        continue

                    # Check if we have OHLCV data for entry
                    key = (coin, entry_ts)
                    if key not in baseline.index:
                        continue

                    # Calculate position size from available equity
                    available = balance - reserved_margin
                    if available <= 10.0:  # Minimum balance check
                        continue

                    position_size_usd = available * position_frac
                    if position_size_usd < 5.0:  # Minimum position size
                        continue

                    # Get entry price
                    brow = baseline.loc[key]
                    entry_price = float(brow['open'])
                    if entry_price <= 0:
                        continue

                    # Create trade
                    trade_id_counter += 1
                    t = Trade(
                        coin=coin,
                        entry_time=entry_ts,
                        entry_price=entry_price,
                        leverage=leverage,
                        position_size_usd=position_size_usd,
                        conf=prob,
                        trade_id=trade_id_counter
                    )
                    
                    open_trades.append(t)
                    reserved_margin += position_size_usd
                    
                    # Update daily trade count
                    daily_trade_count[entry_date] = daily_count + 1
                    
                    # Print trade entry details
                    print(f"📈 Open #{trade_id_counter}: {coin} @ {entry_price:.6f} on {entry_ts.strftime('%Y-%m-%d %H:%M')} "
                          f"(conf: {prob:.3f}, size: ${position_size_usd:.0f}, {leverage:.0f}x)")

        # Force close remaining trades at end of data
        if open_trades:
            print(f"🔚 Force-closing {len(open_trades)} remaining trades...")
            base_reset = baseline.reset_index()
            last_by_coin = base_reset.groupby('coin')['timestamp'].max().to_dict()

            for t in open_trades:
                last_ts = last_by_coin.get(t.coin)
                if last_ts is None:
                    continue
                    
                key = (t.coin, last_ts)
                if key in baseline.index:
                    brow = baseline.loc[key]
                    last_close = float(brow['close'])
                    t.close(last_ts, last_close, 'end_of_data', entry_fee_bps, exit_fee_bps)
                    closed_trades.append(t)
                    reserved_margin -= t.position_size_usd
                    balance += t.net_pl_usd
                    equity_points.append({'timestamp': last_ts, 'equity': balance})
                    
                    # Print force close details
                    win_loss = "WIN" if t.net_pl_usd > 0 else "LOSS"
                    print(f"📉 Close #{t.trade_id}: {t.coin} @ {last_close:.6f} on {last_ts.strftime('%Y-%m-%d %H:%M')} "
                          f"(end_of_data) - {win_loss}: ${t.net_pl_usd:+.2f} (Balance: ${balance:.2f})")

        # Sort trades by exit time
        closed_trades.sort(key=lambda x: x.exit_time or datetime.min)

        # ===== SAVE RESULTS =====
        os.makedirs(opt['output_dir'], exist_ok=True)
        
        # Trade results CSV
        results_csv = os.path.join(opt['output_dir'], 'leverage_trading_results.csv')
        results_df = pd.DataFrame([{
            'trade_id': t.trade_id,
            'coin': t.coin,
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'confidence': t.confidence,
            'position_size_usd': t.position_size_usd,
            'leverage': t.leverage,
            'exit_reason': t.exit_reason,
            'gross_return': t.gross_return,
            'gross_pl_usd': t.gross_pl_usd,
            'fee_usd': t.fee_usd,
            'net_pl_usd': t.net_pl_usd,
            'sl_armed': getattr(t, 'sl_armed', False),
            'dynamic_sl': getattr(t, 'dynamic_sl', None)
        } for t in closed_trades])
        results_df.to_csv(results_csv, index=False)

        # Equity curve CSV
        equity_csv = os.path.join(opt['output_dir'], 'leverage_equity_curve.csv')
        if equity_points:
            equity_df = pd.DataFrame(equity_points).sort_values('timestamp')
            equity_df.to_csv(equity_csv, index=False)

        # ===== CALCULATE PERFORMANCE METRICS =====
        total_trades = len(closed_trades)
        if total_trades == 0:
            print("❌ No trades executed!")
            return

        wins = sum(1 for t in closed_trades if t.net_pl_usd > 0)
        losses = total_trades - wins
        win_rate = wins / total_trades * 100

        # PnL metrics
        winning_trades = [t for t in closed_trades if t.net_pl_usd > 0]
        losing_trades = [t for t in closed_trades if t.net_pl_usd <= 0]
        
        avg_win = np.mean([t.net_pl_usd for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.net_pl_usd for t in losing_trades]) if losing_trades else 0
        
        total_return = (balance / opt['initial_balance'] - 1) * 100
        
        # Risk metrics
        if equity_points:
            equity_series = pd.Series([ep['equity'] for ep in equity_points])
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak * 100
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0

        # ===== PRINT SUMMARY =====
        print(f"\n" + "="*60)
        print(f"📊 SIMULATION SUMMARY")
        print(f"="*60)
        print(f"💰 Performance: ${opt['initial_balance']:,.0f} → ${balance:,.0f} ({total_return:+.1f}%)")
        print(f"📈 Trades: {total_trades:,} total | {wins:,} wins ({win_rate:.1f}%) | {losses:,} losses")
        print(f"💵 Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}")
        if avg_loss != 0 and losses > 0:
            profit_factor = abs(avg_win * wins / (avg_loss * losses))
            print(f"⚡ Profit Factor: {profit_factor:.2f}")
        print(f"📉 Max Drawdown: {max_drawdown:.1f}%")
        
        # Exit reason breakdown
        exit_reasons = {}
        for t in closed_trades:
            reason = t.exit_reason
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
        print(f"\n🎯 Exit Reasons:")
        for reason, count in sorted(exit_reasons.items()):
            pct = count / total_trades * 100
            print(f"   {reason}: {count:,} ({pct:.1f}%)")

        print(f"\n📁 Results saved to: {results_csv}")
        self.stdout.write(self.style.SUCCESS(f"✅ Simulation Complete!"))
