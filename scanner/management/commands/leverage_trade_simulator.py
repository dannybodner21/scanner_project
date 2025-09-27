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
    def __init__(self, coin, entry_time, entry_price, leverage, position_size_usd,
                 pred_prob, confidence, trade_id):
        self.coin = coin
        self.entry_time = entry_time  # naive UTC
        self.entry_price = float(entry_price)
        self.leverage = float(leverage)
        self.position_size_usd = float(position_size_usd)  # margin reserved at entry
        self.pred_prob = float(pred_prob)
        self.confidence = float(confidence) if confidence is not None else np.nan
        self.trade_id = int(trade_id)

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

        # Gross leveraged return (long-only)
        self.gross_return = (self.exit_price / self.entry_price - 1.0) * self.leverage
        self.gross_pl_usd = self.position_size_usd * self.gross_return

        # Fees on notional round-trip (entry + exit)
        notional = self.position_size_usd * self.leverage
        fee_rate = (entry_fee_bps + exit_fee_bps) / 10000.0
        self.fee_usd = notional * fee_rate

        self.net_pl_usd = self.gross_pl_usd - self.fee_usd


class Command(BaseCommand):
    help = 'Simulate trading with leverage using pred_prob (and optional confidence). One trade at a time. Next-bar entries. Fixed TP/SL. Compounding ON.'

    def add_arguments(self, parser):
        parser.add_argument('--predictions-file', type=str, required=True,
                            help='CSV with columns: coin,timestamp,pred_prob[,confidence]')
        parser.add_argument('--baseline-file', type=str, required=True,
                            help='OHLCV CSV with columns: coin,timestamp,open,high,low,close')
        parser.add_argument('--coin', type=str, default='XRPUSDT',
                            help='Coin symbol to trade (e.g., XRPUSDT)')

        # Account & sizing
        parser.add_argument('--initial-balance', type=float, default=1000.0)
        parser.add_argument('--position-size', type=float, default=1.00,
                            help='Fraction of AVAILABLE balance used as margin per trade (e.g., 0.25 = 25%).')

        # Leverage
        parser.add_argument('--leverage', type=float, default=15.0,
                            help='Leverage multiplier.')

        # Entry rule: compare pred_prob to a threshold (+ optional gap) and (optional) confidence threshold
        parser.add_argument('--prob-threshold', type=float, default=0.50,
                            help='Minimum pred_prob to consider an entry (compare to your model threshold).')
        parser.add_argument('--min-prob-gap', type=float, default=0.00,
                            help='Extra buffer above prob threshold; require pred_prob >= threshold + gap.')
        parser.add_argument('--use-confidence', action='store_true', default=False,
                            help='If set, also require confidence >= confidence-threshold.')
        parser.add_argument('--confidence-threshold', type=float, default=0.00,
                            help='Minimum confidence required when --use-confidence is set.')

        # Targets (percent moves on underlying; NOT leveraged)
        parser.add_argument('--take-profit', type=float, default=0.02,
                            help='Take profit % of entry price (0.008 = +0.8%).')
        parser.add_argument('--stop-loss', type=float, default=0.01,
                            help='Stop loss % of entry price (0.004 = -0.4%).')
        parser.add_argument('--max-hold-hours', type=int, default=12,
                            help='Maximum holding time in hours.')

        # Fees (bps on notional, per side)
        parser.add_argument('--entry-fee-bps', type=float, default=0.0,
                            help='Entry fee in basis points (2 bps = 0.02%).')
        parser.add_argument('--exit-fee-bps', type=float, default=0.0,
                            help='Exit fee in basis points (2 bps = 0.02%).')

        # Concurrency & ordering
        parser.add_argument('--max-concurrent-trades', type=int, default=1,
                            help='Max simultaneous positions (set 1 for strict one-at-a-time).')
        parser.add_argument('--same-bar-policy', type=str, default='sl-first',
                            choices=['sl-first', 'tp-first'],
                            help='If a bar touches both TP and SL, which applies first.')
        parser.add_argument('--entry-lag-bars', type=int, default=1,
                            help='Bars to delay entry after signal (1 = enter at next bar open).')

        # Time-based filtering (local hours)
        parser.add_argument('--entry-local-tz', type=str, default='America/Los_Angeles')
        parser.add_argument('--entry-start-hour', type=int, default=4,
                            help='Earliest local hour for entries (inclusive).')
        parser.add_argument('--entry-end-hour', type=int, default=23,
                            help='Latest local hour for entries (exclusive).')

        # Risk limits
        parser.add_argument('--max-daily-trades', type=int, default=80,
                            help='Cap number of entries per UTC day.')

        parser.add_argument('--output-dir', type=str, default='simulation_results',
                    help='Directory to save results (trades, equity curve, summary).')


    # ---- Utility methods ----
    def _in_entry_window(self, ts_utc_naive, tz: ZoneInfo, start_hour: int, end_hour: int) -> bool:
        """Return True if ts falls within [start_hour, end_hour) in local timezone."""
        aware_utc = ts_utc_naive.replace(tzinfo=timezone.utc)
        local = aware_utc.astimezone(tz)
        return (start_hour <= local.hour < end_hour)

    @staticmethod
    def _normalize_timestamps(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Normalize timestamps to naive UTC datetime (5-minute grid assumed)."""
        if pd.api.types.is_datetime64tz_dtype(df[col]):
            df[col] = df[col].dt.tz_convert('UTC').dt.tz_localize(None)
        else:
            df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)
        return df

    def handle(self, *args, **opt):
        pred_path = opt['predictions_file']
        base_path = opt['baseline_file']
        coin_symbol = opt['coin']

        print(f"🎯 LEVERAGE TRADING SIMULATION")
        print(f"   Predictions: {pred_path}")
        print(f"   Baseline OHLCV: {base_path}")
        print(f"   Coin: {coin_symbol}")
        print(f"   Targets: {opt['take_profit']*100:.2f}% TP / {opt['stop_loss']*100:.2f}% SL")
        print(f"   Leverage: {opt['leverage']:.0f}x")
        print(f"   Prob threshold: {opt['prob_threshold']:.3f} (+ gap {opt['min_prob_gap']:.3f})")
        if opt['use_confidence']:
            print(f"   Confidence threshold: {opt['confidence_threshold']:.3f}")

        if not os.path.exists(pred_path) or not os.path.exists(base_path):
            self.stderr.write("❌ Missing required files.")
            return

        # Load data
        predictions = pd.read_csv(pred_path)
        baseline = pd.read_csv(base_path)
        
        # Filter by coin symbol
        predictions = predictions[predictions['coin'] == coin_symbol].copy()
        baseline = baseline[baseline['coin'] == coin_symbol].copy()
        
        if predictions.empty:
            self.stderr.write(f"❌ No predictions found for coin {coin_symbol}")
            return
        if baseline.empty:
            self.stderr.write(f"❌ No OHLCV data found for coin {coin_symbol}")
            return

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

        # Optional confidence column
        has_conf = 'confidence' in predictions.columns

        # Normalize timestamps and sort
        predictions = self._normalize_timestamps(predictions, 'timestamp')
        baseline = self._normalize_timestamps(baseline, 'timestamp')

        predictions.sort_values(['timestamp', 'coin'], inplace=True, kind='mergesort')
        baseline.sort_values(['coin', 'timestamp'], inplace=True, kind='mergesort')

        # Index baseline for O(1) bar lookup
        baseline.set_index(['coin', 'timestamp'], inplace=True)
        if not baseline.index.is_unique:
            dups = baseline.reset_index().duplicated(subset=['coin', 'timestamp'], keep=False).sum()
            raise RuntimeError(f"Baseline has {dups} duplicate (coin,timestamp) rows. Fix baseline file.")

        # Market timeline from baseline (global across coins)
        market_ts = (
            pd.Index(baseline.index.get_level_values('timestamp'))
            .unique()
            .sort_values()
            .tolist()
        )
        print(f"📅 Simulation period: {market_ts[0]} → {market_ts[-1]} ({len(market_ts):,} bars)")

        # ---- Params ----
        balance = float(opt['initial_balance'])
        position_frac = float(opt['position_size'])
        leverage = float(opt['leverage'])

        prob_thr = float(opt['prob_threshold'])
        prob_gap = float(opt['min_prob_gap'])
        eff_prob_thr = prob_thr + prob_gap

        use_conf = bool(opt['use_confidence'])
        conf_thr = float(opt['confidence_threshold'])

        tp = float(opt['take_profit'])
        sl = float(opt['stop_loss'])
        max_hold_minutes = int(opt['max_hold_hours']) * 60

        entry_fee_bps = float(opt['entry_fee_bps'])
        exit_fee_bps = float(opt['exit_fee_bps'])

        max_concurrent = int(opt['max_concurrent_trades'])
        same_bar_policy = opt['same_bar_policy']
        entry_lag = max(0, int(opt['entry_lag_bars']))

        entry_tz = ZoneInfo(opt['entry_local_tz'])
        entry_start_hour = int(opt['entry_start_hour'])
        entry_end_hour = int(opt['entry_end_hour'])

        max_daily_trades = int(opt['max_daily_trades'])

        # ---- State ----
        open_trades = []
        closed_trades = []
        trade_id_counter = 0
        reserved_margin = 0.0
        daily_trade_count = {}  # UTC date -> count
        equity_points = []

        # Helper: fast index of timestamps to positions
        ts_to_index = {ts: i for i, ts in enumerate(market_ts)}

        # Build entry-eligible signal set once
        sig = predictions[predictions['pred_prob'] >= eff_prob_thr].copy()
        if use_conf:
            if not has_conf:
                print("⚠️ --use-confidence set but 'confidence' column not found; ignoring confidence filter.")
            else:
                sig = sig[sig['confidence'] >= conf_thr]
        print(f"📊 Signals above thresholds: {len(sig):,} / {len(predictions):,} "
              f"({len(sig)/max(1,len(predictions))*100:.1f}%)")

        # ---- Main loop ----
        print("\n🚀 Starting simulation...")
        for i, ts in enumerate(market_ts):
            # ===== EXITS =====
            still_open = []
            for t in open_trades:
                # Respect entry lag: trade not live until its entry_time
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

                # Fixed TP & SL levels
                tp_price = t.entry_price * (1.0 + tp)
                sl_price = t.entry_price * (1.0 - sl)

                hit_tp = high >= tp_price
                hit_sl = low <= sl_price

                def _close_and_update(reason, px):
                    nonlocal reserved_margin, balance
                    t.close(ts, px, reason, entry_fee_bps, exit_fee_bps)
                    closed_trades.append(t)
                    reserved_margin -= t.position_size_usd
                    balance += t.net_pl_usd
                    equity_points.append({'timestamp': ts, 'equity': balance})
                    win_loss = "WIN" if t.net_pl_usd > 0 else "LOSS"
                    print(f"📉 Close #{t.trade_id}: {t.coin} @ {t.exit_price:.6f} on {t.exit_time.strftime('%Y-%m-%d %H:%M')} "
                          f"({reason}) - {win_loss}: ${t.net_pl_usd:+.2f} (Balance: ${balance:.2f})")

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
                    _close_and_update('stop_loss', sl_price)
                    continue

                # Max hold
                dur_minutes = (ts - t.entry_time).total_seconds() / 60.0
                if dur_minutes >= max_hold_minutes:
                    _close_and_update('max_hold', close)
                else:
                    still_open.append(t)

            open_trades = still_open

            # ===== ENTRIES =====
            if len(open_trades) < max_concurrent:
                idx = ts_to_index.get(ts)
                if idx is None:
                    continue
                entry_idx = idx + entry_lag
                if entry_idx >= len(market_ts):
                    continue
                entry_ts = market_ts[entry_idx]

                # Local time window
                if not self._in_entry_window(entry_ts, entry_tz, entry_start_hour, entry_end_hour):
                    continue

                # Daily cap (UTC day)
                entry_date = entry_ts.date()
                used_today = daily_trade_count.get(entry_date, 0)
                if used_today >= max_daily_trades:
                    continue

                # Signals stamped at 'ts'; enter at 'entry_ts' (next bar by default)
                sig_rows = sig[sig['timestamp'] == ts]
                if sig_rows.empty:
                    continue

                # Highest prob first
                sig_rows = sig_rows.sort_values('pred_prob', ascending=False)

                for _, row in sig_rows.iterrows():
                    if len(open_trades) >= max_concurrent:
                        break

                    coin = row['coin']
                    prob = float(row['pred_prob'])
                    conf = float(row['confidence']) if has_conf and pd.notna(row['confidence']) else None

                    # Only one position per coin
                    if any(ot.coin == coin for ot in open_trades):
                        continue

                    key = (coin, entry_ts)
                    if key not in baseline.index:
                        continue

                    # Available margin
                    available = balance - reserved_margin
                    if available <= 10.0:
                        continue

                    position_size_usd = available * position_frac
                    if position_size_usd < 5.0:
                        continue

                    entry_price = float(baseline.loc[key]['open'])
                    if entry_price <= 0:
                        continue

                    # Open trade
                    trade_id_counter += 1
                    t = Trade(
                        coin=coin,
                        entry_time=entry_ts,
                        entry_price=entry_price,
                        leverage=leverage,
                        position_size_usd=position_size_usd,
                        pred_prob=prob,
                        confidence=conf,
                        trade_id=trade_id_counter
                    )
                    open_trades.append(t)
                    reserved_margin += position_size_usd

                    # Increment daily count immediately (so multiple same-day entries respect cap)
                    used_today += 1
                    daily_trade_count[entry_date] = used_today

                    conf_str = f", conf: {conf:.3f}" if conf is not None else ""
                    print(f"📈 Open #{trade_id_counter}: {coin} @ {entry_price:.6f} on {entry_ts.strftime('%Y-%m-%d %H:%M')} "
                          f"(pred_prob: {prob:.3f}{conf_str}, size: ${position_size_usd:.0f}, {leverage:.0f}x)")

        # Force close any remaining trades at end of data (at last available close per coin)
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
                    last_close = float(baseline.loc[key]['close'])
                    t.close(last_ts, last_close, 'end_of_data', entry_fee_bps, exit_fee_bps)
                    closed_trades.append(t)
                    reserved_margin -= t.position_size_usd
                    balance += t.net_pl_usd
                    equity_points.append({'timestamp': last_ts, 'equity': balance})
                    win_loss = "WIN" if t.net_pl_usd > 0 else "LOSS"
                    print(f"📉 Close #{t.trade_id}: {t.coin} @ {last_close:.6f} on {last_ts.strftime('%Y-%m-%d %H:%M')} "
                          f"(end_of_data) - {win_loss}: ${t.net_pl_usd:+.2f} (Balance: ${balance:.2f})")

        # Sort closes by time
        closed_trades.sort(key=lambda x: x.exit_time or datetime.min)

        # ===== SAVE RESULTS =====
        os.makedirs(opt['output_dir'], exist_ok=True)
        results_csv = os.path.join(opt['output_dir'], 'leverage_trading_results.csv')
        equity_csv  = os.path.join(opt['output_dir'], 'leverage_equity_curve.csv')

        results_df = pd.DataFrame([{
            'trade_id': t.trade_id,
            'coin': t.coin,
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'pred_prob': t.pred_prob,
            'confidence': t.confidence,
            'position_size_usd': t.position_size_usd,
            'leverage': t.leverage,
            'exit_reason': t.exit_reason,
            'gross_return': t.gross_return,
            'gross_pl_usd': t.gross_pl_usd,
            'fee_usd': t.fee_usd,
            'net_pl_usd': t.net_pl_usd,
        } for t in closed_trades])
        results_df.to_csv(results_csv, index=False)

        if equity_points:
            equity_df = pd.DataFrame(equity_points).sort_values('timestamp')
            equity_df.to_csv(equity_csv, index=False)

        # ===== METRICS =====
        total_trades = len(closed_trades)
        if total_trades == 0:
            print("❌ No trades executed!")
            return

        wins = sum(1 for t in closed_trades if t.net_pl_usd > 0)
        losses = total_trades - wins
        win_rate = wins / total_trades * 100

        winning_trades = [t for t in closed_trades if t.net_pl_usd > 0]
        losing_trades  = [t for t in closed_trades if t.net_pl_usd <= 0]
        avg_win  = float(np.mean([t.net_pl_usd for t in winning_trades])) if winning_trades else 0.0
        avg_loss = float(np.mean([t.net_pl_usd for t in losing_trades]))  if losing_trades else 0.0

        total_return = (balance / opt['initial_balance'] - 1) * 100

        if equity_points:
            equity_series = pd.Series([ep['equity'] for ep in equity_points])
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak * 100
            max_drawdown = float(drawdown.min())
        else:
            max_drawdown = 0.0

        # Exit reason breakdown
        exit_reasons = {}
        for t in closed_trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        # ===== SUMMARY =====
        print("\n" + "="*60)
        print("📊 SIMULATION SUMMARY")
        print("="*60)
        print(f"💰 Performance: ${opt['initial_balance']:,.0f} → ${balance:,.0f} ({total_return:+.1f}%)")
        print(f"📈 Trades: {total_trades:,} total | {wins:,} wins ({win_rate:.1f}%) | {losses:,} losses")
        print(f"💵 Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}")
        if losses > 0 and avg_loss != 0:
            profit_factor = abs((avg_win * wins) / (avg_loss * losses))
            print(f"⚡ Profit Factor: {profit_factor:.2f}")
        print(f"📉 Max Drawdown: {max_drawdown:.1f}%")

        print("\n🎯 Exit Reasons:")
        for reason, count in sorted(exit_reasons.items()):
            pct = count / total_trades * 100
            print(f"   {reason}: {count:,} ({pct:.1f}%)")

        print(f"\n📁 Results saved to: {results_csv}")
        self.stdout.write(self.style.SUCCESS("✅ Simulation Complete!"))
