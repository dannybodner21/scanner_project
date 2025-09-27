from django.core.management.base import BaseCommand
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from datetime import timezone
from collections import Counter


class Trade:
    """
    Supports LONG and SHORT with margin sizing (position_size_usd = margin at entry).

    PnL math:
      notional = margin * lev
      LONG : gross_pl_usd = notional * (exit/entry - 1)
      SHORT: gross_pl_usd = notional * (entry/exit - 1)
      Fees (bps) charged on notional both sides.
    """
    def __init__(self, coin, entry_time, entry_price, leverage, position_size_usd, conf, trade_id, side='short'):
        self.coin = coin
        self.side = side.lower()  # 'long' or 'short'
        self.entry_time = entry_time            # naive UTC
        self.entry_price = float(entry_price)
        self.leverage = float(leverage)
        self.position_size_usd = float(position_size_usd)  # margin reserved at entry
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
        self.dynamic_sl = None  # long: entry*(1+lock); short: entry*(1-lock)

    def maybe_arm_profit_stop(self, bar_high, bar_low, arm_thr=0.02, lock_gain=0.01):
        """
        Arm a profit-stop:
          LONG : arm when bar_high >= entry*(1+arm_thr) -> dynamic_sl = entry*(1+lock_gain)
          SHORT: arm when bar_low  <= entry*(1-arm_thr) -> dynamic_sl = entry*(1-lock_gain)
        """
        if self.sl_armed:
            return
        if self.side == 'long':
            if bar_high >= self.entry_price * (1.0 + arm_thr):
                self.sl_armed = True
                self.dynamic_sl = self.entry_price * (1.0 + lock_gain)
        else:  # short
            if bar_low <= self.entry_price * (1.0 - arm_thr):
                self.sl_armed = True
                self.dynamic_sl = self.entry_price * (1.0 - lock_gain)

    def current_sl_price(self, sl):
        """
        Return the active stop price:
          LONG  base SL = entry*(1 - sl)
          SHORT base SL = entry*(1 + sl)
          If profit-stop armed, return dynamic_sl instead.
        """
        if self.sl_armed and self.dynamic_sl is not None:
            return self.dynamic_sl
        if self.side == 'long':
            return self.entry_price * (1.0 - sl)
        else:
            return self.entry_price * (1.0 + sl)

    def close(self, exit_time, exit_price, reason, entry_fee_bps, exit_fee_bps):
        self.exit_time = exit_time
        self.exit_price = float(exit_price)
        self.exit_reason = reason

        notional = self.position_size_usd * self.leverage
        if self.side == 'long':
            # leveraged return on margin
            self.gross_return = (self.exit_price / self.entry_price - 1.0) * self.leverage
            self.gross_pl_usd = notional * (self.exit_price / self.entry_price - 1.0)
        else:
            # short: profit when price falls
            self.gross_return = (self.entry_price / self.exit_price - 1.0) * self.leverage
            self.gross_pl_usd = notional * (self.entry_price / self.exit_price - 1.0)

        fee_rate = (entry_fee_bps + exit_fee_bps) / 10000.0
        self.fee_usd = notional * fee_rate
        self.net_pl_usd = self.gross_pl_usd - self.fee_usd


class Command(BaseCommand):
    help = 'Simulate trading from predictions and OHLCV with TP/SL first-hit exits + profit-lock stop. Supports LONG/SHORT (default SHORT). Compounding ON.'

    def add_arguments(self, parser):
        parser.add_argument('--predictions-file', type=str, default='uni_short_predictions.csv')
        parser.add_argument('--baseline-file', type=str, default='baseline_ohlcv.csv')
        parser.add_argument('--coin', type=str, default='XRPUSDT',
                            help='Coin symbol to trade (e.g., XRPUSDT)')

        parser.add_argument('--side', type=str, default='short', choices=['long','short'],
                            help='Trade direction (default short). For short, pred_prob is interpreted as P(short).')

        parser.add_argument('--initial-balance', type=float, default=1000.0)
        parser.add_argument('--position-size', type=float, default=1.00,
                            help='Fraction of AVAILABLE balance per new trade (e.g., 0.25 = 25%)')
        parser.add_argument('--leverage', type=float, default=15.0)

        parser.add_argument('--confidence-threshold', type=float, default=0.5)
        parser.add_argument('--take-profit', type=float, default=0.02, help='2% move in your favor (down for SHORT)')
        parser.add_argument('--stop-loss', type=float, default=0.01, help='1% move against you (up for SHORT)')
        parser.add_argument('--max-hold-hours', type=int, default=6, help='Max hold in hours')

        parser.add_argument('--entry-fee-bps', type=float, default=5.0)
        parser.add_argument('--exit-fee-bps', type=float, default=5.0)

        parser.add_argument('--max-concurrent-trades', type=int, default=1)
        parser.add_argument('--output-dir', type=str, default='.')

        parser.add_argument('--same-bar-policy', type=str, default='sl-first', choices=['sl-first', 'tp-first'])
        parser.add_argument('--entry-lag-bars', type=int, default=1,
                            help='Bars to delay entry after signal (0=same bar open, 1=next bar open)')

        # Profit-lock config (arm at 2%, lock 1% by default)
        parser.add_argument('--profit-arm-threshold', type=float, default=1.02,
                            help='Arm profit stop when unrealized gain >= this (abs).')
        parser.add_argument('--profit-stop-level', type=float, default=1.01,
                            help='Once armed, stop sits at this gain relative to entry (fixed, not trailing).')

        # Entry window (kept, but instrumented so it can’t silently block everything)
        parser.add_argument('--entry-local-tz', type=str, default='America/Los_Angeles', help='Timezone for entry window checks')
        parser.add_argument('--entry-start-hour', type=int, default=8, help='Earliest local hour (inclusive) to allow new entries')
        parser.add_argument('--entry-end-hour', type=int, default=23, help='Latest local hour (exclusive) to allow new entries')

    # ---- Utility methods ----
    def _in_entry_window(self, ts_utc_naive, tz: ZoneInfo, start_hour: int, end_hour: int) -> bool:
        """True if ts (naive UTC) falls within [start_hour, end_hour) local time."""
        aware_utc = ts_utc_naive.replace(tzinfo=timezone.utc)
        local = aware_utc.astimezone(tz)
        return (start_hour <= local.hour < end_hour)

    @staticmethod
    def _normalize_timestamps(df: pd.DataFrame, col: str) -> pd.DataFrame:
        # Convert anything to UTC then drop tz -> naive UTC.
        # If already tz-aware, convert to UTC; if naive, treat as UTC.
        if pd.api.types.is_datetime64tz_dtype(df[col]):
            df[col] = df[col].dt.tz_convert('UTC').dt.tz_localize(None)
        else:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce').dt.tz_localize(None)
        if df[col].isna().any():
            bad = int(df[col].isna().sum())
            raise ValueError(f"{bad} rows have unparseable timestamps in column '{col}'.")
        return df

    def handle(self, *args, **opt):
        pred_path = opt['predictions_file']
        base_path = opt['baseline_file']
        coin_symbol = opt['coin']

        if not os.path.exists(pred_path) or not os.path.exists(base_path):
            self.stderr.write("❌ Missing required files.")
            return

        predictions = pd.read_csv(pred_path)
        baseline = pd.read_csv(base_path)

        # Column sanity
        for c in ['coin', 'timestamp', 'pred_prob']:
            if c not in predictions.columns:
                self.stderr.write(f"❌ Predictions file missing '{c}' column.")
                return
        for c in ['coin', 'timestamp', 'open', 'high', 'low', 'close']:
            if c not in baseline.columns:
                self.stderr.write(f"❌ Baseline file missing '{c}' column.")
                return

        # Normalize timestamps to naive UTC
        predictions = self._normalize_timestamps(predictions, 'timestamp')
        baseline = self._normalize_timestamps(baseline, 'timestamp')

        # Enforce single coin throughout
        predictions = predictions[predictions['coin'] == coin_symbol].copy()
        baseline = baseline[baseline['coin'] == coin_symbol].copy()

        if predictions.empty:
            self.stderr.write(f"❌ No predictions found for coin {coin_symbol}")
            return
        if baseline.empty:
            self.stderr.write(f"❌ No OHLCV data found for coin {coin_symbol}")
            return

        # Sort and de-dup
        predictions.sort_values(['timestamp', 'coin'], inplace=True, kind='mergesort')
        baseline.sort_values(['coin', 'timestamp'], inplace=True, kind='mergesort')

        if predictions.duplicated(subset=['coin', 'timestamp']).any():
            dct = predictions[predictions.duplicated(subset=['coin', 'timestamp'], keep=False)]
            ndup = len(dct)
            self.stderr.write(f"⚠️  Predictions contain {ndup} duplicate (coin,timestamp) rows for {coin_symbol}. Keeping first per ts.")
            predictions = predictions.drop_duplicates(subset=['coin', 'timestamp'], keep='first')

        # Enforce unique (coin,timestamp) index for baseline
        baseline.set_index(['coin', 'timestamp'], inplace=True)
        if not baseline.index.is_unique:
            dups = baseline.reset_index().duplicated(subset=['coin', 'timestamp'], keep=False).sum()
            raise RuntimeError(f"Baseline OHLCV has {dups} duplicate (coin,timestamp) rows. Fix your baseline file.")

        # Market timeline from baseline
        market_ts = (
            pd.Index(baseline.index.get_level_values('timestamp'))
            .unique()
            .sort_values()
            .tolist()
        )

        # ---- Sim config ----
        side = opt['side'].lower()
        balance = float(opt['initial_balance'])              # equity, compounded
        position_frac = float(opt['position_size'])          # fraction of AVAILABLE equity per trade
        leverage = float(opt['leverage'])
        conf_thr = float(opt['confidence_threshold'])
        tp = float(opt['take_profit'])
        sl = float(opt['stop_loss'])
        max_hold_minutes = int(opt['max_hold_hours']) * 60
        entry_fee_bps = float(opt['entry_fee_bps'])
        exit_fee_bps  = float(opt['exit_fee_bps'])
        max_concurrent = int(opt['max_concurrent_trades'])
        same_bar_policy = opt['same_bar_policy']
        entry_lag = max(0, int(opt['entry_lag_bars']))

        arm_thr = float(opt['profit_arm_threshold'])     # 0.02 default
        lock_gain = float(opt['profit_stop_level'])      # 0.01 default

        entry_tz = ZoneInfo(opt['entry_local_tz'])
        entry_start_hour = int(opt['entry_start_hour'])
        entry_end_hour = int(opt['entry_end_hour'])

        open_trades = []
        closed_trades = []
        trade_id_counter = 0

        reserved_margin = 0.0

        # Precompute map from timestamp -> index
        ts_to_index = {ts: i for i, ts in enumerate(market_ts)}

        # Keep only necessary prediction columns
        predictions = predictions[['coin', 'timestamp', 'pred_prob']]

        # Diagnostics: prediction ts alignment to baseline
        pred_ts = set(predictions['timestamp'].unique().tolist())
        base_ts = set(market_ts)
        missing_alignment = len(pred_ts - base_ts)
        if missing_alignment > 0:
            self.stderr.write(f"⚠️  {missing_alignment} unique prediction timestamps have no matching OHLCV bar and will be ignored.")

        # Track equity over time (on every close)
        equity_points = []

        # Drop-reason counters for entries
        drop_reasons = Counter()

        for ts in market_ts:
            # ===== EXITS =====
            still_open = []
            for t in open_trades:
                if ts < t.entry_time:
                    still_open.append(t)
                    continue

                key = (t.coin, ts)
                if key not in baseline.index:
                    still_open.append(t)
                    continue

                brow = baseline.loc[key]
                high = float(brow['high'])
                low  = float(brow['low'])
                close = float(brow['close'])

                # Arm profit stop (symmetric)
                t.maybe_arm_profit_stop(
                    bar_high=high,
                    bar_low=low,
                    arm_thr=arm_thr,
                    lock_gain=lock_gain,
                )

                # Price levels
                if t.side == 'long':
                    tp_price = t.entry_price * (1.0 + tp)
                    sl_price = t.current_sl_price(sl)
                    hit_tp = (high >= tp_price)
                    hit_sl = (low  <= sl_price)
                else:  # short
                    tp_price = t.entry_price * (1.0 - tp)
                    sl_price = t.current_sl_price(sl)  # base is entry*(1+sl) unless locked to entry*(1-lock)
                    hit_tp = (low  <= tp_price)
                    hit_sl = (high >= sl_price)

                def _close_and_update(reason, px):
                    nonlocal reserved_margin, balance
                    t.close(ts, px, reason, entry_fee_bps, exit_fee_bps)
                    closed_trades.append(t)
                    reserved_margin -= t.position_size_usd
                    balance += t.net_pl_usd
                    equity_points.append({'timestamp': ts, 'equity': balance})

                if hit_tp and hit_sl:
                    if same_bar_policy == 'sl-first':
                        _close_and_update('both_hit_same_bar_sl_first', sl_price)
                    else:
                        _close_and_update('both_hit_same_bar_tp_first', tp_price)
                    continue

                if hit_tp:
                    _close_and_update('take_profit', tp_price)
                    continue

                if hit_sl:
                    _close_and_update('stop_loss_locked' if t.sl_armed else 'stop_loss', sl_price)
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
                # Check for signals at current timestamp
                sig_rows = predictions[predictions['timestamp'] == ts]
                if sig_rows.empty:
                    continue

                # Apply entry lag - enter at next available timestamp
                idx = ts_to_index.get(ts, None)
                if idx is None:
                    # should not happen; ts is from market_ts
                    drop_reasons['no_market_index'] += len(sig_rows)
                    continue

                entry_idx = idx + entry_lag
                if entry_idx >= len(market_ts):
                    drop_reasons['entry_lag_out_of_range'] += len(sig_rows)
                    continue

                entry_ts = market_ts[entry_idx]

                if not self._in_entry_window(entry_ts, entry_tz, entry_start_hour, entry_end_hour):
                    drop_reasons['outside_entry_window'] += len(sig_rows)
                    continue

                # Highest prob first
                sig_rows = sig_rows.sort_values('pred_prob', ascending=False)
                for _, row in sig_rows.iterrows():
                    if len(open_trades) >= max_concurrent:
                        drop_reasons['concurrency_limit'] += 1
                        break

                    prob = float(row['pred_prob'])
                    if prob < conf_thr:
                        drop_reasons['below_conf_threshold'] += 1
                        continue

                    coin = row['coin']
                    if any(ot.coin == coin for ot in open_trades):
                        drop_reasons['already_have_open_trade_for_coin'] += 1
                        continue

                    # Use entry_ts for OHLCV lookup
                    key = (coin, entry_ts)
                    if key not in baseline.index:
                        drop_reasons['no_ohlcv_at_entry_ts'] += 1
                        continue

                    available = balance - reserved_margin
                    if available <= 0:
                        drop_reasons['no_available_balance'] += 1
                        continue

                    position_size_usd = available * position_frac
                    if position_size_usd <= 0:
                        drop_reasons['nonpositive_position_size'] += 1
                        continue

                    brow = baseline.loc[key]
                    entry_price = float(brow['open'])

                    trade_id_counter += 1
                    t = Trade(
                        coin=coin,
                        entry_time=entry_ts,
                        entry_price=entry_price,
                        leverage=leverage,
                        position_size_usd=position_size_usd,
                        conf=prob,
                        trade_id=trade_id_counter,
                        side=side
                    )
                    open_trades.append(t)
                    reserved_margin += position_size_usd

        # Force-close leftovers at last available close per coin (only one coin here, but keep generic)
        if len(open_trades) > 0:
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

        closed_trades.sort(key=lambda x: x.exit_time or datetime.min)

        # Output CSVs (suffix with _short/_long to avoid clobbering)
        os.makedirs(opt['output_dir'], exist_ok=True)
        suffix = side
        out_csv = os.path.join(opt['output_dir'], f'trading_results_{suffix}.csv')
        out_df = pd.DataFrame([{
            'trade_id': t.trade_id,
            'side': t.side,
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
            'sl_armed': t.sl_armed,
            'dynamic_sl': t.dynamic_sl
        } for t in closed_trades])
        out_df.to_csv(out_csv, index=False)

        eq_csv = os.path.join(opt['output_dir'], f'equity_curve_{suffix}.csv')
        if equity_points:
            pd.DataFrame(equity_points).sort_values('timestamp').to_csv(eq_csv, index=False)

        # Print trade log & summary
        for t in closed_trades:
            result = "WIN" if t.net_pl_usd > 0 else "LOSS"
            self.stdout.write(
                f"[{t.exit_time}] {t.side.upper()} {t.coin} | "
                f"Entry: {t.entry_price:.6f} @ {t.entry_time} | "
                f"Exit: {t.exit_price:.6f} ({t.exit_reason}) | "
                f"Conf: {t.confidence:.3f} | "
                f"{result} | Net PnL: ${t.net_pl_usd:,.2f}"
            )

        total = len(closed_trades)
        wins = sum(1 for t in closed_trades if t.net_pl_usd > 0)
        losses = total - wins
        win_pct = (wins / total * 100.0) if total > 0 else 0.0
        total_net = sum(t.net_pl_usd for t in closed_trades)

        self.stdout.write(self.style.SUCCESS(f"\n✅ Simulation complete. Saved: {out_csv}"))
        if equity_points:
            self.stdout.write(self.style.SUCCESS(f"📈 Equity curve: {eq_csv}"))
        self.stdout.write(self.style.SUCCESS(
            f"📊 Trades: {total}, Wins: {wins}, Losses: {losses}, Win %: {win_pct:.2f}%"))
        self.stdout.write(self.style.SUCCESS(
            f"💰 End Balance: ${balance:,.2f}  (Net: ${total_net:,.2f} from start ${opt['initial_balance']:,.2f})"))

        # Entry diagnostics
        if drop_reasons:
            self.stdout.write("\n🔍 Entry filters summary (dropped signal reasons):")
            for k, v in drop_reasons.most_common():
                self.stdout.write(f" - {k}: {v}")

        # If zero trades, surface likely culprits
        if total == 0:
            self.stderr.write("\n⚠️  No trades were taken. Common causes:")
            self.stderr.write("   • Prediction timestamps don’t align to baseline candles.")
            self.stderr.write("   • Confidence threshold too high.")
            self.stderr.write("   • Entry window hours excluded all candidate entries.")
            self.stderr.write("   • No OHLCV row at the computed entry timestamp (after entry_lag).")
