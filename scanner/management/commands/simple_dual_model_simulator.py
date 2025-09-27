# scanner/management/commands/simple_dual_model_simulator.py
from django.core.management.base import BaseCommand
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from datetime import timezone

class Trade:
    """Simple trade class for dual model simulation"""
    def __init__(self, coin, entry_time, entry_price, leverage, position_size_usd,
                 pred_prob, confidence, trade_id, trade_type, model_type):
        self.coin = coin
        self.entry_time = entry_time
        self.entry_price = float(entry_price)
        self.leverage = float(leverage)
        self.position_size_usd = float(position_size_usd)
        self.pred_prob = float(pred_prob)
        self.confidence = float(confidence) if confidence is not None else np.nan
        self.trade_id = int(trade_id)
        self.trade_type = trade_type  # 'long' or 'short'
        self.model_type = model_type  # 'long_model' or 'short_model'

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

        # Gross leveraged return
        if self.trade_type == 'long':
            self.gross_return = (self.exit_price / self.entry_price - 1.0) * self.leverage
        else:  # short
            self.gross_return = (self.entry_price / self.exit_price - 1.0) * self.leverage
            
        self.gross_pl_usd = self.position_size_usd * self.gross_return

        # Fees on notional round-trip
        notional = self.position_size_usd * self.leverage
        fee_rate = (entry_fee_bps + exit_fee_bps) / 10000.0
        self.fee_usd = notional * fee_rate

        self.net_pl_usd = self.gross_pl_usd - self.fee_usd


class Command(BaseCommand):
    help = 'Simple dual model simulator - use long model for longs, short model for shorts, no regime filtering'

    def add_arguments(self, parser):
        parser.add_argument('--long-predictions', type=str, default='xrp_predictions.csv',
                            help='Long model predictions CSV')
        parser.add_argument('--short-predictions', type=str, default='xrp_simple_short_predictions.csv',
                            help='Short model predictions CSV')
        parser.add_argument('--baseline-file', type=str, default='baseline.csv',
                            help='OHLCV CSV with columns: coin,timestamp,open,high,low,close')

        # Account & sizing
        parser.add_argument('--initial-balance', type=float, default=1000.0)
        parser.add_argument('--position-size', type=float, default=0.25,
                            help='Fraction of AVAILABLE balance used as margin per trade')

        # Leverage
        parser.add_argument('--leverage', type=float, default=15.0,
                            help='Leverage multiplier.')

        # Entry thresholds
        parser.add_argument('--long-threshold', type=float, default=0.5,
                            help='Minimum long model probability to enter long trades.')
        parser.add_argument('--short-threshold', type=float, default=0.5,
                            help='Minimum short model probability to enter short trades.')

        # Targets
        parser.add_argument('--take-profit', type=float, default=0.02,
                            help='Take profit percent of entry price (0.02 = +2 percent)')
        parser.add_argument('--stop-loss', type=float, default=0.01,
                            help='Stop loss percent of entry price (0.01 = -1 percent)')
        parser.add_argument('--max-hold-hours', type=int, default=12,
                            help='Maximum holding time in hours.')

        # Fees
        parser.add_argument('--entry-fee-bps', type=float, default=2.0,
                            help='Entry fee in basis points (2 bps = 0.02 percent)')
        parser.add_argument('--exit-fee-bps', type=float, default=2.0,
                            help='Exit fee in basis points (2 bps = 0.02 percent)')

        # Concurrency
        parser.add_argument('--max-concurrent-trades', type=int, default=1,
                            help='Max simultaneous positions')
        parser.add_argument('--same-bar-policy', type=str, default='sl-first',
                            choices=['sl-first', 'tp-first'],
                            help='If a bar touches both TP and SL, which applies first')
        parser.add_argument('--entry-lag-bars', type=int, default=1,
                            help='Bars to delay entry after signal')

        # Time filtering
        parser.add_argument('--entry-local-tz', type=str, default='America/Los_Angeles')
        parser.add_argument('--entry-start-hour', type=int, default=4,
                            help='Earliest local hour for entries (inclusive)')
        parser.add_argument('--entry-end-hour', type=int, default=23,
                            help='Latest local hour for entries (exclusive)')

        # Risk limits
        parser.add_argument('--max-daily-trades', type=int, default=80,
                            help='Cap number of entries per UTC day')

        parser.add_argument('--output-dir', type=str, default='simulation_results',
                    help='Directory to save results')

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
        long_pred_path = opt['long_predictions']
        short_pred_path = opt['short_predictions']
        base_path = opt['baseline_file']

        print(f"🎯 SIMPLE DUAL MODEL SIMULATION")
        print(f"   Long predictions: {long_pred_path}")
        print(f"   Short predictions: {short_pred_path}")
        print(f"   Baseline OHLCV: {base_path}")
        print(f"   Targets: {opt['take_profit']*100:.2f}% TP / {opt['stop_loss']*100:.2f}% SL")
        print(f"   Leverage: {opt['leverage']:.0f}x")
        print(f"   Long threshold: {opt['long_threshold']:.3f}")
        print(f"   Short threshold: {opt['short_threshold']:.3f}")

        # Load data
        print(f"\n📊 Loading data...")
        
        # Load predictions
        long_df = pd.read_csv(long_pred_path)
        long_df = self._normalize_timestamps(long_df, 'timestamp')
        long_df = long_df.set_index('timestamp')
        
        short_df = pd.read_csv(short_pred_path)
        short_df = self._normalize_timestamps(short_df, 'timestamp')
        short_df = short_df.set_index('timestamp')
        
        # Load baseline OHLCV data
        base_df = pd.read_csv(base_path)
        base_df = self._normalize_timestamps(base_df, 'timestamp')
        
        # Filter for XRPUSDT only
        xrp_df = base_df[base_df['coin'] == 'XRPUSDT'].copy()
        xrp_df = xrp_df.set_index('timestamp')
        
        print(f"   Long predictions: {len(long_df)} rows")
        print(f"   Short predictions: {len(short_df)} rows")
        print(f"   XRP OHLCV: {len(xrp_df)} rows")

        # Find overlapping time range
        start_time = max(long_df.index.min(), short_df.index.min(), xrp_df.index.min())
        end_time = min(long_df.index.max(), short_df.index.max(), xrp_df.index.max())
        
        # Filter to overlapping period
        long_df = long_df.loc[start_time:end_time]
        short_df = short_df.loc[start_time:end_time]
        xrp_df = xrp_df.loc[start_time:end_time]
        
        print(f"   Overlap period: {start_time} to {end_time}")
        print(f"   Overlapping candles: {len(xrp_df)}")

        # Simulation parameters
        initial_balance = opt['initial_balance']
        position_size_pct = opt['position_size']
        leverage = opt['leverage']
        long_threshold = opt['long_threshold']
        short_threshold = opt['short_threshold']
        tp_pct = opt['take_profit']
        sl_pct = opt['stop_loss']
        max_hold_hours = opt['max_hold_hours']
        entry_fee_bps = opt['entry_fee_bps']
        exit_fee_bps = opt['exit_fee_bps']
        same_bar_policy = opt['same_bar_policy']
        entry_lag = opt['entry_lag_bars']
        max_daily_trades = opt['max_daily_trades']
        
        # Timezone for entry filtering
        entry_tz = ZoneInfo(opt['entry_local_tz'])
        entry_start_hour = opt['entry_start_hour']
        entry_end_hour = opt['entry_end_hour']

        # Simulation state
        balance = initial_balance
        trades = []
        active_trade = None
        trade_id = 0
        daily_trade_count = {}
        
        print(f"\n🚀 Starting simulation...")
        print(f"   Initial balance: ${balance:,.2f}")

        # Process each candle
        for i, timestamp in enumerate(xrp_df.index):
            if i % 10000 == 0:
                print(f"   Processed {i:,} candles...")
            
            # Check if we have OHLCV data for this timestamp
            if timestamp not in xrp_df.index:
                continue
                
            ohlcv_row = xrp_df.loc[timestamp]
            
            # Check for active trade exit conditions
            if active_trade is not None:
                exit_price, exit_reason = self._check_exit_conditions(
                    active_trade, timestamp, ohlcv_row, tp_pct, sl_pct, 
                    max_hold_hours, same_bar_policy
                )
                
                if exit_price is not None:
                    # Close the trade
                    active_trade.close(timestamp, exit_price, exit_reason, 
                                     entry_fee_bps, exit_fee_bps)
                    balance += active_trade.net_pl_usd
                    trades.append(active_trade)
                    active_trade = None

            # Check for new entry signals (only if no active trade)
            if active_trade is None:
                # Check daily trade limit
                trade_date = timestamp.date()
                if trade_date not in daily_trade_count:
                    daily_trade_count[trade_date] = 0
                
                if daily_trade_count[trade_date] >= max_daily_trades:
                    continue
                
                # Check time window
                if not self._in_entry_window(timestamp, entry_tz, entry_start_hour, entry_end_hour):
                    continue
                
                # Check long model signal
                if timestamp in long_df.index:
                    long_prob = long_df.loc[timestamp, 'pred_prob']
                    if long_prob >= long_threshold:
                        # Enter long trade
                        entry_time = timestamp + timedelta(minutes=5 * entry_lag)
                        if entry_time in xrp_df.index:
                            entry_price = xrp_df.loc[entry_time, 'open']
                            position_size = balance * position_size_pct
                            
                            active_trade = Trade(
                                coin='XRPUSDT',
                                entry_time=entry_time,
                                entry_price=entry_price,
                                leverage=leverage,
                                position_size_usd=position_size,
                                pred_prob=long_prob,
                                confidence=long_prob,  # Use prob as confidence
                                trade_id=trade_id,
                                trade_type='long',
                                model_type='long_model'
                            )
                            trade_id += 1
                            daily_trade_count[trade_date] += 1
                            continue
                
                # Check short model signal
                if timestamp in short_df.index:
                    short_prob = short_df.loc[timestamp, 'pred_prob']
                    if short_prob >= short_threshold:
                        # Enter short trade
                        entry_time = timestamp + timedelta(minutes=5 * entry_lag)
                        if entry_time in xrp_df.index:
                            entry_price = xrp_df.loc[entry_time, 'open']
                            position_size = balance * position_size_pct
                            
                            active_trade = Trade(
                                coin='XRPUSDT',
                                entry_time=entry_time,
                                entry_price=entry_price,
                                leverage=leverage,
                                position_size_usd=position_size,
                                pred_prob=short_prob,
                                confidence=short_prob,  # Use prob as confidence
                                trade_id=trade_id,
                                trade_type='short',
                                model_type='short_model'
                            )
                            trade_id += 1
                            daily_trade_count[trade_date] += 1

        # Close any remaining active trade
        if active_trade is not None:
            final_timestamp = xrp_df.index[-1]
            final_price = xrp_df.loc[final_timestamp, 'close']
            active_trade.close(final_timestamp, final_price, 'end_of_data', 
                             entry_fee_bps, exit_fee_bps)
            balance += active_trade.net_pl_usd
            trades.append(active_trade)

        # Calculate results
        print(f"\n📊 SIMULATION RESULTS")
        print(f"   Final balance: ${balance:,.2f}")
        print(f"   Total return: {((balance / initial_balance) - 1) * 100:.2f}%")
        print(f"   Total trades: {len(trades)}")
        
        if trades:
            winning_trades = [t for t in trades if t.net_pl_usd > 0]
            losing_trades = [t for t in trades if t.net_pl_usd < 0]
            
            print(f"   Winning trades: {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)")
            print(f"   Losing trades: {len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)")
            print(f"   Average win: ${np.mean([t.net_pl_usd for t in winning_trades]):.2f}")
            print(f"   Average loss: ${np.mean([t.net_pl_usd for t in losing_trades]):.2f}")
            
            # Model breakdown
            long_trades = [t for t in trades if t.model_type == 'long_model']
            short_trades = [t for t in trades if t.model_type == 'short_model']
            
            print(f"\n📈 Trade Breakdown by Model:")
            print(f"   Long model trades: {len(long_trades)}")
            print(f"   Short model trades: {len(short_trades)}")
            
            if long_trades:
                long_wins = [t for t in long_trades if t.net_pl_usd > 0]
                print(f"   Long model win rate: {len(long_wins)/len(long_trades)*100:.1f}%")
                
            if short_trades:
                short_wins = [t for t in short_trades if t.net_pl_usd > 0]
                print(f"   Short model win rate: {len(short_wins)/len(short_trades)*100:.1f}%")

        # Save results
        output_dir = opt['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Save trades to CSV
        trades_data = []
        for trade in trades:
            trades_data.append({
                'trade_id': trade.trade_id,
                'coin': trade.coin,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'trade_type': trade.trade_type,
                'model_type': trade.model_type,
                'pred_prob': trade.pred_prob,
                'confidence': trade.confidence,
                'leverage': trade.leverage,
                'position_size_usd': trade.position_size_usd,
                'gross_return': trade.gross_return,
                'gross_pl_usd': trade.gross_pl_usd,
                'fee_usd': trade.fee_usd,
                'net_pl_usd': trade.net_pl_usd,
                'exit_reason': trade.exit_reason
            })
        
        trades_df = pd.DataFrame(trades_data)
        trades_file = os.path.join(output_dir, 'dual_model_trades.csv')
        trades_df.to_csv(trades_file, index=False)
        
        print(f"\n💾 Results saved to: {output_dir}")
        print(f"   Trades: {trades_file}")

    def _check_exit_conditions(self, trade, timestamp, ohlcv_row, tp_pct, sl_pct, 
                              max_hold_hours, same_bar_policy):
        """Check if trade should be closed and return (exit_price, reason) or (None, None)"""
        
        # Check max hold time
        hold_hours = (timestamp - trade.entry_time).total_seconds() / 3600
        if hold_hours >= max_hold_hours:
            return ohlcv_row['close'], 'max_hold_time'
        
        # Check TP/SL
        high = ohlcv_row['high']
        low = ohlcv_row['low']
        
        if trade.trade_type == 'long':
            tp_price = trade.entry_price * (1 + tp_pct)
            sl_price = trade.entry_price * (1 - sl_pct)
            
            hit_tp = high >= tp_price
            hit_sl = low <= sl_price
            
        else:  # short
            tp_price = trade.entry_price * (1 - tp_pct)
            sl_price = trade.entry_price * (1 + sl_pct)
            
            hit_tp = low <= tp_price
            hit_sl = high >= sl_price
        
        if hit_tp and hit_sl:
            # Both hit in same bar
            if same_bar_policy == 'sl-first':
                return sl_price, 'stop_loss'
            else:
                return tp_price, 'take_profit'
        elif hit_tp:
            return tp_price, 'take_profit'
        elif hit_sl:
            return sl_price, 'stop_loss'
        
        return None, None
