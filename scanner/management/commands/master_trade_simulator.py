


# long only with regime gate
'''
python manage.py master_trade_simulator \
  --mode long \
  --long-predictions xrp_predictions.csv \
  --baseline-file baseline.csv \
  --coin XRPUSDT \
  --use-regime yes \
  --neutral-policy block \
  --both-hit-policy sl_first \
  --compound yes --position-size 1.0 \
  --long-threshold 0.5
  '''

# both long and short
'''
python manage.py master_trade_simulator \
  --mode both \
  --long-predictions xrp_predictions.csv \
  --short-predictions xrp_simple_short_predictions.csv \
  --baseline-file baseline.csv \
  --coin XRPUSDT \
  --use-regime yes \
  --neutral-policy block \
  --both-hit-policy sl_first \
  --compound yes --position-size 1.0
'''



# scanner/management/commands/master_trade_simulator.py
from django.core.management.base import BaseCommand
import pandas as pd
import numpy as np
import os
from datetime import timedelta

# ---------------------------
# ----- Helper Indicators ----
# ---------------------------

def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ = ema(series, fast)
    slow_ = ema(series, slow)
    macd_line = fast_ - slow_
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def true_range(high, low, close):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def adx(high, low, close, period: int = 14):
    # Wilder’s DMI/ADX (simplified and stable)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(high, low, close)

    trs = pd.Series(tr).rolling(window=period).sum()
    plus_dms = pd.Series(plus_dm).rolling(window=period).sum()
    minus_dms = pd.Series(minus_dm).rolling(window=period).sum()

    plus_di = 100 * (plus_dms / trs.replace(0, np.nan))
    minus_di = 100 * (minus_dms / trs.replace(0, np.nan))
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / denom) * 100
    adx_ = dx.rolling(window=period).mean()
    return adx_.fillna(20.0)

def bollinger_width(close, period: int = 20):
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + 2*std
    lower = ma - 2*std
    with np.errstate(divide='ignore', invalid='ignore'):
        width = (upper - lower) / ma.replace(0, np.nan)
    return width.fillna(method="bfill").fillna(0)

def donchian_position(high, low, close, period: int = 20):
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    rng = (hh - ll).replace(0, np.nan)
    pos = (close - ll) / rng
    return pos.clip(0, 1).fillna(0.5)

# ---------------------------
# ----- Regime Scoring  -----
# ---------------------------

def compute_regime(df_5m: pd.DataFrame,
                   resample_tf: str = "1H",
                   adx_floor: float = 18.0,
                   bb_med_window: int = 100,
                   ema_fast: int = 50,
                   ema_slow: int = 200,
                   smooth_span: int = 6,
                   hysteresis_level: float = 30.0,
                   hysteresis_bars: int = 3,
                   cooldown_bars: int = 3):
    """
    Compute regime per resampled bar and expand back to base timeframe.
    Returns a DataFrame indexed by base timestamps with:
      ['regime_score', 'regime_smoothed', 'regime_state'] where state in {'bullish','bearish','neutral'}
    """

    # 1) Resample to 1H OHLCV
    ohlc = df_5m[['open','high','low','close','volume']].copy()
    h = pd.DataFrame({
        'open': ohlc['open'].resample(resample_tf).first(),
        'high': ohlc['high'].resample(resample_tf).max(),
        'low':  ohlc['low'].resample(resample_tf).min(),
        'close':ohlc['close'].resample(resample_tf).last(),
        'volume': ohlc['volume'].resample(resample_tf).sum(),
    }).dropna()

    # Not enough data -> neutral
    if h.empty or len(h) < max(ema_slow+5, 60):
        out = pd.DataFrame(index=df_5m.index, data={
            'regime_score': 0.0,
            'regime_smoothed': 0.0,
            'regime_state': 'neutral'
        })
        return out

    # 2) Indicators on 1H
    ema50 = ema(h['close'], ema_fast)
    ema200 = ema(h['close'], ema_slow)
    with np.errstate(divide='ignore', invalid='ignore'):
        ema200_slope = (ema200 - ema200.shift(12)) / ema200.replace(0, np.nan) * 100  # ~12h slope (bps)
    macd_line, macd_sig, macd_hist = macd(h['close'])
    macd_slope = macd_hist - macd_hist.shift(1)
    rsi14 = rsi(h['close'], 14)
    rsi_slope = rsi14 - rsi14.shift(3)
    dc_pos = donchian_position(h['high'], h['low'], h['close'], 20)
    dc_high_up = h['high'].rolling(5).max().diff().clip(lower=0) > 0
    dc_low_up  = h['low'].rolling(5).min().diff().clip(lower=0) > 0
    dc_expand_up = (dc_high_up & dc_low_up)

    adx14 = adx(h['high'], h['low'], h['close'], 14)
    bbw = bollinger_width(h['close'], 20)
    bbw_med = bbw.rolling(bb_med_window, min_periods=10).median()
    chop = (adx14 < adx_floor) & (bbw < bbw_med)

    # 3) Component signals (-1,0,+1)
    def sgn(x, pos_thr, neg_thr=None):
        if neg_thr is None:
            neg_thr = -pos_thr
        return np.where(x > pos_thr, 1, np.where(x < neg_thr, -1, 0))

    trend_1 = np.where((h['close'] > ema50) & (h['close'] > ema200), 1,
                np.where((h['close'] < ema50) & (h['close'] < ema200), -1, 0))
    trend_2 = sgn(ema200_slope.fillna(0), pos_thr=0.15, neg_thr=-0.15)  # bps

    macd_sig1 = sgn(macd_hist.fillna(0), pos_thr=0.0)  # sign only
    macd_sig2 = sgn(macd_slope.fillna(0), pos_thr=0.0) # slope sign

    rsi_sig = np.where((rsi14 > 55) & (rsi_slope > 0), 1,
                np.where((rsi14 < 45) & (rsi_slope < 0), -1, 0))

    dc_sig = np.where((dc_pos > 0.7) & dc_expand_up, 1,
               np.where((dc_pos < 0.3) & (~dc_expand_up), -1, 0))

    # 4) Weight & score to [-100, +100]
    weights = {
        'trend_1': 2.0,
        'trend_2': 2.0,
        'macd1': 1.0,
        'macd2': 1.0,
        'rsi': 1.0,
        'dc': 1.0
    }
    raw = (weights['trend_1']*trend_1 +
           weights['trend_2']*trend_2 +
           weights['macd1']*macd_sig1 +
           weights['macd2']*macd_sig2 +
           weights['rsi']*rsi_sig +
           weights['dc']*dc_sig).astype(float)

    max_w = sum(weights.values())
    score = 100.0 * raw / max_w
    score = pd.Series(score, index=h.index).where(~chop, other=np.nan)  # veto in chop
    score = score.ffill().fillna(0.0)  # hold last (don’t flip in chop)

    smoothed = ema(score, smooth_span)

    # 5) Hysteresis + cooldown → discrete state
    above = smoothed >= hysteresis_level
    below = smoothed <= -hysteresis_level

    up_cnt = above.groupby((~above).cumsum()).cumcount()+1
    up_cnt = up_cnt.where(above, 0)
    dn_cnt = below.groupby((~below).cumsum()).cumcount()+1
    dn_cnt = dn_cnt.where(below, 0)

    tmp_state = np.where(up_cnt >= hysteresis_bars, 'bullish',
                  np.where(dn_cnt >= hysteresis_bars, 'bearish', 'neutral'))
    state_series = pd.Series(tmp_state, index=h.index)

    # cooldown: prevent immediate flip-flop
    last_state = 'neutral'
    cooldown = 0
    cooled_state = []
    for st in state_series:
        s = st
        if s != last_state and s != 'neutral':
            if cooldown > 0:
                s = last_state  # still cooling
            else:
                last_state = s
                cooldown = cooldown_bars
        cooled_state.append(s)
        cooldown = max(cooldown-1, 0)
    state = pd.Series(cooled_state, index=h.index)

    regime_h = pd.DataFrame({
        'regime_score': score,
        'regime_smoothed': smoothed,
        'regime_state': state
    }, index=h.index)

    # 6) Expand back to base timeframe (asof)
    regime_base = regime_h.reindex(df_5m.index.union(regime_h.index)).sort_index().ffill()
    regime_base = regime_base.loc[df_5m.index]
    return regime_base

# ---------------------------
# --------- Trades ----------
# ---------------------------

class Trade:
    """
    Universal trade class supporting both LONG and SHORT trades
    """
    def __init__(self, coin, entry_time, entry_price, leverage, position_size_usd,
                 pred_prob, confidence, trade_id, trade_type='long', model_type='long_model',
                 regime_state_at_entry='neutral'):
        self.coin = coin
        self.trade_type = trade_type.lower()  # 'long' or 'short'
        self.model_type = model_type  # 'long_model' or 'short_model'
        self.entry_time = entry_time  # UTC index timestamp
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

        self.regime_state_at_entry = regime_state_at_entry

    def close(self, exit_time, exit_price, reason, entry_fee_bps, exit_fee_bps):
        self.exit_time = exit_time
        self.exit_price = float(exit_price)
        self.exit_reason = reason

        # Gross leveraged return (supports both long and short)
        if self.trade_type == 'long':
            self.gross_return = (self.exit_price / self.entry_price - 1.0) * self.leverage
        else:  # short
            self.gross_return = (self.entry_price / self.exit_price - 1.0) * self.leverage

        self.gross_pl_usd = self.position_size_usd * self.gross_return

        # Fees on notional round-trip (entry + exit)
        notional = self.position_size_usd * self.leverage
        fee_rate = (entry_fee_bps + exit_fee_bps) / 10000.0
        self.fee_usd = notional * fee_rate

        self.net_pl_usd = self.gross_pl_usd - self.fee_usd

    def __str__(self):
        return (f"Trade {self.trade_id}: {self.trade_type.upper()} {self.coin} @ {self.entry_price:.6f} "
                f"-> {self.exit_price:.6f} | {self.exit_reason} | PnL: ${self.net_pl_usd:.2f}")

# ---------------------------
# -------- Command ----------
# ---------------------------

class Command(BaseCommand):
    help = "Master trade simulator - supports long only, short only, or both models, with optional regime gating"

    # Instance-level defaults for pred matching
    pred_match_mode = "legacy"
    pred_tol = timedelta(seconds=300)

    def add_arguments(self, parser):
        # Model selection
        parser.add_argument("--mode", type=str, required=True,
                            choices=["long", "short", "both"],
                            help="Simulation mode: long, short, or both models")

        # Prediction files
        parser.add_argument("--long-predictions", type=str,
                            help="Path to long model predictions CSV")
        parser.add_argument("--short-predictions", type=str,
                            help="Path to short model predictions CSV")

        # OHLCV data
        parser.add_argument("--baseline-file", type=str, required=True,
                            help="Path to baseline OHLCV CSV file")
        parser.add_argument("--coin", type=str, default="XRPUSDT",
                            help="Coin symbol to simulate (e.g., XRPUSDT)")

        # Trading parameters
        parser.add_argument("--initial-balance", type=float, default=1000.0,
                            help="Initial balance in USD")
        parser.add_argument("--position-size", type=float, default=1.00,
                            help="Position size as fraction of balance (if --compound yes, fraction of current balance)")
        parser.add_argument("--compound", type=str, choices=["yes", "no"], default="no",
                            help="If yes, recalc position size from current balance per trade")
        parser.add_argument("--leverage", type=float, default=15.0,
                            help="Leverage multiplier")
        parser.add_argument("--take-profit", type=float, default=0.02,
                            help="Take profit percentage (0.02 = 2%)")
        parser.add_argument("--stop-loss", type=float, default=0.01,
                            help="Stop loss percentage (0.01 = 1%)")

        # Thresholds
        parser.add_argument("--long-threshold", type=float, default=0.6,
                            help="Long model confidence threshold")
        parser.add_argument("--short-threshold", type=float, default=0.6,
                            help="Short model confidence threshold")
        parser.add_argument("--threshold", type=float,
                            help="Global confidence threshold for both sides (overrides per-side if provided)")

        # Fees
        parser.add_argument("--entry-fee-bps", type=float, default=0.0,
                            help="Entry fee in basis points")
        parser.add_argument("--exit-fee-bps", type=float, default=0.0,
                            help="Exit fee in basis points")

        # Regime filter
        parser.add_argument("--use-regime", type=str, choices=["yes", "no"], default="no",
                            help="Enable bullish/bearish regime gating")
        parser.add_argument("--regime-hysteresis-level", type=float, default=30.0,
                            help="Smoothed regime score threshold for bullish/bearish")
        parser.add_argument("--regime-hysteresis-bars", type=int, default=3,
                            help="Bars required beyond threshold to confirm regime")
        parser.add_argument("--regime-cooldown-bars", type=int, default=3,
                            help="Bars to wait after regime flip")
        parser.add_argument("--neutral-policy", type=str, choices=["block", "allow"], default="block",
                            help="What to do in neutral regime for entries (default block)")

        # Exit ordering when both TP & SL are hit in same bar
        parser.add_argument("--both-hit-policy", type=str, choices=["sl_first", "tp_first", "mid"], default="sl_first",
                            help="How to resolve both TP and SL in the same bar")

        # Prediction timestamp matching
        parser.add_argument("--pred-match", type=str,
                            choices=["backward", "forward", "nearest", "legacy"],
                            default="legacy",
                            help="Align predictions to candles: backward(<=ts), forward(>=ts within tol), nearest(±tol), legacy(nearest ±tol; mirrors old behavior).")
        parser.add_argument("--pred-tolerance-seconds", type=int, default=300,
                            help="Max |delta| in seconds for nearest/legacy/forward matching")

        # Output
        parser.add_argument("--output-dir", type=str, default="simulation_results",
                            help="Output directory for results")

    def handle(self, *args, **options):
        mode = options["mode"]
        baseline_file = options["baseline_file"]
        coin_symbol = options["coin"]

        # Apply global threshold if provided
        if options.get("threshold") is not None:
            options["long_threshold"] = options["threshold"]
            options["short_threshold"] = options["threshold"]

        # Set prediction matching strategy on the instance
        self.pred_match_mode = options["pred_match"]
        self.pred_tol = timedelta(seconds=int(options["pred_tolerance_seconds"]))

        self.stdout.write(f"🎯 MASTER TRADE SIMULATOR - Mode: {mode.upper()}")
        self.stdout.write(f"   Baseline OHLCV: {baseline_file}")
        self.stdout.write(f"   Targets: {options['take_profit']*100:.1f}% TP / {options['stop_loss']*100:.1f}% SL")
        self.stdout.write(f"   Leverage: {options['leverage']:.0f}x")
        self.stdout.write(f"   Pred matching: {self.pred_match_mode} (±{int(self.pred_tol.total_seconds())}s)")

        # Load OHLCV data
        self.stdout.write("\n📊 Loading data...")
        try:
            df_ohlcv = pd.read_csv(baseline_file)
            for col in ['timestamp','open','high','low','close','volume','coin']:
                if col not in df_ohlcv.columns:
                    raise ValueError(f"Missing column in OHLCV: {col}")

            df_ohlcv['timestamp'] = pd.to_datetime(df_ohlcv['timestamp'], utc=True)
            df_ohlcv = df_ohlcv[df_ohlcv['coin'] == coin_symbol].copy()
            df_ohlcv = df_ohlcv.sort_values('timestamp').set_index('timestamp')
            if df_ohlcv.empty:
                raise ValueError(f"No OHLCV rows for coin {coin_symbol}")
            self.stdout.write(f"   OHLCV candles for {coin_symbol}: {len(df_ohlcv)} rows")
        except Exception as e:
            self.stderr.write(f"Error loading OHLCV data: {e}")
            return

        # Load prediction data based on mode
        long_predictions = None
        short_predictions = None

        def load_preds(path):
            df = pd.read_csv(path)
            if 'timestamp' not in df.columns or 'pred_prob' not in df.columns:
                raise ValueError("Predictions must have 'timestamp' and 'pred_prob' columns")
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.sort_values('timestamp').set_index('timestamp')
            return df

        if mode in ["long", "both"]:
            if not options.get("long_predictions"):
                self.stderr.write("Error: --long-predictions required for long or both mode")
                return
            try:
                long_predictions = load_preds(options["long_predictions"])
                self.stdout.write(f"   Long predictions: {len(long_predictions)} rows")
            except Exception as e:
                self.stderr.write(f"Error loading long predictions: {e}")
                return

        if mode in ["short", "both"]:
            if not options.get("short_predictions"):
                self.stderr.write("Error: --short-predictions required for short or both mode")
                return
            try:
                short_predictions = load_preds(options["short_predictions"])
                self.stdout.write(f"   Short predictions: {len(short_predictions)} rows")
            except Exception as e:
                self.stderr.write(f"Error loading short predictions: {e}")
                return

        # Find common time range
        if mode == "both":
            common_start = max(long_predictions.index.min(), short_predictions.index.min())
            common_end = min(long_predictions.index.max(), short_predictions.index.max())
            self.stdout.write(f"   Common time range: {common_start} to {common_end}")
        elif mode == "long":
            common_start = long_predictions.index.min()
            common_end = long_predictions.index.max()
        else:  # short
            common_start = short_predictions.index.min()
            common_end = short_predictions.index.max()

        # Filter OHLCV to common range
        df_ohlcv = df_ohlcv[(df_ohlcv.index >= common_start) & (df_ohlcv.index <= common_end)].copy()
        if df_ohlcv.empty:
            self.stderr.write("No OHLCV rows in common range; aborting.")
            return
        self.stdout.write(f"   OHLCV candles (filtered): {len(df_ohlcv)} rows")

        # Precompute regime if enabled
        regime_df = None
        if options["use_regime"] == "yes":
            self.stdout.write("   Computing regime filter (1H, smoothed + hysteresis + cooldown)...")
            regime_df = compute_regime(
                df_5m=df_ohlcv,
                resample_tf="1H",
                hysteresis_level=options["regime_hysteresis_level"],
                hysteresis_bars=options["regime_hysteresis_bars"],
                cooldown_bars=options["regime_cooldown_bars"]
            )

        # Run simulation
        self.stdout.write(f"\n🚀 Starting simulation...")
        self.run_simulation(
            mode, long_predictions, short_predictions, df_ohlcv, options, coin_symbol, regime_df
        )

    # ---------------- Simulation Core ----------------

    def run_simulation(self, mode, long_predictions, short_predictions, df_ohlcv, options, coin_symbol, regime_df):
        balance = options["initial_balance"]
        leverage = options["leverage"]
        tp_pct = options["take_profit"]
        sl_pct = options["stop_loss"]
        long_threshold = options["long_threshold"]
        short_threshold = options["short_threshold"]
        compound = (options["compound"] == "yes")
        neutral_blocks = (options["neutral_policy"] == "block")

        trades = []
        active_trade = None
        trade_id = 1

        processed_candles = 0
        trades_opened = 0

        for timestamp, candle in df_ohlcv.iterrows():
            processed_candles += 1
            if processed_candles % 10000 == 0:
                self.stdout.write(f"   Processed {processed_candles} candles, opened {trades_opened} trades...")

            # Close existing trade if needed (intrabar TP/SL using current bar H/L)
            if active_trade:
                should_close, reason, exit_price = self.check_tp_sl(
                    active_trade, candle, tp_pct, sl_pct, options["both_hit_policy"]
                )
                if should_close:
                    active_trade.close(
                        timestamp, exit_price, reason,
                        options["entry_fee_bps"], options["exit_fee_bps"]
                    )
                    balance += active_trade.net_pl_usd
                    trades.append(active_trade)
                    active_trade = None

            # Open new trade if none active
            if active_trade is None:
                # position size (constant or compounding)
                position_size_usd = balance * options["position_size"] if compound else options["initial_balance"] * options["position_size"]

        # regime gate
        regime_state_now = 'neutral'
        if regime_df is not None:
            # regime_df is aligned to base index; safe loc
            row = regime_df.loc[timestamp]
            regime_state_now = str(row['regime_state'])

            new_trade = self.check_entry_signals(
                    mode, timestamp, candle,
                    long_predictions, short_predictions,
                    long_threshold, short_threshold, trade_id,
                    position_size_usd, leverage, coin_symbol,
                    regime_state_now, neutral_blocks, options["use_regime"] == "yes"
                )
            if new_trade:
                    active_trade = new_trade
                    trade_id += 1
                    trades_opened += 1

        # Close final trade if still open
        if active_trade:
            active_trade.close(
                df_ohlcv.index[-1], df_ohlcv.iloc[-1]['close'], "END_OF_DATA",
                options["entry_fee_bps"], options["exit_fee_bps"]
            )
            balance += active_trade.net_pl_usd
            trades.append(active_trade)

        # Results
        self.print_results(balance, trades, options)
        self.save_results(trades, options)

    # --------------- Prediction matching ----------------

    def _match_pred(self, preds: pd.DataFrame, ts: pd.Timestamp):
        """
        Match a prediction row to candle timestamp ts according to:
          - backward: latest prediction <= ts (no tolerance)
          - forward: earliest prediction >= ts within tolerance
          - nearest: whichever side is closer within tolerance
          - legacy: exact/nearest within tolerance (mirrors your old behavior)
        Returns the row (as Series) or None.
        """
        if preds is None or preds.empty:
            return None

        idx = preds.index

        mode = self.pred_match_mode
        tol = self.pred_tol

        if mode == "backward":
            pos = idx.get_indexer([ts], method="pad")
            if pos[0] == -1:
                return None
            return preds.iloc[pos[0]]

        # locate neighbors
        right = idx.searchsorted(ts, side="left")
        left = right - 1

        left_row = preds.iloc[left] if left >= 0 else None
        right_row = preds.iloc[right] if right < len(idx) else None

        if mode == "forward":
            if right_row is None:
                return None
            if (right_row.name - ts) <= tol:
                return right_row
            return None

        # nearest & legacy: pick closest side if within tol
        left_ok = left_row is not None and (ts - left_row.name) <= tol
        right_ok = right_row is not None and (right_row.name - ts) <= tol

        if not left_ok and not right_ok:
            return None

        if left_ok and right_ok:
            # choose closer one
            if (ts - left_row.name) <= (right_row.name - ts):
                return left_row
            else:
                return right_row

        return left_row if left_ok else right_row

    # --------------- Entries ----------------

    def check_entry_signals(self, mode, timestamp, candle, long_predictions, short_predictions,
                            long_threshold, short_threshold, trade_id, position_size_usd, leverage,
                            coin_symbol, regime_state_now, neutral_blocks, use_regime=False):
        """Check for entry signals based on mode and regime state"""
        allow_long = True
        allow_short = True
        
        # Only apply regime gating if regime is enabled
        if use_regime:
            if regime_state_now == 'bullish':
                allow_short = False
            elif regime_state_now == 'bearish':
                allow_long = False
            else:  # neutral
                if neutral_blocks:
                    allow_long = False
                    allow_short = False

        if mode == "long":
            if not allow_long:
                return None
            return self.check_long_signal(timestamp, candle, long_predictions,
                                          long_threshold, trade_id, position_size_usd, leverage,
                                          coin_symbol, regime_state_now)

        elif mode == "short":
            if not allow_short:
                return None
            return self.check_short_signal(timestamp, candle, short_predictions,
                                           short_threshold, trade_id, position_size_usd, leverage,
                                           coin_symbol, regime_state_now)
        else:  # both
            long_trade = None
            short_trade = None
            if allow_long:
                long_trade = self.check_long_signal(timestamp, candle, long_predictions,
                                                    long_threshold, trade_id, position_size_usd, leverage,
                                                    coin_symbol, regime_state_now)
            if allow_short:
                short_trade = self.check_short_signal(timestamp, candle, short_predictions,
                                                      short_threshold, trade_id, position_size_usd, leverage,
                                                      coin_symbol, regime_state_now)

            if long_trade and short_trade:
                return long_trade if long_trade.confidence >= short_trade.confidence else short_trade
            return long_trade or short_trade

    def check_long_signal(self, timestamp, candle, long_predictions,
                          threshold, trade_id, position_size_usd, leverage, coin_symbol, regime_state_now):
        if long_predictions is None:
            return None
        row = self._match_pred(long_predictions, timestamp)
        if row is None:
            return None
        pred_prob = float(row['pred_prob'])
        confidence = float(row['confidence']) if 'confidence' in row.index else pred_prob

        if confidence >= threshold:
            return Trade(
                coin=coin_symbol, entry_time=timestamp, entry_price=candle['close'],
                leverage=leverage, position_size_usd=position_size_usd,
                pred_prob=pred_prob, confidence=confidence, trade_id=trade_id,
                trade_type='long', model_type='long_model',
                regime_state_at_entry=regime_state_now
            )
        return None

    def check_short_signal(self, timestamp, candle, short_predictions,
                           threshold, trade_id, position_size_usd, leverage, coin_symbol, regime_state_now):
        if short_predictions is None:
            return None
        row = self._match_pred(short_predictions, timestamp)
        if row is None:
            return None
        pred_prob = float(row['pred_prob'])
        confidence = float(row['confidence']) if 'confidence' in row.index else pred_prob

        if confidence >= threshold:
            return Trade(
                coin=coin_symbol, entry_time=timestamp, entry_price=candle['close'],
                leverage=leverage, position_size_usd=position_size_usd,
                pred_prob=pred_prob, confidence=confidence, trade_id=trade_id,
                trade_type='short', model_type='short_model',
                regime_state_at_entry=regime_state_now
            )
        return None

    # --------------- Exits ----------------

    def check_tp_sl(self, trade, candle, tp_pct, sl_pct, both_policy: str):
        """Check if trade should be closed due to TP or SL on this candle. Returns (bool, reason, exit_price)."""
        entry_price = trade.entry_price
        high = float(candle['high'])
        low = float(candle['low'])

        if trade.trade_type == 'long':
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
            tp_hit = high >= tp_price
            sl_hit = low <= sl_price
            if tp_hit and sl_hit:
                if both_policy == "sl_first":
                    return True, "BOTH_HIT_SAME_BAR_SL_FIRST", sl_price
                elif both_policy == "tp_first":
                    return True, "BOTH_HIT_SAME_BAR_TP_FIRST", tp_price
                else:
                    return True, "BOTH_HIT_SAME_BAR_MID", (tp_price + sl_price) / 2.0
            elif tp_hit:
                return True, "TAKE_PROFIT", tp_price
            elif sl_hit:
                return True, "STOP_LOSS", sl_price
            return False, None, None

        else:  # short
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)
            tp_hit = low <= tp_price
            sl_hit = high >= sl_price
            if tp_hit and sl_hit:
                if both_policy == "sl_first":
                    return True, "BOTH_HIT_SAME_BAR_SL_FIRST", sl_price
                elif both_policy == "tp_first":
                    return True, "BOTH_HIT_SAME_BAR_TP_FIRST", tp_price
                else:
                    return True, "BOTH_HIT_SAME_BAR_MID", (tp_price + sl_price) / 2.0
            elif tp_hit:
                return True, "TAKE_PROFIT", tp_price
            elif sl_hit:
                return True, "STOP_LOSS", sl_price
            return False, None, None

    # --------------- Reporting ----------------

    def print_results(self, final_balance, trades, options):
        initial_balance = options["initial_balance"]
        total_return = (final_balance - initial_balance) / initial_balance * 100

        wins = [t for t in trades if t.net_pl_usd > 0]
        losses = [t for t in trades if t.net_pl_usd < 0]
        win_rate = 100 * len(wins) / len(trades) if trades else 0.0
        avg_win = np.mean([t.net_pl_usd for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.net_pl_usd for t in losses]) if losses else 0.0
        profit_factor = (sum([t.net_pl_usd for t in wins]) / max(1e-9, -sum([t.net_pl_usd for t in losses]))) if losses else float('inf')

        self.stdout.write(f"\n📊 SIMULATION RESULTS")
        self.stdout.write(f"   Final balance: ${final_balance:.2f}")
        self.stdout.write(f"   Total return: {total_return:.2f}%")
        self.stdout.write(f"   Total trades: {len(trades)}")
        self.stdout.write(f"   Winning trades: {len(wins)} ({win_rate:.1f}%)")
        self.stdout.write(f"   Losing trades: {len(losses)} ({100-win_rate:.1f}%)")
        self.stdout.write(f"   Average win: ${avg_win:.2f}")
        self.stdout.write(f"   Average loss: ${avg_loss:.2f}")
        self.stdout.write(f"   Profit Factor: {profit_factor:.2f}")

        # Breakdown by trade type
        if trades:
            long_trades = [t for t in trades if t.trade_type == 'long']
            short_trades = [t for t in trades if t.trade_type == 'short']
            if long_trades:
                long_wr = 100 * len([t for t in long_trades if t.net_pl_usd > 0]) / len(long_trades)
                self.stdout.write(f"\n📈 Trade Breakdown by Type:")
                self.stdout.write(f"   Long trades: {len(long_trades)} (win rate: {long_wr:.1f}%)")
            if short_trades:
                short_wr = 100 * len([t for t in short_trades if t.net_pl_usd > 0]) / len(short_trades)
                self.stdout.write(f"   Short trades: {len(short_trades)} (win rate: {short_wr:.1f}%)")

        # Breakdown by regime at entry (if available)
        if trades and hasattr(trades[0], 'regime_state_at_entry'):
            by_regime = {}
            for t in trades:
                key = t.regime_state_at_entry or 'neutral'
                by_regime.setdefault(key, []).append(t)
            self.stdout.write(f"\n🧭 Performance by Regime at Entry:")
            for key, arr in by_regime.items():
                wr = 100 * len([t for t in arr if t.net_pl_usd > 0]) / len(arr) if arr else 0
                pf = (sum([t.net_pl_usd for t in arr if t.net_pl_usd > 0]) /
                      max(1e-9, -sum([t.net_pl_usd for t in arr if t.net_pl_usd < 0]))) if any(t.net_pl_usd < 0 for t in arr) else float('inf')
                self.stdout.write(f"   {key:>7}: {len(arr):4d} trades | WR {wr:5.1f}% | PF {pf:5.2f}")

    def save_results(self, trades, options):
        output_dir = options["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        trade_data = []
        for t in trades:
            trade_data.append({
                'trade_id': t.trade_id,
                'coin': t.coin,
                'trade_type': t.trade_type,
                'model_type': t.model_type,
                'entry_time': t.entry_time,
                'entry_price': t.entry_price,
                'exit_time': t.exit_time,
                'exit_price': t.exit_price,
                'exit_reason': t.exit_reason,
                'leverage': t.leverage,
                'position_size_usd': t.position_size_usd,
                'pred_prob': t.pred_prob,
                'confidence': t.confidence,
                'gross_return': t.gross_return,
                'gross_pl_usd': t.gross_pl_usd,
                'fee_usd': t.fee_usd,
                'net_pl_usd': t.net_pl_usd,
                'regime_state_at_entry': getattr(t, 'regime_state_at_entry', 'neutral')
            })

        df_trades = pd.DataFrame(trade_data)
        trades_file = os.path.join(output_dir, "master_trades.csv")
        df_trades.to_csv(trades_file, index=False)

        self.stdout.write(f"\n💾 Results saved to: {output_dir}")
        self.stdout.write(f"   Trades: {trades_file}")
