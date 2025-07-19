from django.core.management.base import BaseCommand, CommandError
import pandas as pd
import numpy as np
import os
from datetime import datetime

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

        self.pnl = self.pnl_pct * self.leverage * 100  # Leverage applied here

class Command(BaseCommand):
    help = 'Run trading simulation on enhanced_predictions.csv including long and short trades'

    def add_arguments(self, parser):



        # short_four_enhanced_predictions.csv
        # 0.62 -> Trades: 327, Wins: 239, Losses: 88, Win %: 73.09%
        # TP: 1% SL: 2%
        # Final Balance: $31,091.05 (Leverage: 15.0x)


        parser.add_argument('--predictions-file', type=str, default='short_four_enhanced_predictions.csv')
        parser.add_argument('--initial-balance', type=float, default=5000)

        parser.add_argument('--confidence-threshold', type=float, default=0.62)

        parser.add_argument('--position-size', type=float, default=0.25)

        parser.add_argument('--stop-loss', type=float, default=0.02)
        parser.add_argument('--take-profit', type=float, default=0.01)

        parser.add_argument('--max-hold-hours', type=int, default=48)

        parser.add_argument('--output-dir', type=str, default='.')
        parser.add_argument('--leverage', type=float, default=15.0)
        parser.add_argument('--trade-direction', type=str, default='short', choices=['long', 'short'])

    def handle(self, *args, **options):
        file_path = options['predictions_file']
        if not os.path.exists(file_path):
            raise CommandError(f"File not found: {file_path}")

        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df = df.sort_values('timestamp')

        trades = []
        open_trades = []
        trade_id_counter = 0
        balance = options['initial_balance']

        position_size_pct = options['position_size']
        stop_loss = options['stop_loss']
        take_profit = options['take_profit']
        max_hold_minutes = options['max_hold_hours'] * 60
        leverage = options['leverage']
        direction = options['trade_direction']

        for i, row in df.iterrows():
            timestamp = row['timestamp']
            coin = row['coin']
            pred = row['prediction']
            conf = row['prediction_prob']
            open_price = row['open']
            high = row['high']
            low = row['low']
            close = row['close']

            for trade in open_trades[:]:
                if trade.coin != coin:
                    continue

                duration = (timestamp - trade.entry_time).total_seconds() / 60
                sl_price = trade.entry_price * (1 + stop_loss) if trade.direction == 'short' else trade.entry_price * (1 - stop_loss)
                tp_price = trade.entry_price * (1 - take_profit) if trade.direction == 'short' else trade.entry_price * (1 + take_profit)

                hit_tp = low <= tp_price if trade.direction == 'short' else high >= tp_price
                hit_sl = high >= sl_price if trade.direction == 'short' else low <= sl_price

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

            if pred == 1 and conf >= options['confidence_threshold']:
                already_open = any(t.coin == coin for t in open_trades)
                if not already_open:
                    trade_id_counter += 1
                    entry_price = open_price
                    trade = Trade(coin, timestamp, entry_price, direction, conf, trade_id_counter, leverage)
                    open_trades.append(trade)

        if not df.empty:
            last_time = df['timestamp'].iloc[-1]
            for trade in open_trades:
                last_close = df[df['coin'] == trade.coin]['close'].iloc[-1]
                trade.close_trade(last_time, last_close, 'end_of_data')
                trades.append(trade)

        results = pd.DataFrame([{
            'trade_id': t.trade_id,
            'coin': t.coin,
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'pnl_pct': t.pnl_pct,
            'pnl': t.pnl,
            'exit_reason': t.exit_reason,
            'confidence': t.confidence,
            'leverage': t.leverage,
            'direction': t.direction
        } for t in trades])

        os.makedirs(options['output_dir'], exist_ok=True)
        out_path = os.path.join(options['output_dir'], 'trading_results.csv')
        results.to_csv(out_path, index=False)

        total_trades = len(trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        losses = sum(1 for t in trades if t.pnl <= 0)
        win_pct = (wins / total_trades) * 100 if total_trades > 0 else 0

        self.stdout.write(self.style.SUCCESS(f"✅ Simulation complete. Results saved to {out_path}"))
        self.stdout.write(self.style.SUCCESS(f"📊 Trades: {total_trades}, Wins: {wins}, Losses: {losses}, Win %: {win_pct:.2f}%"))
        self.stdout.write(self.style.SUCCESS(f"💰 Final Balance: ${balance:,.2f} (Leverage: {leverage}x)"))
