import pandas as pd
from django.core.management.base import BaseCommand
from datetime import date

# python manage.py short_trade_testing four_short_predictions.csv

class Command(BaseCommand):
    help = 'Simulate sequential SHORT trades on test data with 1 open trade max'

    def add_arguments(self, parser):
        parser.add_argument('test_csv', type=str, help='Path to test CSV with predictions and labels')

    def handle(self, *args, **options):
        test_csv = options['test_csv']
        df = pd.read_csv(test_csv, parse_dates=['timestamp'])
        df.sort_values('timestamp', inplace=True)

        if 'prediction' not in df.columns:
            raise ValueError("Missing 'prediction' column. Are you using the correct file with model outputs?")

        initial_balance = 2500.0
        balance = initial_balance
        open_trade = None  # dict with keys: coin, entry_price, position_size, entry_index
        leverage = 10

        take_profit_pct = 0.04  # Profit if price drops 6%
        stop_loss_pct = 0.02    # Loss if price rises 3%

        total_trades = 0
        wins = 0
        losses = 0
        trades_today = 0
        current_day = None

        for idx, row in df.iterrows():
            coin = row['coin'] if 'coin' in row else 'UNKNOWN'
            row_day = row['timestamp'].date()

            if current_day != row_day:
                current_day = row_day
                trades_today = 0  # reset daily counter

            # If no open trade and model signals short entry
            if open_trade is None and row.get('prediction', 0) == 1 and trades_today < 3:
                position_size = balance * 0.25 if balance < 100000 else 20000.0
                entry_price = row['close']
                open_trade = {
                    'coin': coin,
                    'entry_price': entry_price,
                    'position_size': position_size,
                    'entry_index': idx,
                    'entry_timestamp': row['timestamp']
                }
                trades_today += 1
                total_trades += 1
                self.stdout.write(f"SHORT {total_trades} OPENED for {coin} at index {idx}, entry price: {entry_price:.6f}, position size: {position_size:.2f}, balance: {balance:.2f}")
                continue

            # If open trade, evaluate outcome
            if open_trade is not None:
                if row['timestamp'] <= open_trade['entry_timestamp']:
                    continue

                high = row['high']
                low = row['low']
                entry_price = open_trade['entry_price']
                coin_open = open_trade['coin']

                sl_price = entry_price * (1 + stop_loss_pct)  # stop loss on upward movement
                tp_price = entry_price * (1 - take_profit_pct)  # take profit on downward movement

                if high >= sl_price:
                    loss_amount = open_trade['position_size'] * leverage * stop_loss_pct
                    balance -= loss_amount
                    losses += 1
                    self.stdout.write(f"SHORT {total_trades} CLOSED for {coin_open} at index {idx} with STOP LOSS, exit price: {sl_price:.6f}, loss: {loss_amount:.2f}, balance: {balance:.2f}")
                    open_trade = None

                elif low <= tp_price:
                    profit_amount = open_trade['position_size'] * leverage * take_profit_pct
                    balance += profit_amount
                    wins += 1
                    self.stdout.write(f"SHORT {total_trades} CLOSED for {coin_open} at index {idx} with TAKE PROFIT, exit price: {tp_price:.6f}, profit: {profit_amount:.2f}, balance: {balance:.2f}")
                    open_trade = None

            if balance < 0:
                self.stdout.write("Balance dropped below zero. Stopping backtest.")
                break

        self.stdout.write("Backtest complete.")
        self.stdout.write(f"Total trades: {total_trades}, Wins: {wins}, Losses: {losses}")
        self.stdout.write(f"Final balance: ${balance:,.2f}")
