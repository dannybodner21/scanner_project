import pandas as pd
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Simulate sequential trading on test data with 1 open trade max'

    def add_arguments(self, parser):
        parser.add_argument('test_csv', type=str, help='Path to test CSV with predictions and labels')

    def handle(self, *args, **options):
        test_csv = options['test_csv']

        df = pd.read_csv(test_csv, parse_dates=['timestamp'], index_col=0)

        initial_balance = 5000.0
        balance = initial_balance
        open_trade = None  # dict with keys: entry_price, position_size, entry_index
        leverage = 10

        take_profit_pct = 0.06
        stop_loss_pct = 0.03

        total_trades = 0
        wins = 0
        losses = 0

        for idx, row in df.iterrows():
            # If no open trade and model signals long entry (label == 1)
            if open_trade is None and row.get('label', 0) == 1:
                # Determine position size
                if balance < 1000:
                    position_size = balance * 0.10
                else:
                    position_size = 1000.0

                entry_price = row['close']
                open_trade = {
                    'entry_price': entry_price,
                    'position_size': position_size,
                    'entry_index': idx,
                }
                total_trades += 1
                self.stdout.write(f"Trade {total_trades} opened at index {idx}, price: {entry_price:.6f}, position size: {position_size:.2f}, balance: {balance:.2f}")
                continue

            # If open trade, check if TP or SL hit on this bar
            if open_trade is not None:
                high = row['high']
                low = row['low']
                entry_price = open_trade['entry_price']

                tp_price = entry_price * (1 + take_profit_pct)
                sl_price = entry_price * (1 - stop_loss_pct)

                # Check stop loss first (assuming SL hits first)
                if low <= sl_price:
                    loss_amount = open_trade['position_size'] * leverage * stop_loss_pct
                    balance -= loss_amount
                    losses += 1
                    self.stdout.write(f"Trade {total_trades} STOP LOSS at index {idx}, loss: {loss_amount:.2f}, balance: {balance:.2f}")
                    open_trade = None
                elif high >= tp_price:
                    profit_amount = open_trade['position_size'] * leverage * take_profit_pct
                    balance += profit_amount
                    wins += 1
                    self.stdout.write(f"Trade {total_trades} TAKE PROFIT at index {idx}, profit: {profit_amount:.2f}, balance: {balance:.2f}")
                    open_trade = None
                else:
                    # No exit yet, wait for next bar
                    pass

            # Prevent negative balance
            if balance < 0:
                self.stdout.write("Balance dropped below zero. Stopping backtest.")
                break

        self.stdout.write("Backtest complete.")
        self.stdout.write(f"Total trades: {total_trades}, Wins: {wins}, Losses: {losses}")
        self.stdout.write(f"Final balance: ${balance:,.2f}")
