import pandas as pd
from django.core.management.base import BaseCommand
from datetime import date

# python manage.py trade_testing seven_long_predictions.csv

class Command(BaseCommand):
    help = 'Simulate sequential trading on test data with 1 open trade max'

    def add_arguments(self, parser):
        parser.add_argument('test_csv', type=str, help='Path to test CSV with predictions and labels')

    def handle(self, *args, **options):
        test_csv = options['test_csv']
        df = pd.read_csv(test_csv, parse_dates=['timestamp'])
        df.sort_values('timestamp', inplace=True)

        if 'prediction' not in df.columns:
            raise ValueError("Missing 'prediction' column. Are you using the correct file with model outputs?")

        initial_balance = 5000.0
        balance = initial_balance
        open_trade = None
        leverage = 10

        take_profit_pct = 0.04
        stop_loss_pct = 0.02

        total_trades = 0
        wins = 0
        losses = 0
        trades_today = 0
        current_day = None

        winning_confidence_total = 0
        losing_confidence_total = 0
        highest_confidence = 0
        lowest_confidence = 100

        high_confidence_trades = 0
        winning_high_confidence_trades = 0

        # milestone tracking
        milestones = {
            50_000: None,
            100_000: None,
            250_000: None,
            500_000: None,
            1_000_000: None,
        }
        remaining_milestones = set(milestones.keys())

        for idx, row in df.iterrows():
            coin = row['coin'] if 'coin' in row else 'UNKNOWN'
            row_day = row['timestamp'].date()

            if current_day != row_day:
                current_day = row_day
                trades_today = 0

            if open_trade is None and row.get('prediction', 0) == 1 and trades_today < 4:
                position_size = balance * 0.25 if balance < 25000000 else 60000
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

                self.stdout.write(f"Trade {total_trades} OPENED for {coin} at index {idx}, entry price: {entry_price:.6f}, position size: {position_size:.2f}, balance: {balance:.2f}")
                continue

            if open_trade is not None:
                if row['timestamp'] <= open_trade['entry_timestamp']:
                    continue

                high = row['high']
                low = row['low']
                entry_price = open_trade['entry_price']
                coin_open = open_trade['coin']

                tp_price = entry_price * (1 + take_profit_pct)
                sl_price = entry_price * (1 - stop_loss_pct)

                if low <= sl_price:

                    loss_amount = open_trade['position_size'] * leverage * stop_loss_pct

                    balance -= loss_amount
                    losses += 1

                    entry_index = open_trade['entry_index']
                    entry_confidence = df.loc[entry_index, 'prediction_prob']
                    self.stdout.write(f"losing confidence: {entry_confidence:.4f}")
                    losing_confidence_total += entry_confidence

                    if entry_confidence > 0.97:
                        high_confidence_trades += 1

                    if entry_confidence < lowest_confidence:
                        lowest_confidence = entry_confidence

                    self.stdout.write(f" X Trade {total_trades} CLOSED | Coin: {coin_open} | Date: {row['timestamp'].date()} | Index: {idx} | Balance: {balance:.2f}")
                    open_trade = None

                elif high >= tp_price:

                    profit_amount = open_trade['position_size'] * leverage * take_profit_pct

                    balance += profit_amount
                    wins += 1

                    entry_index = open_trade['entry_index']
                    entry_confidence = df.loc[entry_index, 'prediction_prob']
                    self.stdout.write(f"winning confidence: {entry_confidence:.4f}")
                    winning_confidence_total += entry_confidence

                    if entry_confidence > 0.97:
                        high_confidence_trades += 1
                        winning_high_confidence_trades += 1

                    if entry_confidence > highest_confidence:
                        highest_confidence = entry_confidence

                    self.stdout.write(f"🟢 Trade {total_trades} CLOSED | Coin: {coin_open} | Date: {row['timestamp'].date()} | Index: {idx} | TAKE PROFIT | Exit: {tp_price:.6f} | Profit: {profit_amount:.2f} | Balance: {balance:.2f}")
                    open_trade = None

                # ✅ check for milestone after each trade resolution
                for milestone in sorted(remaining_milestones):
                    if balance >= milestone:
                        milestones[milestone] = row['timestamp'].date()
                        remaining_milestones.remove(milestone)
                        break

            if balance < 0:
                self.stdout.write("Balance dropped below zero. Stopping backtest.")
                break

        average_winning_confidence = winning_confidence_total / wins if wins > 0 else 0
        average_losing_confidence = losing_confidence_total / losses if losses > 0 else 0

        self.stdout.write(f"\naverage winning confidence: {average_winning_confidence:.4f}")
        self.stdout.write(f"average losing confidence: {average_losing_confidence:.4f}")
        self.stdout.write(f"highest confidence: {highest_confidence:.4f}")
        self.stdout.write(f"lowest confidence: {lowest_confidence:.4f}")

        self.stdout.write("\nBacktest complete.")
        self.stdout.write(f"Total trades: {total_trades}, Wins: {wins}, Losses: {losses}")
        self.stdout.write(f"Final balance: ${balance:,.2f}")

        self.stdout.write(f"high confidence trades: {high_confidence_trades}")
        self.stdout.write(f"winning high confidence trades: {winning_high_confidence_trades}")

        self.stdout.write(f"high confidence trade percentage: {winning_high_confidence_trades/high_confidence_trades}%")
        self.stdout.write(f"Final success rate: {wins/total_trades}%")

        self.stdout.write("\nMilestone Dates:")
        for milestone, milestone_date in milestones.items():
            if milestone_date:
                self.stdout.write(f"${milestone:,.0f} reached on {milestone_date}")
            else:
                self.stdout.write(f"${milestone:,.0f} not reached")
