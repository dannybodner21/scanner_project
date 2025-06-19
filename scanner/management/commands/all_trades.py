import pandas as pd
from django.core.management.base import BaseCommand

# Run with:
# python manage.py all_trades updated_predictions.csv short_predictions.csv

class Command(BaseCommand):
    help = 'Simulate combined long and short trading with max 3 open trades and 3 trades per day'

    def add_arguments(self, parser):
        parser.add_argument('long_csv', type=str, help='Path to long predictions CSV')
        parser.add_argument('short_csv', type=str, help='Path to short predictions CSV')

    def handle(self, *args, **options):
        long_df = pd.read_csv(options['long_csv'], parse_dates=['timestamp'])
        short_df = pd.read_csv(options['short_csv'], parse_dates=['timestamp'])

        long_df['position_type'] = 'long'
        short_df['position_type'] = 'short'

        # Rename predictions to avoid collision
        long_df = long_df.rename(columns={'prediction': 'long_prediction'})
        short_df = short_df.rename(columns={'prediction': 'short_prediction'})

        # Align columns
        common_cols = ['timestamp', 'coin', 'high', 'low', 'close']
        long_df = long_df[common_cols + ['long_prediction']]
        short_df = short_df[common_cols + ['short_prediction']]

        df = pd.merge(long_df, short_df, on=common_cols, how='outer')
        df = df.sort_values('timestamp')

        initial_balance = 5000.0
        balance = initial_balance
        leverage = 5
        max_open_trades = 3
        open_trades = []

        tp_pct = 0.06
        sl_pct = 0.03

        total_trades = 0
        wins = 0
        losses = 0
        trades_today = 0
        current_day = None

        for idx, row in df.iterrows():
            timestamp = row['timestamp']
            coin = row['coin']
            high = row['high']
            low = row['low']
            close = row['close']
            this_day = timestamp.date()

            # Reset daily trade counter
            if current_day != this_day:
                current_day = this_day
                trades_today = 0

            # Check for closing open trades
            still_open = []
            for trade in open_trades:
                if timestamp <= trade['entry_timestamp']:
                    still_open.append(trade)
                    continue

                entry_price = trade['entry_price']
                pos_size = trade['position_size']

                if trade['type'] == 'long':
                    tp_price = entry_price * (1 + tp_pct)
                    sl_price = entry_price * (1 - sl_pct)

                    if low <= sl_price:
                        balance -= pos_size * leverage * sl_pct
                        losses += 1
                    elif high >= tp_price:
                        balance += pos_size * leverage * tp_pct
                        wins += 1
                    else:
                        still_open.append(trade)
                        continue

                elif trade['type'] == 'short':
                    tp_price = entry_price * (1 - tp_pct)
                    sl_price = entry_price * (1 + sl_pct)

                    if high >= sl_price:
                        balance -= pos_size * leverage * sl_pct
                        losses += 1
                    elif low <= tp_price:
                        balance += pos_size * leverage * tp_pct
                        wins += 1
                    else:
                        still_open.append(trade)
                        continue

                total_trades += 1

            open_trades = still_open

            # Trade entry constraints
            if len(open_trades) < max_open_trades and trades_today < 3:
                pos_size = balance * 0.10 if balance < 100000 else 10000

                if row.get('long_prediction') == 1:
                    open_trades.append({
                        'type': 'long',
                        'coin': coin,
                        'entry_price': close,
                        'position_size': pos_size,
                        'entry_timestamp': timestamp
                    })
                    trades_today += 1

                elif row.get('short_prediction') == 1 and len(open_trades) < max_open_trades and trades_today < 3:
                    open_trades.append({
                        'type': 'short',
                        'coin': coin,
                        'entry_price': close,
                        'position_size': pos_size,
                        'entry_timestamp': timestamp
                    })
                    trades_today += 1

        self.stdout.write("Backtest complete.")
        self.stdout.write(f"Total trades: {total_trades}, Wins: {wins}, Losses: {losses}")
        self.stdout.write(f"Final balance: ${balance:,.2f}")
