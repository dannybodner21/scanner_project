import pandas as pd
from django.core.management.base import BaseCommand

# Run with:
# python manage.py all_trades updated_predictions.csv short_predictions.csv

class Command(BaseCommand):
    help = 'Simulate combined long and short trading with max 3 open trades and 3 trades per day, with dynamic trailing stop'

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
        initial_sl_pct = 0.03
        trade_fee_pct = 0.004  # 0.4% round trip

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

            if current_day != this_day:
                current_day = this_day
                trades_today = 0

            still_open = []
            for trade in open_trades:
                if timestamp <= trade['entry_timestamp']:
                    still_open.append(trade)
                    continue

                entry_price = trade['entry_price']
                pos_size = trade['position_size']
                trade_type = trade['type']

                # Determine adjusted stop loss
                if trade_type == 'long':
                    move_pct = (high - entry_price) / entry_price
                    sl_price = entry_price * (1 - initial_sl_pct)
                    if move_pct >= 0.05:
                        sl_price = entry_price * 1.03
                    elif move_pct >= 0.04:
                        sl_price = entry_price * 1.02
                    elif move_pct >= 0.03:
                        sl_price = entry_price * 1.01

                    tp_price = entry_price * (1 + tp_pct)

                    if low <= sl_price:
                        balance -= pos_size * leverage * trade_fee_pct
                        profit = (sl_price - entry_price) * leverage * (pos_size / entry_price)
                        balance += profit
                        result = 'STOP LOSS (adjusted)'
                        if profit > 0:
                            wins += 1
                        else:
                            losses += 1
                    elif high >= tp_price:
                        balance -= pos_size * leverage * trade_fee_pct
                        profit = pos_size * leverage * tp_pct
                        balance += profit
                        result = 'TAKE PROFIT'
                        wins += 1
                    else:
                        still_open.append(trade)
                        continue

                elif trade_type == 'short':
                    move_pct = (entry_price - low) / entry_price
                    sl_price = entry_price * (1 + initial_sl_pct)
                    if move_pct >= 0.05:
                        sl_price = entry_price * 0.97
                    elif move_pct >= 0.04:
                        sl_price = entry_price * 0.98
                    elif move_pct >= 0.03:
                        sl_price = entry_price * 0.99

                    tp_price = entry_price * (1 - tp_pct)

                    if high >= sl_price:
                        balance -= pos_size * leverage * trade_fee_pct
                        profit = (entry_price - sl_price) * leverage * (pos_size / entry_price)
                        balance += profit
                        result = 'STOP LOSS (adjusted)'
                        if profit > 0:
                            wins += 1
                        else:
                            losses += 1
                    elif low <= tp_price:
                        balance -= pos_size * leverage * trade_fee_pct
                        profit = pos_size * leverage * tp_pct
                        balance += profit
                        result = 'TAKE PROFIT'
                        wins += 1
                    else:
                        still_open.append(trade)
                        continue

                total_trades += 1

            open_trades = still_open

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
