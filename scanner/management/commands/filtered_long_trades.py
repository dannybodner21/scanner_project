
import pandas as pd
from django.core.management.base import BaseCommand
from datetime import datetime, timedelta

'''
Use this file to simulate long trades while also looking at the confidence scores from the short model.
Trying to see if we can eliminate noise by making sure a long trade also has very low confidence from the short model.
'''

class Command(BaseCommand):
    help = 'Simulate long trades based on long and short model predictions with confidence filtering'


    def handle(self, *args, **options):

        long_preds_path = "three_enhanced_predictions.csv"
        short_preds_path = "short_two_enhanced_predictions.csv"
        long_conf_thresh = 0.01
        short_conf_thresh = 0.99

        balance = 5000
        trade_size_pct = 0.25
        max_trades_per_day = 4
        tp_pct = 0.03
        sl_pct = 0.02


        long_df = pd.read_csv(long_preds_path, parse_dates=['timestamp'])
        short_df = pd.read_csv(short_preds_path, parse_dates=['timestamp'])


        long_df['timestamp'] = long_df['timestamp'].dt.floor('min')
        short_df['timestamp'] = short_df['timestamp'].dt.floor('min')

        balance = 5000
        open_trades = []
        closed_trades = []
        active_days = {}

        for _, row in long_df.iterrows():
            coin = row['coin']
            timestamp = row['timestamp']
            prediction = row['prediction']
            confidence = row['prediction_prob']

            if prediction != 1 or confidence < long_conf_thresh:
                continue

            short_match = short_df[(short_df['coin'] == coin) & (short_df['timestamp'] == timestamp)]
            if short_match.empty:
                continue

            short_confidence = short_match.iloc[0]['prediction_prob']
            if short_confidence > short_conf_thresh:
                continue

            date_str = timestamp.date().isoformat()
            if date_str not in active_days:
                active_days[date_str] = {
                    'trades_today': 0,
                    'coins_traded': set()
                }

            if active_days[date_str]['trades_today'] >= 4 or coin in active_days[date_str]['coins_traded']:
                continue

            entry_price = row['close']
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)

            trade = {
                'coin': coin,
                'timestamp': timestamp,
                'entry': entry_price,
                'tp': tp_price,
                'sl': sl_price,
                'size': balance * 0.25
            }

            future = long_df[(long_df['coin'] == coin) & (long_df['timestamp'] > timestamp)].copy()

            for _, future_row in future.iterrows():
                high = future_row['high']
                low = future_row['low']

                if high >= tp_price:
                    balance += trade['size'] * tp_pct
                    trade['exit'] = future_row['timestamp']
                    trade['exit_price'] = tp_price
                    trade['result'] = 'win'
                    closed_trades.append(trade)
                    break
                elif low <= sl_price:
                    balance -= trade['size'] * sl_pct
                    trade['exit'] = future_row['timestamp']
                    trade['exit_price'] = sl_price
                    trade['result'] = 'loss'
                    closed_trades.append(trade)
                    break

            active_days[date_str]['trades_today'] += 1
            active_days[date_str]['coins_traded'].add(coin)

        wins = len([t for t in closed_trades if t['result'] == 'win'])
        losses = len([t for t in closed_trades if t['result'] == 'loss'])

        self.stdout.write(f"Wins: {wins} | Losses: {losses} | Final Balance: ${balance:.2f}")
