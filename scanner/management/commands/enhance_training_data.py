import pandas as pd
import ta
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Load existing training CSV, add extra indicators, and save enhanced CSV'

    def handle(self, *args, **options):
        input_file = 'new_long_test_data.csv'
        output_file = 'new_long_test_enhanced.csv'

        print(f"Loading {input_file}...")
        df = pd.read_csv(input_file, parse_dates=['timestamp'])
        df = df.set_index(['timestamp', 'coin']).sort_index()

        print("Adding extra indicators...")

        high_14 = df['high'].rolling(window=14).max()
        low_14 = df['low'].rolling(window=14).min()
        df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)

        df['cci'] = ta.trend.CCIIndicator(
            high=df['high'], low=df['low'], close=df['close'], window=20).cci()

        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
            high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=20).chaikin_money_flow()

        print("Dropping rows with null values...")
        df = df.dropna()

        print(f"Saving enhanced dataset to {output_file}...")
        df.to_csv(output_file)

        print("Done.")
