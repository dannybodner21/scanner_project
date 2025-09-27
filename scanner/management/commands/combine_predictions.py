# scanner/management/commands/combine_predictions.py
from django.core.management.base import BaseCommand
import pandas as pd
import numpy as np
import os

class Command(BaseCommand):
    help = 'Combine baseline OHLCV data with all long and short model predictions into one CSV'

    def add_arguments(self, parser):
        parser.add_argument('--baseline-file', type=str, default='baseline.csv', help='Baseline OHLCV CSV file')
        parser.add_argument('--output-file', type=str, default='combined_data.csv', help='Output combined CSV file')
        parser.add_argument('--long-pred-dir', type=str, default='.', help='Directory containing long model prediction files')
        parser.add_argument('--short-pred-dir', type=str, default='.', help='Directory containing short model prediction files')

    def handle(self, *args, **options):
        baseline_file = options['baseline_file']
        output_file = options['output_file']
        long_pred_dir = options['long_pred_dir']
        short_pred_dir = options['short_pred_dir']

        self.stdout.write("🔄 COMBINING PREDICTIONS WITH BASELINE DATA")
        self.stdout.write(f"📊 Baseline file: {baseline_file}")
        self.stdout.write(f"📁 Long predictions dir: {long_pred_dir}")
        self.stdout.write(f"📁 Short predictions dir: {short_pred_dir}")
        self.stdout.write(f"💾 Output file: {output_file}")

        # Define the 15 coins
        coins = ['ADAUSDT', 'ATOMUSDT', 'AVAXUSDT', 'BTCUSDT', 'DOGEUSDT', 'DOTUSDT', 
                'ETHUSDT', 'LINKUSDT', 'LTCUSDT', 'SHIBUSDT', 'SOLUSDT', 'UNIUSDT', 
                'TRXUSDT', 'XLMUSDT', 'XRPUSDT']
        
        # Load baseline data
        self.stdout.write("▶ Loading baseline OHLCV data...")
        if not os.path.exists(baseline_file):
            self.stderr.write(f"❌ Baseline file not found: {baseline_file}")
            return

        baseline_df = pd.read_csv(baseline_file)
        baseline_df['timestamp'] = pd.to_datetime(baseline_df['timestamp'])
        
        # Sort baseline by timestamp, then by coin
        baseline_df = baseline_df.sort_values(['timestamp', 'coin']).reset_index(drop=True)
        self.stdout.write(f"📈 Baseline data: {len(baseline_df)} rows from {baseline_df['timestamp'].min()} to {baseline_df['timestamp'].max()}")

        # Start with baseline data
        combined_df = baseline_df.copy()

        # Load long model predictions
        self.stdout.write("▶ Loading long model predictions...")
        long_predictions = {}
        
        for coin in coins:
            # Convert coin name to lowercase for filename
            coin_lower = coin.lower().replace('usdt', '')
            long_file = os.path.join(long_pred_dir, f"{coin_lower}_predictions.csv")
            
            if os.path.exists(long_file):
                self.stdout.write(f"  📈 Loading {coin} long predictions...")
                long_df = pd.read_csv(long_file)
                long_df['timestamp'] = pd.to_datetime(long_df['timestamp'])
                
                # Store predictions by coin and timestamp
                long_predictions[coin] = long_df.set_index('timestamp')['pred_prob'].to_dict()
                
                self.stdout.write(f"    ✅ {coin} long: {long_df['pred_prob'].notna().sum()} predictions")
            else:
                self.stdout.write(f"  ⚠️  {coin} long predictions not found: {long_file}")
                long_predictions[coin] = {}

        # Load short model predictions
        self.stdout.write("▶ Loading short model predictions...")
        short_predictions = {}
        
        for coin in coins:
            # Convert coin name to lowercase for filename
            coin_lower = coin.lower().replace('usdt', '')
            short_file = os.path.join(short_pred_dir, f"{coin_lower}_simple_short_predictions.csv")
            
            if os.path.exists(short_file):
                self.stdout.write(f"  📉 Loading {coin} short predictions...")
                short_df = pd.read_csv(short_file)
                short_df['timestamp'] = pd.to_datetime(short_df['timestamp'])
                
                # Store predictions by coin and timestamp
                short_predictions[coin] = short_df.set_index('timestamp')['pred_prob'].to_dict()
                
                self.stdout.write(f"    ✅ {coin} short: {short_df['pred_prob'].notna().sum()} predictions")
            else:
                self.stdout.write(f"  ⚠️  {coin} short predictions not found: {short_file}")
                short_predictions[coin] = {}

        # Add confidence columns to combined data
        self.stdout.write("▶ Adding confidence scores...")
        combined_df['long_confidence'] = 0.0
        combined_df['short_confidence'] = 0.0
        
        for idx, row in combined_df.iterrows():
            coin = row['coin']
            timestamp = row['timestamp']
            
            # Get long confidence for this coin and timestamp
            if coin in long_predictions and timestamp in long_predictions[coin]:
                combined_df.at[idx, 'long_confidence'] = long_predictions[coin][timestamp]
            
            # Get short confidence for this coin and timestamp
            if coin in short_predictions and timestamp in short_predictions[coin]:
                combined_df.at[idx, 'short_confidence'] = short_predictions[coin][timestamp]

        # Final sorting
        combined_df = combined_df.sort_values(['timestamp', 'coin']).reset_index(drop=True)

        # Save combined data
        self.stdout.write(f"💾 Saving combined data to {output_file}...")
        combined_df.to_csv(output_file, index=False)

        # Summary statistics
        self.stdout.write("\n📊 COMBINED DATA SUMMARY:")
        self.stdout.write(f"  • Total rows: {len(combined_df):,}")
        self.stdout.write(f"  • Time range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
        self.stdout.write(f"  • Coins: {len(coins)}")
        self.stdout.write(f"  • Rows per timestamp: {len(coins)} (one per coin)")
        
        # Check for missing data
        long_missing = combined_df['long_confidence'].isna().sum()
        short_missing = combined_df['short_confidence'].isna().sum()
        
        self.stdout.write(f"  • Long confidence missing values: {long_missing:,}")
        self.stdout.write(f"  • Short confidence missing values: {short_missing:,}")
        
        # Show sample of data
        self.stdout.write(f"\n📋 SAMPLE DATA (first 10 rows):")
        sample_cols = ['coin', 'timestamp', 'close', 'long_confidence', 'short_confidence']
        self.stdout.write(combined_df[sample_cols].head(10).to_string(index=False))

        self.stdout.write(self.style.SUCCESS(f"\n✅ Combined data saved successfully to {output_file}"))
        self.stdout.write(f"📁 File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
