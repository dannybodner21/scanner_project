from django.core.management.base import BaseCommand
import pandas as pd
import os

class Command(BaseCommand):
    help = "Count bearish, bullish, and neutral regimes in combined predictions CSV"

    def add_arguments(self, parser):
        parser.add_argument(
            '--file', 
            type=str, 
            default='xrp_combined_predictions.csv',
            help='Path to combined predictions CSV file'
        )

    def handle(self, *args, **options):
        file_path = options['file']
        
        if not os.path.exists(file_path):
            self.stderr.write(f"Error: File not found: {file_path}")
            return

        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Count regimes
            regime_counts = df['regime'].value_counts()
            
            # Display results
            self.stdout.write(f"\n📊 Regime Counts for {file_path}:")
            self.stdout.write("=" * 50)
            
            total = len(df)
            for regime in ['bullish', 'bearish', 'neutral']:
                count = regime_counts.get(regime, 0)
                percentage = (count / total) * 100
                self.stdout.write(f"{regime.capitalize():>8}: {count:>6,} ({percentage:>5.1f}%)")
            
            self.stdout.write("=" * 50)
            self.stdout.write(f"{'Total':>8}: {total:>6,} (100.0%)")
            
            # Additional stats
            long_signals = df['long_signal'].sum()
            short_signals = df['short_signal'].sum()
            no_signals = total - long_signals - short_signals
            
            self.stdout.write(f"\n📡 Signal Counts:")
            self.stdout.write(f"Long signals:  {long_signals:>6,} ({(long_signals/total*100):>5.1f}%)")
            self.stdout.write(f"Short signals: {short_signals:>6,} ({(short_signals/total*100):>5.1f}%)")
            self.stdout.write(f"No signals:    {no_signals:>6,} ({(no_signals/total*100):>5.1f}%)")
            
        except Exception as e:
            self.stderr.write(f"Error reading file: {e}")
            return
