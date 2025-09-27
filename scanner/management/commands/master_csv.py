import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from django.core.management.base import BaseCommand
import argparse

class Command(BaseCommand):
    help = 'Create master CSV with all data needed for simulation'

    def add_arguments(self, parser):
        parser.add_argument("--start-date", type=str, default="2025-07-01 00:00:00+00:00")
        parser.add_argument("--end-date", type=str, default="2025-08-19 23:55:00+00:00")
        parser.add_argument("--output-file", type=str, default="master_simulation.csv")

    def handle(self, *args, **options):
        start_date = pd.to_datetime(options['start_date'])
        end_date = pd.to_datetime(options['end_date'])
        output_file = options['output_file']

        self.stdout.write(f"🚀 Creating master simulation CSV...")
        self.stdout.write(f"📅 Date range: {start_date} to {end_date}")

        # Create 5-minute timestamps
        timestamps = pd.date_range(start=start_date, end=end_date, freq='5min')
        timestamps = timestamps.tz_localize(None)  # Remove timezone info
        self.stdout.write(f"⏰ Generated {len(timestamps)} timestamps")

        # Initialize master data dictionary
        master_data = {'timestamp': timestamps}

        # Load baseline OHLCV data
        self.stdout.write("📊 Loading baseline OHLCV data...")
        baseline_df = pd.read_csv('baseline_ohlcv.csv')
        baseline_df['timestamp'] = pd.to_datetime(baseline_df['timestamp']).dt.tz_localize(None)
        
        # Get list of coins
        coins = baseline_df['coin'].unique()
        self.stdout.write(f"🪙 Found {len(coins)} coins: {list(coins)}")

        # Add OHLCV data for each coin
        for coin in coins:
            self.stdout.write(f"📈 Processing {coin} OHLCV data...")
            coin_data = baseline_df[baseline_df['coin'] == coin].copy()
            coin_data = coin_data.set_index('timestamp')
            
            # Reindex to match master timestamps and forward fill
            for col in ['open', 'high', 'low', 'close', 'volume']:
                coin_reindexed = coin_data[col].reindex(timestamps, method='ffill')
                master_data[f'{coin}_{col}'] = coin_reindexed.values

        # Load prediction data for all models
        self.stdout.write("🤖 Loading prediction data...")
        
        # Long models
        long_models = {
            'ADAUSDT': 'ada_two_predictions.csv',
            'ATOMUSDT': 'atom_two_predictions.csv', 
            'AVAXUSDT': 'avax_two_predictions.csv',
            'BTCUSDT': 'btc_two_predictions.csv',
            'DOGEUSDT': 'doge_two_predictions.csv',
            'DOTUSDT': 'dot_two_predictions.csv',
            'ETHUSDT': 'eth_two_predictions.csv',
            'LINKUSDT': 'link_two_predictions.csv',
            'LTCUSDT': 'ltc_two_predictions.csv',
            'SOLUSDT': 'sol_two_predictions.csv',
            'UNIUSDT': 'uni_two_predictions.csv',
            'XLMUSDT': 'xlm_two_predictions.csv',
            'XRPUSDT': 'xrp_two_predictions.csv',
            'SHIBUSDT': 'shib_two_predictions.csv',
            'TRXUSDT': 'trx_two_predictions.csv'
        }

        # Short models
        short_models = {
            'ADAUSDT': 'ada_simple_short_predictions.csv',
            'ATOMUSDT': 'atom_simple_short_predictions.csv',
            'AVAXUSDT': 'avax_simple_short_predictions.csv', 
            'BTCUSDT': 'btc_simple_short_predictions.csv',
            'DOGEUSDT': 'doge_simple_short_predictions.csv',
            'DOTUSDT': 'dot_simple_short_predictions.csv',
            'ETHUSDT': 'eth_simple_short_predictions.csv',
            'LINKUSDT': 'link_simple_short_predictions.csv',
            'LTCUSDT': 'ltc_simple_short_predictions.csv',
            'SOLUSDT': 'sol_simple_short_predictions.csv',
            'UNIUSDT': 'uni_simple_short_predictions.csv',
            'XLMUSDT': 'xlm_simple_short_predictions.csv',
            'XRPUSDT': 'xrp_simple_short_predictions.csv',
            'SHIBUSDT': 'shib_simple_short_predictions.csv',
            'TRXUSDT': 'trx_simple_short_predictions.csv'
        }

        # Load long model predictions
        for coin, filename in long_models.items():
            if os.path.exists(filename):
                self.stdout.write(f"📊 Loading {coin} long predictions...")
                pred_df = pd.read_csv(filename)
                pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp']).dt.tz_localize(None)
                pred_df = pred_df.set_index('timestamp')
                
                # Reindex prediction data to match master timestamps and forward fill
                pred_reindexed = pred_df.reindex(timestamps, method='ffill')
                master_data[f'{coin}_long_confidence'] = pred_reindexed['pred_prob'].values
            else:
                self.stdout.write(f"⚠️ {filename} not found, skipping {coin} long predictions")
                master_data[f'{coin}_long_confidence'] = np.nan

        # Load short model predictions  
        for coin, filename in short_models.items():
            if os.path.exists(filename):
                self.stdout.write(f"📊 Loading {coin} short predictions...")
                pred_df = pd.read_csv(filename)
                pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp']).dt.tz_localize(None)
                pred_df = pred_df.set_index('timestamp')
                
                # Reindex prediction data to match master timestamps and forward fill
                pred_reindexed = pred_df.reindex(timestamps, method='ffill')
                master_data[f'{coin}_short_confidence'] = pred_reindexed['pred_prob'].values
            else:
                self.stdout.write(f"⚠️ {filename} not found, skipping {coin} short predictions")
                master_data[f'{coin}_short_confidence'] = np.nan

        # Load market regime data
        self.stdout.write("🌊 Loading market regime data...")
        if os.path.exists('market_regime_live_8h.csv'):
            regime_df = pd.read_csv('market_regime_live_8h.csv')
            regime_df['timestamp'] = pd.to_datetime(regime_df['timestamp']).dt.tz_localize(None)
            regime_df = regime_df.set_index('timestamp')
            
            # Reindex regime data to match master timestamps and forward fill
            regime_reindexed = regime_df.reindex(timestamps, method='ffill')
            
            # Map regime data to master timestamps
            master_data['market_regime'] = regime_reindexed['regime'].values
            master_data['bull_strength'] = regime_reindexed['bull_strength'].values
            master_data['bear_strength'] = regime_reindexed['bear_strength'].values
        else:
            self.stdout.write("⚠️ market_regime_live_4h.csv not found, skipping regime data")
            master_data['market_regime'] = 'neutral'
            master_data['bull_strength'] = 0.0
            master_data['bear_strength'] = 0.0

        # Add thresholds for each coin
        self.stdout.write("🎯 Adding confidence thresholds...")
        
        # Long thresholds
        long_thresholds = {
            'ADAUSDT': 0.55, 'ATOMUSDT': 0.5, 'AVAXUSDT': 0.5, 'BTCUSDT': 0.38,
            'DOGEUSDT': 0.5, 'DOTUSDT': 0.55, 'ETHUSDT': 0.4, 'LINKUSDT': 0.45,
            'LTCUSDT': 0.55, 'SOLUSDT': 0.5, 'UNIUSDT': 0.55, 'XLMUSDT': 0.5,
            'XRPUSDT': 0.55, 'SHIBUSDT': 0.55, 'TRXUSDT': 0.1
        }

        # Short thresholds
        short_thresholds = {
            'ADAUSDT': 0.55, 'ATOMUSDT': 0.5, 'AVAXUSDT': 0.5, 'BTCUSDT': 0.5,
            'DOGEUSDT': 0.5, 'DOTUSDT': 0.55, 'ETHUSDT': 0.5, 'LINKUSDT': 0.5,
            'LTCUSDT': 0.55, 'SOLUSDT': 0.5, 'UNIUSDT': 0.55, 'XLMUSDT': 0.5,
            'XRPUSDT': 0.55, 'SHIBUSDT': 0.55, 'TRXUSDT': 0.55
        }

        for coin in coins:
            master_data[f'{coin}_long_threshold'] = long_thresholds.get(coin, 0.5)
            master_data[f'{coin}_short_threshold'] = short_thresholds.get(coin, 0.5)

        # Create DataFrame from all collected data
        self.stdout.write("🔄 Creating master DataFrame...")
        master_df = pd.DataFrame(master_data)
        
        # Forward fill missing values (use previous candle's data if current is missing)
        self.stdout.write("🔄 Forward filling missing values...")
        master_df = master_df.fillna(method='ffill')

        # Save master CSV
        self.stdout.write(f"💾 Saving master CSV to {output_file}...")
        master_df.to_csv(output_file, index=False)

        # Summary
        self.stdout.write(f"✅ Master CSV created successfully!")
        self.stdout.write(f"📊 Total rows: {len(master_df)}")
        self.stdout.write(f"📅 Date range: {master_df['timestamp'].min()} to {master_df['timestamp'].max()}")
        self.stdout.write(f"🪙 Coins: {len(coins)}")
        self.stdout.write(f"📈 Columns: {len(master_df.columns)}")
        
        # Show sample
        self.stdout.write(f"\n📋 Sample data:")
        self.stdout.write(master_df.head().to_string())
