# scanner/management/commands/calculate_market_regime.py
# Pre-calculate market regime for each timestamp using all model confidence scores
# This creates a CSV that can be used by the trade simulator for fast regime lookup

from django.core.management.base import BaseCommand
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from datetime import timezone
import json

class Command(BaseCommand):
    help = 'Calculate market regime for each timestamp using all model confidence scores'

    def add_arguments(self, parser):
        parser.add_argument('--output-file', type=str, default='market_regime.csv', help='Output CSV file')
        parser.add_argument('--lookback-bars', type=int, default=12, help='Bars to look back for regime detection')
        parser.add_argument('--regime-threshold', type=float, default=0.1, help='Min difference to switch regimes')
        parser.add_argument('--min-regime-strength', type=float, default=0.6, help='Min regime strength to trade')
        parser.add_argument('--bias-correction', type=str, default='confidence_weight', 
                          choices=['none', 'confidence_weight', 'equal_weight'], 
                          help='How to handle bias from different numbers of long vs short models')

    def load_model_predictions(self):
        """Load all model prediction files"""
        predictions = {}
        
        # Long model files (hardcoded)
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
        
        # Short model files (hardcoded)
        short_models = {
            'ADAUSDT': 'ada_simple_short_predictions.csv',
            'ATOMUSDT': 'atom_simple_short_predictions.csv',
            'AVAXUSDT': 'avax_simple_short_predictions.csv',
            'DOGEUSDT': 'doge_simple_short_predictions.csv',
            'DOTUSDT': 'dot_simple_short_predictions.csv',
            'ETHUSDT': 'eth_simple_short_predictions.csv',
            'LINKUSDT': 'link_simple_short_predictions.csv',
            'LTCUSDT': 'ltc_simple_short_predictions.csv',
            'SOLUSDT': 'sol_simple_short_predictions.csv',
            'UNIUSDT': 'uni_simple_short_predictions.csv',
            'XRPUSDT': 'xrp_simple_short_predictions.csv',
            'SHIBUSDT': 'shib_simple_short_predictions.csv'
        }
        
        # Model optimal thresholds (for normalizing confidence scores)
        long_thresholds = {
            'ADAUSDT': 0.55, 'ATOMUSDT': 0.5, 'AVAXUSDT': 0.5, 'BTCUSDT': 0.38,
            'DOGEUSDT': 0.5, 'DOTUSDT': 0.55, 'ETHUSDT': 0.4, 'LINKUSDT': 0.45,
            'LTCUSDT': 0.55, 'SOLUSDT': 0.5, 'UNIUSDT': 0.55, 'XLMUSDT': 0.5,
            'XRPUSDT': 0.55, 'SHIBUSDT': 0.55, 'TRXUSDT': 0.1
        }
        
        short_thresholds = {
            'ADAUSDT': 0.55, 'ATOMUSDT': 0.5, 'AVAXUSDT': 0.5, 'DOGEUSDT': 0.5,
            'DOTUSDT': 0.55, 'ETHUSDT': 0.4, 'LINKUSDT': 0.45, 'LTCUSDT': 0.55,
            'SOLUSDT': 0.5, 'UNIUSDT': 0.55, 'XRPUSDT': 0.55, 'SHIBUSDT': 0.55
        }
        
        # Load long predictions
        for coin, file_path in long_models.items():
            if os.path.exists(file_path) and long_thresholds[coin] > 0:
                try:
                    df = pd.read_csv(file_path)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
                    predictions[f'{coin}_long'] = {
                        'data': df,
                        'threshold': long_thresholds[coin],
                        'optimal_threshold': long_thresholds[coin]
                    }
                    self.stdout.write(f"✅ Loaded long model for {coin} (threshold: {long_thresholds[coin]})")
                except Exception as e:
                    self.stdout.write(f"❌ Failed to load long model for {coin}: {e}")
        
        # Load short predictions
        for coin, file_path in short_models.items():
            if os.path.exists(file_path) and short_thresholds[coin] > 0:
                try:
                    df = pd.read_csv(file_path)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
                    predictions[f'{coin}_short'] = {
                        'data': df,
                        'threshold': short_thresholds[coin],
                        'optimal_threshold': short_thresholds[coin]
                    }
                    self.stdout.write(f"✅ Loaded short model for {coin} (threshold: {short_thresholds[coin]})")
                except Exception as e:
                    self.stdout.write(f"❌ Failed to load short model for {coin}: {e}")
        
        return predictions

    def calculate_regime_batch(self, predictions, all_timestamps, lookback_bars=12, bias_correction='confidence_weight'):
        """
        Calculate regime for all timestamps using vectorized pandas operations
        Much faster than individual timestamp calculations
        """
        # Create a master DataFrame with all timestamps
        master_df = pd.DataFrame({'timestamp': all_timestamps})
        master_df = master_df.sort_values('timestamp').reset_index(drop=True)
        
        # Add normalized scores for each model
        for key, model_data in predictions.items():
            df = model_data['data'].copy()
            optimal_threshold = model_data['optimal_threshold']
            
            # Normalize confidence scores
            if 'pred_prob' in df.columns:
                df['normalized_score'] = df['pred_prob'] / optimal_threshold
                
                # Merge with master DataFrame
                if key.endswith('_long'):
                    master_df = master_df.merge(
                        df[['timestamp', 'normalized_score']].rename(columns={'normalized_score': f'{key}_score'}),
                        on='timestamp', how='left'
                    )
                elif key.endswith('_short'):
                    master_df = master_df.merge(
                        df[['timestamp', 'normalized_score']].rename(columns={'normalized_score': f'{key}_score'}),
                        on='timestamp', how='left'
                    )
        
        # Calculate rolling averages for bull and bear scores
        long_cols = [col for col in master_df.columns if col.endswith('_long_score')]
        short_cols = [col for col in master_df.columns if col.endswith('_short_score')]
        
        # Calculate rolling mean of available scores for each timestamp
        master_df['bull_strength'] = master_df[long_cols].mean(axis=1, skipna=True)
        master_df['bear_strength'] = master_df[short_cols].mean(axis=1, skipna=True)
        
        # Fill NaN values with 0
        master_df['bull_strength'] = master_df['bull_strength'].fillna(0.0)
        master_df['bear_strength'] = master_df['bear_strength'].fillna(0.0)
        
        # Count available models for each timestamp
        master_df['bull_model_count'] = master_df[long_cols].count(axis=1)
        master_df['bear_model_count'] = master_df[short_cols].count(axis=1)
        
        # Apply bias correction based on method chosen
        if bias_correction == 'confidence_weight':
            # Weight by model availability (more models = higher confidence)
            master_df['bull_confidence_weight'] = master_df['bull_model_count'] / len(long_cols)
            master_df['bear_confidence_weight'] = master_df['bear_model_count'] / len(short_cols)
            master_df['bull_strength'] = master_df['bull_strength'] * master_df['bull_confidence_weight']
            master_df['bear_strength'] = master_df['bear_strength'] * master_df['bear_confidence_weight']
            
        elif bias_correction == 'equal_weight':
            # Normalize by total number of models to ensure equal weighting
            # This prevents bias from having different numbers of long vs short models
            total_long_models = len(long_cols)
            total_short_models = len(short_cols)
            master_df['bull_strength'] = master_df['bull_strength'] * (total_short_models / total_long_models)
            # Bear strength stays the same (used as baseline)
            
        # If bias_correction == 'none', use raw scores without adjustment
        
        # Apply rolling window for lookback
        master_df['bull_strength'] = master_df['bull_strength'].rolling(window=lookback_bars, min_periods=1).mean()
        master_df['bear_strength'] = master_df['bear_strength'].rolling(window=lookback_bars, min_periods=1).mean()
        
        # Calculate regime
        master_df['regime_diff'] = abs(master_df['bull_strength'] - master_df['bear_strength'])
        
        def determine_regime(row):
            if row['regime_diff'] < self.regime_threshold:
                return 'neutral'
            elif row['bull_strength'] > row['bear_strength']:
                return 'bull'
            else:
                return 'bear'
        
        master_df['regime'] = master_df.apply(determine_regime, axis=1)
        
        return master_df[['timestamp', 'regime', 'bull_strength', 'bear_strength', 'regime_diff']]

    def handle(self, *args, **opt):
        # Load configuration
        self.regime_threshold = float(opt['regime_threshold'])
        self.min_regime_strength = float(opt['min_regime_strength'])
        lookback_bars = int(opt['lookback_bars'])
        output_file = opt['output_file']
        bias_correction = opt['bias_correction']
        
        self.stdout.write("🚀 MARKET REGIME CALCULATOR")
        self.stdout.write("📊 Pre-calculating market regime for all timestamps")
        self.stdout.write(f"🔧 Bias correction method: {bias_correction}")
        
        # Load all model predictions
        self.stdout.write("▶ Loading model predictions...")
        predictions = self.load_model_predictions()
        
        if not predictions:
            self.stderr.write("❌ No model predictions loaded.")
            return
        
        self.stdout.write(f"✅ Loaded {len(predictions)} model predictions")
        
        # Get all unique timestamps from all models
        all_timestamps = set()
        for key, model_data in predictions.items():
            df = model_data['data']
            if not df.empty:
                timestamps = df['timestamp'].tolist()
                all_timestamps.update(timestamps)
                self.stdout.write(f"  {key}: {len(timestamps)} timestamps, range: {min(timestamps)} to {max(timestamps)}")
        
        all_timestamps = sorted(all_timestamps)
        self.stdout.write(f"▶ Processing {len(all_timestamps)} unique timestamps using vectorized operations...")
        self.stdout.write(f"📅 Date range: {min(all_timestamps)} to {max(all_timestamps)}")
        
        # Calculate regime for all timestamps at once (FAST!)
        regime_df = self.calculate_regime_batch(predictions, all_timestamps, lookback_bars, bias_correction)
        
        # Save results
        regime_df.to_csv(output_file, index=False)
        
        # Print statistics
        regime_counts = regime_df['regime'].value_counts()
        total_timestamps = len(regime_df)
        
        self.stdout.write(self.style.SUCCESS(f"\n✅ MARKET REGIME CALCULATION COMPLETE"))
        self.stdout.write(self.style.SUCCESS(f"📊 Output file: {output_file}"))
        self.stdout.write(self.style.SUCCESS(f"📈 Total timestamps: {total_timestamps}"))
        self.stdout.write(self.style.SUCCESS(f"🐂 Bull regime: {regime_counts.get('bull', 0)} ({regime_counts.get('bull', 0)/total_timestamps*100:.1f}%)"))
        self.stdout.write(self.style.SUCCESS(f"🐻 Bear regime: {regime_counts.get('bear', 0)} ({regime_counts.get('bear', 0)/total_timestamps*100:.1f}%)"))
        self.stdout.write(self.style.SUCCESS(f"⚖️ Neutral regime: {regime_counts.get('neutral', 0)} ({regime_counts.get('neutral', 0)/total_timestamps*100:.1f}%)"))
        
        # Show regime strength statistics
        bull_strength_avg = regime_df[regime_df['regime'] == 'bull']['bull_strength'].mean()
        bear_strength_avg = regime_df[regime_df['regime'] == 'bear']['bear_strength'].mean()
        
        self.stdout.write(self.style.SUCCESS(f"📊 Average bull strength: {bull_strength_avg:.3f}"))
        self.stdout.write(self.style.SUCCESS(f"📊 Average bear strength: {bear_strength_avg:.3f}"))
        
        # Show sample of regime data
        self.stdout.write(self.style.SUCCESS(f"\n📋 Sample regime data (first 10):"))
        sample_df = regime_df.head(10)
        for _, row in sample_df.iterrows():
            self.stdout.write(f"  {row['timestamp']}: {row['regime']} (Bull: {row['bull_strength']:.3f}, Bear: {row['bear_strength']:.3f})")
        
        self.stdout.write(self.style.SUCCESS(f"\n📋 Sample regime data (last 10):"))
        sample_df = regime_df.tail(10)
        for _, row in sample_df.iterrows():
            self.stdout.write(f"  {row['timestamp']}: {row['regime']} (Bull: {row['bull_strength']:.3f}, Bear: {row['bear_strength']:.3f})")
