#!/usr/bin/env python3
"""
IMPROVED SHORT MODEL TRAINING
Addresses the fundamental issues with SHORT trade prediction:
1. Better labeling strategy for SHORT trades
2. SHORT-specific feature engineering
3. Improved class balancing and model training
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import optuna
from datetime import datetime, timezone
import warnings
import os
from typing import Dict, List, Tuple

from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from pandas.errors import SettingWithCopyWarning

warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def add_short_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    SHORT-SPECIFIC feature engineering
    Focus on bearish momentum, distribution patterns, and overbought conditions
    """
    df = df.copy()
    df.set_index('timestamp', inplace=True)

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_price = df['open']

    # === CORE TECHNICAL INDICATORS ===
    # EMAs optimized for SHORT detection
    for period in [9, 21, 50, 100, 200]:
        df[f'ema_{period}'] = EMAIndicator(close, window=period).ema_indicator()
        df[f'sma_{period}'] = SMAIndicator(close, window=period).sma_indicator()

    # EMA relationships for bearish trends
    df['ema_9_21_ratio'] = df['ema_9'] / df['ema_21']
    df['ema_21_50_ratio'] = df['ema_21'] / df['ema_50']
    df['ema_50_200_ratio'] = df['ema_50'] / df['ema_200']
    df['price_below_ema_21'] = (close < df['ema_21']).astype(int)
    df['price_below_ema_50'] = (close < df['ema_50']).astype(int)

    # RSI indicators
    for period in [14, 21]:
        rsi = RSIIndicator(close, window=period).rsi()
        df[f'rsi_{period}'] = rsi
        df[f'rsi_{period}_overbought'] = (rsi > 70).astype(int)
        df[f'rsi_{period}_extreme_ob'] = (rsi > 80).astype(int)

    # MACD for momentum
    macd_indicator = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['macd_histogram'] = macd_indicator.macd_diff()
    df['macd_bearish'] = (df['macd'] < df['macd_signal']).astype(int)

    # Volume analysis
    df['volume_sma_20'] = volume.rolling(window=20).mean()
    df['volume_ratio'] = volume / df['volume_sma_20']
    df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(int)

    # Volatility
    df['atr_14'] = AverageTrueRange(high, low, close, window=14).average_true_range()
    df['atr_percent'] = (df['atr_14'] / close) * 100

    # Bollinger Bands for overbought detection
    bb = BollingerBands(close, window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_squeeze'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle'] < 0.1).astype(int)

    # === SHORT-SPECIFIC PATTERNS ===
    
    # Price action patterns
    df['body_size'] = abs(close - open_price) / open_price
    df['upper_shadow'] = (high - np.maximum(close, open_price)) / open_price
    df['lower_shadow'] = (np.minimum(close, open_price) - low) / open_price
    
    # Distribution candles (bearish reversal patterns)
    df['distribution_candle'] = (
        (close < open_price) &  # Red candle
        (df['upper_shadow'] > df['body_size'] * 1.5) &  # Long upper shadow
        (df['volume_ratio'] > 1.2)  # Above average volume
    ).astype(int)
    
    # Exhaustion patterns
    df['buying_exhaustion'] = (
        (df['rsi_14'] > 75) &
        (df['bb_position'] > 0.8) &
        (df['volume_ratio'] > 1.5)
    ).astype(int)
    
    # Bearish divergence signals
    df['rsi_divergence'] = (
        (close > close.shift(5)) &  # Price higher
        (df['rsi_14'] < df['rsi_14'].shift(5))  # RSI lower
    ).astype(int)
    
    # Mean reversion setups (from overbought)
    df['mean_reversion_short'] = (
        (df['bb_position'] > 0.85) &
        (df['rsi_14'] > 70) &
        (df['volume_ratio'] > 1.3)
    ).astype(int)
    
    # Trend weakness indicators
    df['trend_weakness'] = (
        (df['ema_9_21_ratio'] < 1.0) &  # Short EMA below long
        (df['macd_histogram'] < 0) &    # MACD declining
        (df['rsi_14'] < df['rsi_14'].shift(3))  # RSI declining
    ).astype(int)
    
    # Multi-timeframe bearish alignment
    df['bearish_1h'] = (close < close.shift(12)).astype(int)  # 1h bearish
    df['bearish_4h'] = (close < close.shift(48)).astype(int)  # 4h bearish
    df['bearish_alignment'] = df['bearish_1h'] + df['bearish_4h']
    
    # Market structure breaks
    df['lower_high'] = (
        (high < high.shift(5)) &  # Lower high
        (df['rsi_14'] > 60)       # Still overbought
    ).astype(int)
    
    # Volume-price divergence
    df['volume_price_divergence'] = (
        (close > close.shift(5)) &  # Price up
        (volume < volume.shift(5))  # Volume down
    ).astype(int)
    
    # Momentum deceleration
    df['momentum_decel'] = (
        (df['rsi_14'] < df['rsi_14'].shift(1)) &  # RSI declining
        (df['rsi_14'].shift(1) < df['rsi_14'].shift(2)) &  # Previous also declining
        (df['rsi_14'] > 50)  # Still above neutral
    ).astype(int)

    df.reset_index(inplace=True)
    return df


def get_improved_short_labels(df: pd.DataFrame, forward_periods: int = 24) -> pd.Series:
    """
    IMPROVED labeling strategy for SHORT trades
    Uses asymmetric risk/reward optimized for SHORT success
    """
    labels = pd.Series(index=df.index, dtype="float64")

    # IMPROVED parameters based on market analysis
    tp_pct = 0.015   # 1.5% take profit (more realistic for shorts)
    sl_pct = 0.008   # 0.8% stop loss (tight risk control)
    
    # This creates a 1.875:1 reward-to-risk ratio, better for shorts
    
    for i in range(len(df) - forward_periods):
        entry_price = df.loc[df.index[i], 'close']
        tp_price = entry_price * (1 - tp_pct)  # SHORT: profit on price drop
        sl_price = entry_price * (1 + sl_pct)  # SHORT: loss on price rise

        future_highs = df['high'].iloc[i+1:i+1+forward_periods].values
        future_lows = df['low'].iloc[i+1:i+1+forward_periods].values

        hit_tp = False
        hit_sl = False

        # Check each future candle
        for j, (high, low) in enumerate(zip(future_highs, future_lows)):
            # For SHORT trades, check SL first (price rising)
            if high >= sl_price:
                hit_sl = True
                break
            # Then check TP (price falling)
            if low <= tp_price:
                hit_tp = True
                break
            
        if hit_tp:
            labels.iloc[i] = 1  # WIN: TP hit first
        else:
            labels.iloc[i] = 0  # LOSS: SL hit or no clear direction

    # Mark incomplete future data as NaN
    labels.iloc[-forward_periods:] = np.nan
    return labels


class Command(BaseCommand):
    help = "Train improved SHORT model with optimized features and labeling"

    def add_arguments(self, parser):
        parser.add_argument('--coins', type=str, default='BTCUSDT,ETHUSDT,SOLUSDT,ADAUSDT,XRPUSDT', 
                          help='Comma-separated list of coins')
        parser.add_argument('--start', type=str, default='2025-05-01', help='Start date (YYYY-MM-DD)')
        parser.add_argument('--end', type=str, default='2025-07-25', help='End date (YYYY-MM-DD)')
        parser.add_argument('--n_trials', type=int, default=100, help='Optuna trials')
        parser.add_argument('--skip_tuning', action='store_true', help='Skip hyperparameter tuning')

    def handle(self, *args, **options):
        coins = options['coins'].split(',')
        start = datetime.strptime(options['start'], '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end = datetime.strptime(options['end'], '%Y-%m-%d').replace(tzinfo=timezone.utc)
        cutoff = datetime(2025, 7, 1, tzinfo=timezone.utc)  # Train before July, test after
        
        forward_periods = 24  # 2 hours for prediction window
        min_samples = 1000
        
        # File paths
        TRAIN_FILE = "improved_short_training.csv"
        TEST_FILE = "improved_short_testing.csv"
        MODEL_FILE = "improved_short_model.joblib"
        SCALER_FILE = "improved_short_scaler.joblib"
        FEATURES_FILE = "improved_short_features.joblib"
        PREDICTION_FILE = "improved_short_predictions.csv"

        self.stdout.write(self.style.SUCCESS("🚀 IMPROVED SHORT MODEL TRAINING"))
        self.stdout.write(f"📊 Coins: {coins}")
        self.stdout.write(f"📅 Date range: {start.date()} to {end.date()}")
        self.stdout.write(f"📊 Forward periods: {forward_periods}")

        # Step 1: Generate training data
        self.run_data_generation(coins, start, end, cutoff, forward_periods, min_samples, TRAIN_FILE, TEST_FILE)
        
        # Step 2: Train model
        self.run_model_training(TRAIN_FILE, MODEL_FILE, SCALER_FILE, FEATURES_FILE, options['n_trials'], options['skip_tuning'])
        
        # Step 3: Generate predictions
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        selected_features = joblib.load(FEATURES_FILE)
        
        self.run_predictions(model, scaler, selected_features, TEST_FILE, PREDICTION_FILE)
        self.stdout.write(self.style.SUCCESS("\n🎉 Improved SHORT model training complete!"))

    def run_data_generation(self, coins, start, end, cutoff, forward_periods, min_samples, train_path, test_path):
        self.stdout.write(self.style.SUCCESS("\n--- Step 1: Improved Data Generation ---"))

        coin_dfs = {}

        # Process each coin with SHORT-specific features
        for coin in coins:
            self.stdout.write(f"  - Processing {coin}...")
            qs = CoinAPIPrice.objects.filter(
                coin=coin,
                timestamp__gte=start,
                timestamp__lte=end
            ).order_by("timestamp")

            if not qs.exists():
                self.stdout.write(f"    ⚠️  No data found for {coin}")
                continue

            df = pd.DataFrame.from_records(qs.values())

            # Data cleaning
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df.dropna(inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            if len(df) < min_samples:
                self.stdout.write(f"    ⚠️  Insufficient data for {coin}: {len(df)} samples")
                continue

            # Add SHORT-specific features
            df_featured = add_short_specific_features(df)

            # Add improved SHORT labels
            df_featured['label'] = get_improved_short_labels(df_featured, forward_periods)

            coin_dfs[coin] = df_featured
            self.stdout.write(f"    ✅ {coin}: {len(df_featured)} samples processed")

        # Combine all coins
        all_dfs = []
        for coin, df in coin_dfs.items():
            df['coin'] = coin
            all_dfs.append(df)

        if not all_dfs:
            self.stdout.write(self.style.ERROR("No valid data found for any coin!"))
            return

        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df = full_df.dropna(subset=['label'])

        # Split by time
        train_df = full_df[full_df['timestamp'] < cutoff].copy()
        test_df = full_df[full_df['timestamp'] >= cutoff].copy()

        self.stdout.write(f"  - Training samples: {len(train_df)}")
        self.stdout.write(f"  - Testing samples: {len(test_df)}")

        # Analyze class distribution
        train_balance = train_df['label'].value_counts()
        win_rate = train_balance.get(1, 0) / len(train_df) * 100
        self.stdout.write(f"  - Training win rate: {win_rate:.2f}%")
        self.stdout.write(f"  - Class distribution: {train_balance.to_dict()}")

        # Improved balancing strategy
        if win_rate < 20 or win_rate > 80:
            self.stdout.write("  - Applying balanced sampling...")
            min_class_count = min(train_balance.values())
            max_samples_per_class = min(min_class_count * 2, 50000)  # Cap at 50k per class
            
            df_0 = train_df[train_df['label'] == 0].sample(n=min(len(train_df[train_df['label'] == 0]), max_samples_per_class), random_state=42)
            df_1 = train_df[train_df['label'] == 1].sample(n=min(len(train_df[train_df['label'] == 1]), max_samples_per_class), random_state=42)
            
            train_df = pd.concat([df_0, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)
            self.stdout.write(f"  - Balanced training samples: {len(train_df)}")

        # Save datasets
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        self.stdout.write(f"  - Saved training data: {train_path}")
        self.stdout.write(f"  - Saved testing data: {test_path}")

    def run_model_training(self, train_path, model_path, scaler_path, features_path, n_trials, skip_tuning):
        self.stdout.write(self.style.SUCCESS("\n--- Step 2: Improved Model Training ---"))
        
        train_df = pd.read_csv(train_path, parse_dates=['timestamp'])
        
        # Prepare features (exclude non-feature columns)
        exclude_cols = ['timestamp', 'coin', 'label', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        X = train_df[feature_cols]
        y = train_df['label']
        
        self.stdout.write(f"  - Features: {len(feature_cols)}")
        self.stdout.write(f"  - Samples: {len(X)}")
        self.stdout.write(f"  - Win rate: {y.mean()*100:.2f}%")

        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=min(50, len(feature_cols)))
        X_selected = selector.fit_transform(X, y)
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        
        self.stdout.write(f"  - Selected features: {len(selected_features)}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Hyperparameter tuning or use defaults
        if not skip_tuning:
            self.stdout.write(f"  - Running hyperparameter tuning ({n_trials} trials)...")
            best_params = self.run_hyperparameter_tuning(n_trials, X_scaled, y, tscv)
        else:
            best_params = {
                'learning_rate': 0.05,
                'num_leaves': 80,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'min_child_samples': 100
            }
        
        self.stdout.write(f"  - Best parameters: {best_params}")
        
        # Train final model
        model = lgb.LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            random_state=42,
            n_estimators=1000,
            early_stopping_rounds=100,
            **best_params
        )
        
        # Use part of data for validation
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(0)]
        )
        
        # Save model components
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(selected_features, features_path)
        
        # Evaluate model
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        accuracy = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        self.stdout.write(f"  - Validation accuracy: {accuracy:.4f}")
        self.stdout.write(f"  - Validation AUC: {auc:.4f}")
        self.stdout.write(f"  - Model saved: {model_path}")

    def run_hyperparameter_tuning(self, n_trials, X, y, tscv):
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.5),
                'min_child_samples': trial.suggest_int('min_child_samples', 50, 200)
            }
            
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = lgb.LGBMClassifier(
                    objective='binary',
                    boosting_type='gbdt',
                    random_state=42,
                    n_estimators=100,
                    **params
                )
                
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(0)])
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred_proba)
                scores.append(auc)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def run_predictions(self, model, scaler, selected_features, test_path, output_path):
        self.stdout.write(self.style.SUCCESS("\n--- Step 3: Generating Predictions ---"))
        
        test_df = pd.read_csv(test_path, parse_dates=['timestamp'])
        
        # Prepare features
        X_test = test_df[selected_features]
        X_test_scaled = scaler.transform(X_test)
        
        # Generate predictions
        predictions = model.predict_proba(X_test_scaled)[:, 1]
        
        # Create output dataframe
        output_df = test_df[['timestamp', 'coin', 'open', 'high', 'low', 'close', 'volume']].copy()
        output_df['prediction'] = (predictions > 0.5).astype(int)
        output_df['prediction_prob'] = predictions
        
        # Add important features for analysis
        for feature in ['rsi_14', 'volume_ratio', 'bb_position', 'ema_9_21_ratio', 'distribution_candle', 'buying_exhaustion']:
            if feature in test_df.columns:
                output_df[feature] = test_df[feature]
        
        output_df.to_csv(output_path, index=False)
        self.stdout.write(f"  - Predictions saved: {output_path}")
        self.stdout.write(f"  - Total predictions: {len(output_df)}")
        self.stdout.write(f"  - Positive predictions: {output_df['prediction'].sum()}")
        self.stdout.write(f"  - Average confidence: {output_df['prediction_prob'].mean():.4f}")