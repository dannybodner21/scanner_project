# main_app/management/commands/run_pipeline.py

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


# one


warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def add_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced feature engineering with more predictive features"""
    df = df.copy()
    df.set_index('timestamp', inplace=True)

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_price = df['open']

    # === Price-based features ===
    # Multiple timeframe EMAs
    for period in [9, 21, 50, 100, 200]:
        df[f'ema_{period}'] = EMAIndicator(close, window=period).ema_indicator()
        df[f'sma_{period}'] = SMAIndicator(close, window=period).sma_indicator()

    # EMA relationships (key trend indicators)
    df['ema_9_21_ratio'] = df['ema_9'] / df['ema_21']
    df['ema_21_50_ratio'] = df['ema_21'] / df['ema_50']
    df['ema_50_200_ratio'] = df['ema_50'] / df['ema_200']
    df['price_above_ema_200'] = (close > df['ema_200']).astype(int)

    # === Momentum indicators ===
    # RSI with multiple periods
    for period in [14, 21]:
        rsi = RSIIndicator(close, window=period).rsi()
        df[f'rsi_{period}'] = rsi
        df[f'rsi_{period}_oversold'] = (rsi < 30).astype(int)
        df[f'rsi_{period}_overbought'] = (rsi > 70).astype(int)

    # MACD
    macd_indicator = MACD(close, window_slow=26, window_fast=36, window_sign=9)
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['macd_histogram'] = macd_indicator.macd_diff()
    df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)

    # Stochastic RSI (using the indicator you already had)
    stoch_rsi = StochRSIIndicator(close, window=14)
    df['stoch_k'] = stoch_rsi.stochrsi_k()
    df['stoch_d'] = stoch_rsi.stochrsi_d()
    df['stoch_oversold'] = (df['stoch_k'] < 0.2).astype(int)
    df['stoch_overbought'] = (df['stoch_k'] > 0.8).astype(int)

    # === Volume indicators ===
    df['obv'] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df['volume_sma_20'] = volume.rolling(window=20).mean()
    df['volume_ratio'] = volume / df['volume_sma_20']
    df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
    df['volume_trend'] = df['volume_sma_20'].pct_change(5)  # 5-period volume trend

    # === Volatility indicators ===
    df['atr_14'] = AverageTrueRange(high, low, close, window=14).average_true_range()
    df['atr_21'] = AverageTrueRange(high, low, close, window=21).average_true_range()

    # Bollinger Bands
    bb = BollingerBands(close, window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.8).astype(int)

    # === Returns (multiple timeframes) ===
    for periods in [1, 3, 6, 12, 24, 48]:  # 5min, 15min, 30min, 1h, 2h, 4h
        df[f'returns_{periods}p'] = close.pct_change(periods)
        df[f'returns_{periods}p_abs'] = np.abs(df[f'returns_{periods}p'])

    # === Price action patterns ===
    # OHLC relationships
    df['body_size'] = np.abs(close - open_price) / open_price
    df['upper_shadow'] = (high - np.maximum(close, open_price)) / open_price
    df['lower_shadow'] = (np.minimum(close, open_price) - low) / open_price
    df['is_green'] = (close > open_price).astype(int)

    # Price position indicators
    df['high_low_ratio'] = high / low
    df['close_position'] = (close - low) / (high - low)

    # === Market structure ===
    # Support/Resistance levels
    df['dist_from_high_24h'] = (close / high.rolling(288).max()) - 1  # 288 = 24h in 5min bars
    df['dist_from_low_24h'] = (close / low.rolling(288).min()) - 1
    df['dist_from_high_7d'] = (close / high.rolling(2016).max()) - 1   # 2016 = 7d in 5min bars
    df['dist_from_low_7d'] = (close / low.rolling(2016).min()) - 1

    # === Trend strength ===
    # Price slopes
    for window in [6, 12, 24]:  # 30min, 1h, 2h trends
        df[f'slope_{window}p'] = close.rolling(window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan,
            raw=False
        )

    # === Time-based features ===
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['is_us_hours'] = ((df.index.hour >= 13) & (df.index.hour <= 21)).astype(int)  # 9AM-5PM EST in UTC
    df['is_asia_hours'] = ((df.index.hour >= 0) & (df.index.hour <= 8)).astype(int)   # Asia trading hours

    # === Lag features (recent momentum) ===
    for lag in [1, 2, 3, 6]:
        df[f'close_lag_{lag}'] = close.shift(lag)
        df[f'volume_lag_{lag}'] = volume.shift(lag)
        df[f'rsi_14_lag_{lag}'] = df['rsi_14'].shift(lag)

    # === Cross-coin features (if we have multiple coins) ===
    # We'll add these in the main processing loop

    df.reset_index(inplace=True)
    return df

def get_direction_labels(df: pd.DataFrame, forward_periods: int = 36) -> pd.Series:
    """
    Simple direction prediction: will price be lower in N periods?
    This is much more learnable than complex TP/SL logic
    """
    current_close = df['close']
    future_close = df['close'].shift(-forward_periods)

    # 1 if price will be lower, 0 if higher
    goal_price = current_close * 0.985

    labels = (future_close < goal_price).astype(int)

    # Remove last N rows where we don't have future data
    labels.iloc[-forward_periods:] = np.nan

    return labels

def add_cross_coin_features(coin_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Add features that compare across coins (market regime indicators)"""

    # Calculate BTC dominance features if BTC is present
    btc_coins = ['BTCUSDT', 'BTCUSD']
    btc_df = None
    for btc_coin in btc_coins:
        if btc_coin in coin_dfs:
            btc_df = coin_dfs[btc_coin]
            break

    if btc_df is not None:
        # BTC trend features
        btc_trend = btc_df['ema_21'] > btc_df['ema_50']
        btc_strong_trend = btc_df['ema_9'] > btc_df['ema_21']

        for coin_name, coin_df in coin_dfs.items():
            if coin_name not in btc_coins:
                # Add BTC trend features to altcoins
                coin_df['btc_bull_trend'] = btc_trend.values[:len(coin_df)]
                coin_df['btc_strong_trend'] = btc_strong_trend.values[:len(coin_df)]

                # Correlation with BTC
                coin_df['btc_correlation'] = coin_df['returns_12p'].rolling(48).corr(
                    btc_df['returns_12p'][:len(coin_df)]
                )

    return coin_dfs

def select_best_features(X: pd.DataFrame, y: pd.Series, k: int = 50) -> List[str]:
    """Select the most predictive features"""
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)

    feature_scores = pd.DataFrame({
        'feature': X.columns,
        'score': selector.scores_
    }).sort_values('score', ascending=False)

    return feature_scores.head(k)['feature'].tolist()

class Command(BaseCommand):
    help = "Enhanced ML pipeline for crypto direction prediction"

    def add_arguments(self, parser):
        parser.add_argument('--skip-generation', action='store_true')
        parser.add_argument('--skip-tuning', action='store_true')
        parser.add_argument('--n-trials', type=int, default=5)
        parser.add_argument('--forward-periods', type=int, default=36)
        parser.add_argument('--min-samples', type=int, default=10000)

    def handle(self, *args, **options):
        # Updated coin list with more liquid pairs

        # COINS = ['LTCUSDT', 'XRPUSDT', 'DOTUSDT', 'LINKUSDT', 'UNIUSDT']

        COINS = ['BTCUSDT','ETHUSDT','XRPUSDT','LTCUSDT','SOLUSDT','DOGEUSDT','LINKUSDT','DOTUSDT', 'SHIBUSDT', 'ADAUSDT', 'UNIUSDT', 'AVAXUSDT', 'XLMUSDT']


        START_DATE = datetime(2022, 1, 1, tzinfo=timezone.utc)
        END_DATE = datetime(2025, 7, 14, tzinfo=timezone.utc)
        CUTOFF_DATE = datetime(2025, 5, 1, tzinfo=timezone.utc)

        FORWARD_PERIODS = options['forward_periods']
        MIN_SAMPLES = options['min_samples']

        TRAIN_FILE = 'short_one_training.csv'
        TEST_FILE = 'short_one_testing.csv'
        MODEL_FILE = 'short_one_model.joblib'
        SCALER_FILE = 'short_one_feature_scaler.joblib'
        FEATURES_FILE = 'short_one_selected_features.joblib'
        PREDICTION_FILE = 'short_one_enhanced_predictions.csv'

        if not options['skip_generation']:
            self.run_data_generation(COINS, START_DATE, END_DATE, CUTOFF_DATE,
                                   FORWARD_PERIODS, MIN_SAMPLES, TRAIN_FILE, TEST_FILE)

        self.stdout.write("💾 Loading data for training...")
        if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
            self.stdout.write(self.style.ERROR("Data files not found."))
            return

        train_df = pd.read_csv(TRAIN_FILE)

        # Remove non-feature columns
        non_feature_cols = ['timestamp', 'coin', 'open', 'high', 'low', 'close', 'volume', 'label']
        feature_cols = [col for col in train_df.columns if col not in non_feature_cols]

        X = train_df[feature_cols]
        y = train_df['label']

        # Handle missing values
        X = X.fillna(X.median())

        # Feature selection
        self.stdout.write("🔍 Selecting best features...")
        selected_features = select_best_features(X, y, k=min(50, len(feature_cols)))
        X_selected = X[selected_features]

        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_selected),
            columns=X_selected.columns,
            index=X_selected.index
        )

        # Save preprocessing objects
        joblib.dump(scaler, SCALER_FILE)
        joblib.dump(selected_features, FEATURES_FILE)

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        if not options['skip_tuning']:
            best_params = self.run_hyperparameter_tuning(
                options['n_trials'], X_scaled, y, tscv
            )
        else:
            best_params = {
                'learning_rate': 0.05,
                'num_leaves': 100,
                'max_depth': 8,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1
            }

        final_model = self.train_final_model(X_scaled, y, best_params)
        joblib.dump(final_model, MODEL_FILE)

        self.run_predictions(final_model, scaler, selected_features, TEST_FILE, PREDICTION_FILE)
        self.stdout.write(self.style.SUCCESS("\n🎉 Enhanced pipeline finished successfully!"))

    def run_data_generation(self, coins, start, end, cutoff, forward_periods, min_samples, train_path, test_path):
        self.stdout.write(self.style.SUCCESS("\n--- Step 1: Enhanced Data Generation ---"))

        coin_dfs = {}

        # First pass: load and process each coin
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

            # Skip if insufficient data
            if len(df) < min_samples:
                self.stdout.write(f"    ⚠️  Insufficient data for {coin}: {len(df)} samples")
                continue

            # Add features
            df_featured = add_enhanced_features(df)

            # Add direction labels
            df_featured['label'] = get_direction_labels(df_featured, forward_periods)

            coin_dfs[coin] = df_featured
            self.stdout.write(f"    ✅ {coin}: {len(df_featured)} samples processed")

        # Second pass: add cross-coin features
        if len(coin_dfs) > 1:
            self.stdout.write("  - Adding cross-coin features...")
            coin_dfs = add_cross_coin_features(coin_dfs)

        # Combine all coins
        all_dfs = []
        for coin, df in coin_dfs.items():
            df['coin'] = coin
            all_dfs.append(df)

        if not all_dfs:
            self.stdout.write(self.style.ERROR("No valid data found for any coin!"))
            return

        full_df = pd.concat(all_dfs, ignore_index=True)

        # Remove rows with NaN labels
        full_df = full_df.dropna(subset=['label'])

        # Split by time
        train_df = full_df[full_df['timestamp'] < cutoff].copy()
        test_df = full_df[full_df['timestamp'] >= cutoff].copy()

        self.stdout.write(f"  - Training samples: {len(train_df)}")
        self.stdout.write(f"  - Testing samples: {len(test_df)}")

        # Check class balance
        train_balance = train_df['label'].value_counts(normalize=True)
        self.stdout.write(f"  - Training class balance: {train_balance.to_dict()}")

        # Save datasets
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        self.stdout.write(self.style.SUCCESS("✅ Enhanced dataset generation complete."))

    def run_hyperparameter_tuning(self, n_trials, X, y, tscv):
        self.stdout.write(self.style.SUCCESS("\n--- Step 2: Enhanced Hyperparameter Tuning ---"))

        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'n_estimators': 1000,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 31, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample_freq': 1,
                'random_state': 42
            }

            # Time series cross-validation
            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    eval_metric='binary_logloss',
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )

                preds = model.predict_proba(X_val_fold)[:, 1]
                score = roc_auc_score(y_val_fold, preds)
                cv_scores.append(score)

            return np.mean(cv_scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.stdout.write(self.style.SUCCESS("✅ Enhanced tuning complete."))
        self.stdout.write(f"  - Best CV AUC: {study.best_value:.4f}")

        return study.best_params

    def train_final_model(self, X, y, params):
        self.stdout.write(self.style.SUCCESS("\n--- Step 3: Training Final Model ---"))

        params.update({
            'objective': 'binary',
            'metric': 'binary_logloss',
            'random_state': 42,
            'n_estimators': 1500  # More trees for final model
        })

        model = lgb.LGBMClassifier(**params)
        model.fit(X, y)

        # Print feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.stdout.write("📊 Top 10 most important features:")
        for _, row in feature_importance.head(10).iterrows():
            self.stdout.write(f"  - {row['feature']}: {row['importance']:.2f}")

        return model

    def run_predictions(self, model, scaler, selected_features, test_path, prediction_path):
        self.stdout.write(self.style.SUCCESS("\n--- Step 4: Enhanced Predictions ---"))

        test_df = pd.read_csv(test_path, parse_dates=['timestamp'])

        # Prepare features
        non_feature_cols = ['timestamp', 'coin', 'open', 'high', 'low', 'close', 'volume', 'label']
        available_features = [col for col in test_df.columns if col not in non_feature_cols]

        X_test = test_df[available_features].fillna(test_df[available_features].median())
        X_test_selected = X_test[selected_features]
        X_test_scaled = scaler.transform(X_test_selected)

        # Make predictions
        probabilities = model.predict_proba(X_test_scaled)[:, 1]
        predictions = (probabilities > 0.3).astype(int)

        # Calculate accuracy on test set
        test_accuracy = accuracy_score(test_df['label'], predictions)
        test_auc = roc_auc_score(test_df['label'], probabilities)

        self.stdout.write(f"📈 Test Set Performance:")
        self.stdout.write(f"  - Accuracy: {test_accuracy:.4f}")
        self.stdout.write(f"  - AUC: {test_auc:.4f}")

        # Save predictions
        output_df = test_df[['timestamp', 'coin', 'open', 'high', 'low', 'close', 'label']].copy()
        output_df['prediction'] = predictions
        output_df['prediction_prob'] = probabilities
        output_df['correct'] = (output_df['prediction'] == output_df['label']).astype(int)

        output_df.to_csv(prediction_path, index=False)
        self.stdout.write(f"✅ Enhanced predictions saved to {prediction_path}")

        # Per-coin performance
        self.stdout.write("📊 Per-coin performance:")
        for coin in output_df['coin'].unique():
            coin_data = output_df[output_df['coin'] == coin]
            coin_acc = coin_data['correct'].mean()
            coin_auc = roc_auc_score(coin_data['label'], coin_data['prediction_prob'])
            self.stdout.write(f"  - {coin}: Accuracy={coin_acc:.4f}, AUC={coin_auc:.4f}")
