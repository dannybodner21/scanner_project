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
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from ta.trend import EMAIndicator, MACD, SMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from pandas.errors import SettingWithCopyWarning


warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def add_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Greatly enhanced feature engineering with TA, market structure, and regime features.
    """
    df = df.copy()
    df.set_index('timestamp', inplace=True)

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_price = df['open']

    # === Volatility & Price Action ===
    df['atr_14'] = AverageTrueRange(high, low, close, window=14).average_true_range()
    df['atr_volatility'] = df['atr_14'] / close  # Normalized ATR
    df['atr_volatility_std_14'] = df['atr_volatility'].rolling(14).std() # Volatility of volatility

    bb = BollingerBands(close, window=20, window_dev=2)
    df['bb_width'] = bb.bollinger_wband()
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).quantile(0.1)).astype(int)

    # === Trend & Momentum (Enhanced) ===
    # Multiple EMAs
    for period in [9, 21, 50, 100, 200]:
        df[f'ema_{period}'] = EMAIndicator(close, window=period).ema_indicator()

    df['ema_9_21_ratio'] = df['ema_9'] / df['ema_21']
    df['ema_50_200_ratio'] = df['ema_50'] / df['ema_200']
    df['price_vs_ema200'] = (close - df['ema_200']) / df['ema_200'] # Distance from key EMA

    # RSI
    df['rsi_14'] = RSIIndicator(close, window=14).rsi()
    df['rsi_divergence'] = df['rsi_14'].diff(3) - close.diff(3) # Simple divergence proxy

    # MACD
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df['macd_diff'] = macd.macd_diff()
    df['macd_trending'] = (df['macd_diff'] * df['macd_diff'].shift(1) > 0).astype(int) # Is histogram consistent



    # === Ichimoku Cloud (Powerful all-in-one indicator) ===
    ichimoku = IchimokuIndicator(high=high, low=low, window1=9, window2=26, window3=52)
    df['tenkan_sen'] = ichimoku.ichimoku_conversion_line() # Corrected
    df['kijun_sen'] = ichimoku.ichimoku_base_line()      # Corrected
    df['senkou_a'] = ichimoku.ichimoku_a()
    df['senkou_b'] = ichimoku.ichimoku_b()



    
    # Is price in the cloud? (ranging/indecisive)
    df['in_cloud'] = ((close > df['senkou_a']) & (close < df['senkou_b'])).astype(int)
    # Is the cloud bullish? (Senkou A > Senkou B)
    df['cloud_is_bullish'] = (df['senkou_a'] > df['senkou_b']).astype(int)
    # Price distance from Kijun-sen (key trendline)
    df['price_vs_kijun'] = (close - df['kijun_sen']) / df['kijun_sen']

    # === Volume & Market Structure ===
    df['obv'] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df['volume_sma_30'] = volume.rolling(window=30).mean()
    df['volume_spike'] = (volume > df['volume_sma_30'] * 2).astype(int)

    # Price distance from recent highs/lows
    df['dist_from_high_24h'] = (close / high.rolling(288).max()) - 1  # 288 = 24h in 5min bars
    df['dist_from_low_24h'] = (close / low.rolling(288).min()) - 1

    # === Time-based & Lag Features ===
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_us_hours'] = ((df.index.hour >= 13) & (df.index.hour <= 21)).astype(int)

    for p in [1, 3, 6, 12, 24]: # Returns over multiple periods
        df[f'returns_{p}p'] = close.pct_change(p)

    df.reset_index(inplace=True)
    return df


def get_direction_labels(df: pd.DataFrame, forward_periods: int = 18, tp_atr_mult: float = 2.0, sl_atr_mult: float = 1.5) -> pd.Series:
    """
    Dynamic labeling for short trades based on ATR.
    A trade is a "win" (1) if the price drops by (tp_atr_mult * ATR) before it
    rises by (sl_atr_mult * ATR) within the next `forward_periods`.
    """
    labels = pd.Series(index=df.index, dtype="float64")
    atr_series = df['atr_14']

    for i in range(len(df) - forward_periods):
        entry_price = df.loc[df.index[i], 'close']
        atr_at_entry = atr_series.iloc[i]

        if atr_at_entry == 0:  # Skip if ATR is zero
            labels.iloc[i] = np.nan
            continue

        tp_price = entry_price - (atr_at_entry * tp_atr_mult)  # Take Profit price
        sl_price = entry_price + (atr_at_entry * sl_atr_mult)  # Stop Loss price

        future_highs = df['high'].iloc[i+1 : i+1+forward_periods]
        future_lows = df['low'].iloc[i+1 : i+1+forward_periods]

        # Find the index of the first time SL or TP is hit
        sl_hit_time = future_highs[future_highs >= sl_price].first_valid_index()
        tp_hit_time = future_lows[future_lows <= tp_price].first_valid_index()

        if tp_hit_time is not None and (sl_hit_time is None or tp_hit_time < sl_hit_time):
            labels.iloc[i] = 1  # Win: TP was hit first
        else:
            labels.iloc[i] = 0  # Loss: SL was hit first or neither was hit

    labels.iloc[-forward_periods:] = np.nan
    return labels


def add_cross_coin_features(coin_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Adds market-wide features (regime filters)."""
    # Create a market-wide index based on average returns
    all_returns = []
    for coin_name, coin_df in coin_dfs.items():
        # Ensure consistent indexing by timestamp before processing
        indexed_df = coin_df.set_index('timestamp')
        all_returns.append(indexed_df['returns_12p'].rename(coin_name))

    if not all_returns:
        return coin_dfs
    
    returns_panel = pd.concat(all_returns, axis=1)
    market_avg_return = returns_panel.mean(axis=1)
    
    # Add market features to each coin's DataFrame
    for coin_name, coin_df in coin_dfs.items():
        # Merge market data; use a temporary index for alignment
        temp_df = coin_df.set_index('timestamp')
        temp_df['market_avg_return_12p'] = market_avg_return
        
        # Calculate correlation with the market average
        temp_df['market_correlation'] = temp_df['returns_12p'].rolling(48).corr(temp_df['market_avg_return_12p'])
        coin_dfs[coin_name] = temp_df.reset_index()

    return coin_dfs


def select_best_features(X: pd.DataFrame, y: pd.Series, k: int = 75) -> List[str]:
    """Selects the k most predictive features using univariate selection."""
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    return X.columns[selector.get_support()].tolist()

def run_trade_simulation(predictions_df: pd.DataFrame, test_data_df: pd.DataFrame, prob_threshold: float, sl_atr_mult: float, tp_atr_mult: float):
    """Simulates trading based on model predictions and evaluates profitability."""
    trades = []
    initial_capital = 10000
    capital = initial_capital
    position_size_usd = 500 # Fixed position size per trade
    
    # Merge predictions with the full test dataset to access future price data and ATR
    sim_df = pd.merge(predictions_df, test_data_df[['timestamp', 'coin', 'atr_14']], on=['timestamp', 'coin'], how='left')

    for index, row in sim_df.iterrows():
        if row['prediction_prob'] > prob_threshold:
            entry_price = row['close']
            atr_at_entry = row['atr_14']
            
            if pd.isna(atr_at_entry) or atr_at_entry == 0:
                continue

            sl_price = entry_price + (atr_at_entry * sl_atr_mult)
            tp_price = entry_price - (atr_at_entry * tp_atr_mult)

            # Look ahead in the original test data to see outcome
            future_df = test_data_df[(test_data_df['coin'] == row['coin']) & (test_data_df['timestamp'] > row['timestamp'])].head(18)
            
            pnl = 0
            outcome = 'Timed Out'
            
            for _, future_row in future_df.iterrows():
                if future_row['high'] >= sl_price:
                    pnl = - (sl_price - entry_price) / entry_price
                    outcome = 'Stop Loss'
                    break
                elif future_row['low'] <= tp_price:
                    pnl = (entry_price - tp_price) / entry_price
                    outcome = 'Take Profit'
                    break
            
            capital += pnl * position_size_usd
            trades.append({'coin': row['coin'], 'pnl_pct': pnl, 'outcome': outcome, 'capital': capital})

    if not trades:
        print("No trades were executed with the current threshold.")
        return

    trade_log = pd.DataFrame(trades)
    
    # --- Performance Metrics ---
    win_rate = (trade_log['pnl_pct'] > 0).mean() * 100
    loss_rate = (trade_log['pnl_pct'] < 0).mean() * 100
    
    gains = trade_log[trade_log['pnl_pct'] > 0]['pnl_pct'].sum()
    losses = abs(trade_log[trade_log['pnl_pct'] < 0]['pnl_pct'].sum())
    profit_factor = gains / losses if losses > 0 else float('inf')
    
    total_pnl = trade_log['capital'].iloc[-1] - initial_capital
    
    # Max Drawdown
    trade_log['cumulative_max'] = trade_log['capital'].cummax()
    trade_log['drawdown'] = (trade_log['capital'] - trade_log['cumulative_max']) / trade_log['cumulative_max']
    max_drawdown = trade_log['drawdown'].min() * 100

    # Sharpe Ratio (simplified)
    daily_returns = trade_log.groupby(pd.to_datetime(trade_log.index).date)['pnl_pct'].sum()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365) if daily_returns.std() > 0 else 0


    print(f"\n--- 📈 Trade Simulation Results (Threshold: {prob_threshold}) ---")
    print(f"Total Trades: {len(trade_log)}")
    print(f"Total PnL: ${total_pnl:.2f} ({total_pnl/initial_capital:.2%})")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Sharpe Ratio (annualized): {sharpe_ratio:.2f}")
    print("--------------------------------------------------")


class Command(BaseCommand):
    help = "Enhanced ML pipeline for crypto direction prediction"

    def add_arguments(self, parser):
        parser.add_argument('--skip-generation', action='store_true', help="Skip data generation step")
        parser.add_argument('--skip-tuning', action='store_true', help="Skip hyperparameter tuning")
        parser.add_argument('--n-trials', type=int, default=50, help="Number of Optuna trials")
        parser.add_argument('--forward-periods', type=int, default=18, help="How many 5-min candles to look ahead for labeling (90 mins)")
        parser.add_argument('--min-samples', type=int, default=5000)

    def handle(self, *args, **options):
        COINS = ['BTCUSDT','ETHUSDT','XRPUSDT','LTCUSDT','SOLUSDT','DOGEUSDT', 'LINKUSDT', 'AVAXUSDT']
        START_DATE = datetime(2022, 1, 1, tzinfo=timezone.utc)
        END_DATE = datetime(2025, 7, 23, tzinfo=timezone.utc)
        CUTOFF_DATE = datetime(2025, 6, 1, tzinfo=timezone.utc)

        FORWARD_PERIODS = options['forward_periods']
        MIN_SAMPLES = options['min_samples']
        TP_ATR_MULT = 2.0
        SL_ATR_MULT = 1.5

        # File Paths
        base_dir = "short_model_artifacts"
        os.makedirs(base_dir, exist_ok=True)
        TRAIN_FILE = os.path.join(base_dir, 'short_one_training.csv')
        TEST_FILE = os.path.join(base_dir, 'short_one_testing.csv')
        MODEL_FILE = os.path.join(base_dir, 'short_one_model.joblib')
        SCALER_FILE = os.path.join(base_dir, 'short_one_feature_scaler.joblib')
        FEATURES_FILE = os.path.join(base_dir, 'short_one_selected_features.joblib')
        PREDICTION_FILE = os.path.join(base_dir, 'short_one_predictions.csv')

        if not options['skip_generation']:
            self.run_data_generation(COINS, START_DATE, END_DATE, CUTOFF_DATE,
                                   FORWARD_PERIODS, MIN_SAMPLES, TRAIN_FILE, TEST_FILE, TP_ATR_MULT, SL_ATR_MULT)

        self.stdout.write("💾 Loading data for training...")
        if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
            self.stdout.write(self.style.ERROR("Data files not found. Run without --skip-generation first."))
            return

        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
        
        # Drop rows where label could not be computed
        train_df.dropna(subset=['label'], inplace=True)
        test_df.dropna(subset=['label'], inplace=True)
        train_df['label'] = train_df['label'].astype(int)
        test_df['label'] = test_df['label'].astype(int)

        non_feature_cols = ['timestamp', 'coin', 'open', 'high', 'low', 'close', 'volume', 'label']
        feature_cols = [col for col in train_df.columns if col not in non_feature_cols]

        X_train = train_df[feature_cols].copy()
        y_train = train_df['label'].copy()
        
        # Median imputation for robustness
        imputation_values = X_train.median()
        X_train.fillna(imputation_values, inplace=True)
        
        # --- Feature Selection ---
        self.stdout.write("🔍 Selecting best features...")
        selected_features = select_best_features(X_train, y_train, k=75)
        joblib.dump(selected_features, FEATURES_FILE)
        self.stdout.write(f"  - Selected {len(selected_features)} features.")
        
        X_train_selected = X_train[selected_features]
        
        # --- Scaling ---
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_selected), columns=selected_features)
        joblib.dump(scaler, SCALER_FILE)

        # --- Hyperparameter Tuning ---
        if not options['skip_tuning']:
            best_params = self.run_hyperparameter_tuning(options['n_trials'], X_train_scaled, y_train)
        else:
            self.stdout.write("⏩ Skipping tuning, using default parameters.")
            best_params = {'learning_rate': 0.02, 'num_leaves': 80, 'max_depth': 7, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1}

        # --- Model Training ---
        final_model = self.train_final_model(X_train_scaled, y_train, best_params)
        joblib.dump(final_model, MODEL_FILE)
        
        # --- Prediction & Evaluation ---
        self.run_predictions(final_model, scaler, selected_features, imputation_values, TEST_FILE, PREDICTION_FILE)
        
        # --- Profitability Simulation ---
        predictions_df = pd.read_csv(PREDICTION_FILE)
        # Use a higher probability threshold to find higher quality signals
        for threshold in [0.55, 0.60, 0.65, 0.70]:
            run_trade_simulation(predictions_df, test_df, threshold, SL_ATR_MULT, TP_ATR_MULT)

        self.stdout.write(self.style.SUCCESS("\n🎉 Enhanced pipeline finished successfully!"))

    def run_data_generation(self, coins, start, end, cutoff, forward_periods, min_samples, train_path, test_path, tp_mult, sl_mult):
        self.stdout.write(self.style.SUCCESS("\n--- Step 1: Enhanced Data Generation ---"))
        coin_dfs = {}
        for coin in coins:
            self.stdout.write(f"  - Processing {coin}...")
            qs = CoinAPIPrice.objects.filter(coin=coin, timestamp__gte=start, timestamp__lte=end).order_by("timestamp")
            if not qs.exists() or qs.count() < min_samples:
                self.stdout.write(f"    ⚠️  Insufficient data for {coin}")
                continue

            df = pd.DataFrame.from_records(qs.values('timestamp', 'open', 'high', 'low', 'close', 'volume'))
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce')
            df.dropna(inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df_featured = add_enhanced_features(df)
            df_featured['label'] = get_direction_labels(df_featured, forward_periods, tp_mult, sl_mult)
            coin_dfs[coin] = df_featured
            self.stdout.write(f"    ✅ {coin}: {len(df_featured)} samples processed")
        
        if len(coin_dfs) > 1:
            self.stdout.write("  - Adding cross-coin (market regime) features...")
            coin_dfs = add_cross_coin_features(coin_dfs)

        all_dfs = [df.assign(coin=coin) for coin, df in coin_dfs.items()]
        if not all_dfs:
            self.stdout.write(self.style.ERROR("No valid data found for any coin!"))
            return

        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df.dropna(subset=['label'], inplace=True)
        
        train_df = full_df[full_df['timestamp'] < cutoff].copy()
        test_df = full_df[full_df['timestamp'] >= cutoff].copy()

        self.stdout.write(f"  - Training samples: {len(train_df)} | Testing samples: {len(test_df)}")
        train_balance = train_df['label'].value_counts(normalize=True)
        self.stdout.write(f"  - Original training class balance: {train_balance.to_dict()}")

        self.stdout.write("  - Balancing training data via undersampling...")
        min_class_count = train_df['label'].value_counts().min()
        df_0 = train_df[train_df['label'] == 0].sample(n=min_class_count, random_state=42)
        df_1 = train_df[train_df['label'] == 1].sample(n=min_class_count, random_state=42)
        train_df_balanced = pd.concat([df_0, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)
        self.stdout.write(f"  - New balanced training samples: {len(train_df_balanced)}")

        train_df_balanced.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        self.stdout.write(self.style.SUCCESS("✅ Data generation complete."))

    def run_hyperparameter_tuning(self, n_trials, X, y):
        self.stdout.write(self.style.SUCCESS("\n--- Step 2: Hyperparameter Tuning with Optuna ---"))
        tscv = TimeSeriesSplit(n_splits=5)
        
        def objective(trial):
            params = {
                'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'boosting_type': 'gbdt',
                'n_estimators': 1000, 'random_state': 42,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            }
            model = lgb.LGBMClassifier(**params)
            scores = []
            for train_idx, val_idx in tscv.split(X):
                model.fit(X.iloc[train_idx], y.iloc[train_idx],
                          eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
                preds = model.predict_proba(X.iloc[val_idx])[:, 1]
                scores.append(roc_auc_score(y.iloc[val_idx], preds))
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        self.stdout.write(self.style.SUCCESS(f"✅ Tuning complete. Best CV AUC: {study.best_value:.4f}"))
        return study.best_params

    def train_final_model(self, X, y, params):
        self.stdout.write(self.style.SUCCESS("\n--- Step 3: Training Final Model ---"))
        params.update({'objective': 'binary', 'metric': 'auc', 'random_state': 42, 'n_estimators': 2000})
        model = lgb.LGBMClassifier(**params)
        model.fit(X, y)
        
        feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
        self.stdout.write("📊 Top 15 most important features:")
        self.stdout.write(feature_importance.head(15).to_string(index=False))
        return model

    def run_predictions(self, model, scaler, selected_features, imputation_values, test_path, prediction_path):
        self.stdout.write(self.style.SUCCESS("\n--- Step 4: Generating Predictions on Test Set ---"))
        test_df = pd.read_csv(test_path, parse_dates=['timestamp'])
        
        X_test = test_df[selected_features].copy()
        X_test.fillna(imputation_values, inplace=True)
        X_test_scaled = scaler.transform(X_test)

        probabilities = model.predict_proba(X_test_scaled)[:, 1]
        predictions = (probabilities > 0.5).astype(int)
        
        # Note: accuracy/AUC on imbalanced test set can be misleading. The simulation is the true test.
        self.stdout.write(f"📈 Test Set Performance (Reference):")
        self.stdout.write(f"  - Accuracy: {accuracy_score(test_df['label'], predictions):.4f}")
        self.stdout.write(f"  - AUC: {roc_auc_score(test_df['label'], probabilities):.4f}")

        output_df = test_df[['timestamp', 'coin', 'open', 'high', 'low', 'close', 'label']].copy()
        output_df['prediction_prob'] = probabilities
        output_df.to_csv(prediction_path, index=False)
        self.stdout.write(f"✅ Predictions saved to {prediction_path}")