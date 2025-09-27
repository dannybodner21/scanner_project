import pandas as pd
import numpy as np
from django.core.management.base import BaseCommand
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class Command(BaseCommand):
    help = 'Train flow-enhanced models with CVD, taker buy ratio, liquidation distance, and BTC lead-lag features'

    def add_arguments(self, parser):
        parser.add_argument('--coin', type=str, default='DOTUSDT', help='Coin to train model for')
        parser.add_argument('--train-end', type=str, default='2025-08-01', help='Training end date')
        parser.add_argument('--test-start', type=str, default='2025-08-01', help='Test start date')
        parser.add_argument('--output-dir', type=str, default='.', help='Output directory for models')
        parser.add_argument('--tp-pct', type=float, default=2.0, help='Take profit percentage')
        parser.add_argument('--sl-pct', type=float, default=1.0, help='Stop loss percentage')
        parser.add_argument('--leverage', type=float, default=15.0, help='Leverage multiplier')
        parser.add_argument('--max-hold', type=int, default=288, help='Max hold periods (288 = 24 hours)')
        parser.add_argument('--min-features', type=int, default=20, help='Minimum number of features to select')
        parser.add_argument('--max-features', type=int, default=50, help='Maximum number of features to select')

    def handle(self, *args, **options):
        coin = options['coin']
        train_end = pd.to_datetime(options['train_end'])
        test_start = pd.to_datetime(options['test_start'])
        output_dir = options['output_dir']
        tp_pct = options['tp_pct']
        sl_pct = options['sl_pct']
        leverage = options['leverage']
        max_hold = options['max_hold']
        min_features = options['min_features']
        max_features = options['max_features']

        self.stdout.write(f"🚀 FLOW-ENHANCED TRAINING for {coin}")
        self.stdout.write(f"📅 Train end: {train_end}, Test start: {test_start}")
        self.stdout.write(f"🎯 TP: {tp_pct}%, SL: {sl_pct}%, Leverage: {leverage}x")

        # Load OHLCV data
        self.stdout.write("📊 Loading OHLCV data...")
        ohlcv_df = pd.read_csv('baseline_ohlcv.csv')
        ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp']).dt.tz_localize(None)
        coin_data = ohlcv_df[ohlcv_df['coin'] == coin].copy()
        
        if coin_data.empty:
            self.stdout.write(self.style.ERROR(f"No data found for {coin}"))
            return

        # Load flow features
        self.stdout.write("🌊 Loading flow features...")
        flow_df = pd.read_csv('flow_features_full.csv')
        flow_df['timestamp'] = pd.to_datetime(flow_df['timestamp'], format='mixed').dt.tz_localize(None)
        
        # Merge OHLCV with flow features
        merged_df = pd.merge(coin_data, flow_df, on='timestamp', how='inner')
        merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)

        # Feature engineering
        self.stdout.write("🔧 Engineering features...")
        # Engineer features on the full dataset first
        feature_columns = self.engineer_features(merged_df, coin)
        
        # Create labels with TP/SL logic
        self.stdout.write("🏷️ Creating labels...")
        labels = self.create_labels(merged_df, tp_pct, sl_pct, leverage, max_hold)
        merged_df['label'] = labels

        # Split data
        train_df = merged_df[merged_df['timestamp'] < train_end].copy()
        test_df = merged_df[merged_df['timestamp'] >= test_start].copy()

        self.stdout.write(f"📈 Train samples: {len(train_df)}, Test samples: {len(test_df)}")
        self.stdout.write(f"📊 Train label distribution: {train_df['label'].value_counts().to_dict()}")
        
        # Prepare training data
        X_train = train_df[feature_columns].fillna(0)
        y_train = train_df['label']
        X_test = test_df[feature_columns].fillna(0)
        y_test = test_df['label']

        # Feature selection
        self.stdout.write("🎯 Selecting best features...")
        selected_features = self.select_features(X_train, y_train, min_features, max_features)
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        self.stdout.write(f"✅ Selected {len(selected_features)} features")

        # Train models
        self.stdout.write("🤖 Training models...")
        models = self.train_models(X_train_selected, y_train)

        # Evaluate models
        self.stdout.write("📊 Evaluating models...")
        results = self.evaluate_models(models, X_test_selected, y_test)

        # Save best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['balanced_accuracy'])
        best_model = models[best_model_name]
        
        self.stdout.write(f"🏆 Best model: {best_model_name} (Balanced Accuracy: {results[best_model_name]['balanced_accuracy']:.4f})")

        # Save model and artifacts
        model_filename = f"{coin.lower()}_flow_enhanced_model.joblib"
        scaler_filename = f"{coin.lower()}_flow_enhanced_scaler.joblib"
        features_filename = f"{coin.lower()}_flow_enhanced_features.json"
        config_filename = f"{coin.lower()}_flow_enhanced_config.json"

        joblib.dump(best_model, model_filename)
        joblib.dump(RobustScaler().fit(X_train_selected), scaler_filename)
        
        with open(features_filename, 'w') as f:
            json.dump(selected_features, f)
        
        config = {
            'coin': coin,
            'model_type': best_model_name,
            'features': selected_features,
            'tp_pct': tp_pct,
            'sl_pct': sl_pct,
            'leverage': leverage,
            'max_hold': max_hold,
            'train_end': train_end.isoformat(),
            'test_start': test_start.isoformat(),
            'performance': results[best_model_name]
        }
        
        with open(config_filename, 'w') as f:
            json.dump(config, f, indent=2)

        # Generate predictions
        self.stdout.write("🔮 Generating predictions...")
        predictions = self.generate_predictions(best_model, X_test_selected, test_df['timestamp'], coin)
        
        pred_filename = f"{coin.lower()}_flow_enhanced_predictions.csv"
        predictions.to_csv(pred_filename, index=False)

        self.stdout.write(self.style.SUCCESS("✅ Flow-enhanced training completed!"))
        self.stdout.write(f"📁 Saved: {model_filename}, {scaler_filename}, {features_filename}, {config_filename}, {pred_filename}")

    def create_labels(self, df, tp_pct, sl_pct, leverage, max_hold):
        """Create labels based on TP/SL logic"""
        labels = []
        
        for i in range(len(df)):
            if i + max_hold >= len(df):
                labels.append(0)  # Not enough data for max hold
                continue
                
            entry_price = df.iloc[i]['close']
            tp_price = entry_price * (1 + tp_pct / 100)
            sl_price = entry_price * (1 - sl_pct / 100)
            
            # Check next max_hold periods
            future_prices = df.iloc[i+1:i+max_hold+1]['close'].values
            
            if len(future_prices) == 0:
                labels.append(0)
                continue
                
            # Check if TP hit first
            tp_hit = np.any(future_prices >= tp_price)
            sl_hit = np.any(future_prices <= sl_price)
            
            if tp_hit and not sl_hit:
                labels.append(1)  # Win
            elif sl_hit:
                labels.append(0)  # Loss
            else:
                labels.append(0)  # No clear outcome
        
        return labels

    def engineer_features(self, df, coin):
        """Engineer comprehensive features including flow features"""
        features = []
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        df['price_ma_5'] = df['close'].rolling(5).mean()
        df['price_ma_20'] = df['close'].rolling(20).mean()
        df['price_ma_50'] = df['close'].rolling(50).mean()
        
        # Technical indicators
        df['rsi'] = self.calculate_rsi(df['close'], 14)
        df['macd'] = self.calculate_macd(df['close'])
        df['bb_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
        df['bb_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['price_volume'] = df['close'] * df['volume']
        
        # Flow features (already in the data) - prioritize the most important ones
        flow_features = [col for col in df.columns if any(x in col for x in [
            'cvd_5m', 'cvd_momentum', 'cvd_ratio',  # CVD features
            'taker_buy_ratio', 'taker_buy_momentum',  # Taker buy features
            'liq_distance_above', 'liq_distance_below', 'liq_pressure',  # Liquidation features
            'btc_eth_lead_lag_corr', 'btc_dominance', 'btc_dominance_momentum',  # BTC features
            'market_cvd', 'market_taker_buy_ratio'  # Market features
        ])]
        
        # Time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
            df[f'volume_std_{window}'] = df['volume'].rolling(window).std()
            df[f'price_skew_{window}'] = df['returns'].rolling(window).skew()
            df[f'price_kurt_{window}'] = df['returns'].rolling(window).kurt()
            df[f'price_quantile_25_{window}'] = df['returns'].rolling(window).quantile(0.25)
            df[f'price_quantile_75_{window}'] = df['returns'].rolling(window).quantile(0.75)
        
        # Advanced price features
        df['price_momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['price_momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['price_momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Volume-price relationship
        df['volume_price_trend'] = df['volume'] * df['returns']
        df['volume_price_corr'] = df['volume'].rolling(20).corr(df['close'])
        
        # Support and resistance levels
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])
        
        # Volatility features
        df['atr'] = self.calculate_atr(df, 14)
        df['volatility_ratio'] = df['volatility'] / df['atr']
        
        # Trend strength
        df['trend_strength'] = abs(df['price_ma_5'] - df['price_ma_20']) / df['price_ma_20']
        
        # Feature list
        price_features = ['returns', 'log_returns', 'volatility', 'price_ma_5', 'price_ma_20', 'price_ma_50']
        tech_features = ['rsi', 'macd', 'bb_position']
        volume_features = ['volume_ratio', 'price_volume', 'volume_price_trend', 'volume_price_corr']
        time_features = ['hour', 'day_of_week', 'is_weekend']
        lag_features = [col for col in df.columns if 'lag_' in col]
        rolling_features = [col for col in df.columns if any(x in col for x in ['_std_', '_skew_', '_kurt_', '_quantile_'])]
        advanced_features = [
            'price_momentum_5', 'price_momentum_10', 'price_momentum_20',
            'price_position', 'atr', 'volatility_ratio', 'trend_strength'
        ]
        
        features = price_features + tech_features + volume_features + flow_features + time_features + lag_features + rolling_features + advanced_features
        
        # Remove any features that don't exist
        features = [f for f in features if f in df.columns]
        
        return features

    def calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd

    def calculate_atr(self, df, window=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=window).mean()
        return atr

    def select_features(self, X, y, min_features, max_features):
        """Select best features using Random Forest importance"""
        # Remove any columns with all NaN or infinite values
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Remove constant features
        constant_features = X_clean.columns[X_clean.nunique() <= 1].tolist()
        X_clean = X_clean.drop(columns=constant_features)
        
        self.stdout.write(f"🔍 Starting with {X_clean.shape[1]} features after removing constants")
        
        # Use Random Forest for feature selection with better parameters
        rf = RandomForestClassifier(
            n_estimators=1000, 
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42, 
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.stdout.write("🌲 Training Random Forest for feature selection...")
        rf.fit(X_clean, y)
        
        # Get feature importances
        feature_importance = pd.Series(rf.feature_importances_, index=X_clean.columns)
        feature_importance = feature_importance.sort_values(ascending=False)
        
        # Select top features
        selected_features = feature_importance.head(max_features).index.tolist()
        
        # Ensure we have at least min_features
        if len(selected_features) < min_features:
            remaining_features = [f for f in X_clean.columns if f not in selected_features]
            if remaining_features:
                additional_needed = min_features - len(selected_features)
                selected_features.extend(remaining_features[:additional_needed])
        
        self.stdout.write(f"✅ Selected {len(selected_features)} features from {X_clean.shape[1]} available")
        self.stdout.write(f"🏆 Top 10 features: {selected_features[:10]}")
        
        return selected_features

    def train_models(self, X, y):
        """Train multiple models with optimized parameters"""
        models = {}
        
        # Random Forest - More aggressive parameters
        models['RandomForest'] = RandomForestClassifier(
            n_estimators=1000,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # HistGradientBoosting - More iterations and better parameters
        models['HistGradientBoosting'] = HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=12,
            learning_rate=0.05,
            min_samples_leaf=1,
            l2_regularization=0.1,
            random_state=42,
            class_weight='balanced'
        )
        
        # Logistic Regression with regularization
        models['LogisticRegression'] = LogisticRegression(
            C=0.1,
            penalty='l1',
            solver='liblinear',
            random_state=42,
            class_weight='balanced',
            max_iter=2000
        )
        
        # XGBoost (if available) - skip for now due to architecture issues
        # try:
        #     import xgboost as xgb
        #     models['XGBoost'] = xgb.XGBClassifier(
        #         n_estimators=1000,
        #         max_depth=8,
        #         learning_rate=0.05,
        #         subsample=0.8,
        #         colsample_bytree=0.8,
        #         random_state=42,
        #         scale_pos_weight=len(y[y==0])/len(y[y==1]) if len(y[y==1]) > 0 else 1
        #     )
        # except ImportError:
        #     self.stdout.write("⚠️ XGBoost not available, skipping...")
        
        # Train all models
        for name, model in models.items():
            self.stdout.write(f"Training {name}...")
            model.fit(X, y)
        
        return models

    def evaluate_models(self, models, X_test, y_test):
        """Evaluate all models"""
        results = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0,
                'mcc': matthews_corrcoef(y_test, y_pred)
            }
            
            self.stdout.write(f"{name}: Accuracy={results[name]['accuracy']:.4f}, "
                            f"Balanced={results[name]['balanced_accuracy']:.4f}, "
                            f"F1={results[name]['f1']:.4f}")
        
        return results

    def generate_predictions(self, model, X_test, timestamps, coin):
        """Generate predictions for test data"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        predictions_df = pd.DataFrame({
            'coin': coin,
            'timestamp': timestamps,
            'prediction': y_pred,
            'pred_prob': y_pred_proba
        })
        
        return predictions_df
