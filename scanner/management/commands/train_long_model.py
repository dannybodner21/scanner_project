from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from django.utils.timezone import make_aware
from datetime import datetime
import joblib
import os

class Command(BaseCommand):
    help = 'Train XGBoost model on full RickisMetrics between March 22, 2025 and April 22, 2025'

    def handle(self, *args, **kwargs):
        self.stdout.write("🚀 Loading RickisMetrics data...")

        start_date = make_aware(datetime(2025, 3, 22))
        end_date = make_aware(datetime(2025, 4, 23))

        qs = RickisMetrics.objects.filter(
            timestamp__gte=start_date,
            timestamp__lt=end_date
        ).values(
            'price', 'volume', 'change_5m', 'change_1h', 'change_24h',
            'high_24h', 'low_24h', 'avg_volume_1h', 'relative_volume',
            'sma_5', 'sma_20', 'ema_12', 'ema_26', 'macd', 'macd_signal',
            'rsi', 'stddev_1h', 'price_slope_1h', 'stochastic_k', 'stochastic_d',
            'support_level', 'resistance_level', 'atr_1h', 'long_result'
        )

        df = pd.DataFrame(qs)

        if df.empty:
            self.stdout.write(self.style.ERROR("❌ No data found in RickisMetrics for the given date range."))
            return

        self.stdout.write(f"✅ Loaded {len(df)} entries.")

        # Convert object columns to float
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'long_result':
                df[col] = df[col].astype('float')

        # Ensure long_result is an integer (binary classification)
        df['long_result'] = df['long_result'].astype(int)

        # Drop rows with missing values
        df = df.dropna()
        self.stdout.write(f"✅ {len(df)} entries after dropping missing values.")

        # Split into features and labels
        X = df.drop(columns=['long_result'])
        y = df['long_result']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.stdout.write("🚀 Training XGBoost model...")

        # Train XGBoost
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_estimators=100,
            max_depth=6,
            tree_method='hist'  # Best for 4GB RAM
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.stdout.write(self.style.SUCCESS(f"✅ Model training complete."))
        self.stdout.write(self.style.SUCCESS(f"✅ Model Accuracy: {accuracy:.4f}"))
        self.stdout.write(self.style.SUCCESS(f"\nClassification Report:\n{classification_report(y_test, y_pred)}"))

        # ✅ NEW: Save model
        model_path = '/workspace/scanner/xgboost_long_model.pkl'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        self.stdout.write(self.style.SUCCESS(f"✅ Model saved to {model_path}"))
