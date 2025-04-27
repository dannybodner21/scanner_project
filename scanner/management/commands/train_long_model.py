from django.core.management.base import BaseCommand
from scanner.models import RickisMetrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from django.utils.timezone import make_aware
from datetime import datetime

class Command(BaseCommand):
    help = 'Train XGBoost model on RickisMetrics from 2025-03-22 to 2025-04-22'

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

        df = df.dropna()
        self.stdout.write(f"✅ {len(df)} entries after dropping rows with missing values.")

        X = df.drop(columns=['long_result'])
        y = df['long_result']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.stdout.write("🚀 Training XGBoost model...")

        #model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            max_depth=3,      # limit tree depth
            n_estimators=50   # fewer trees
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        self.stdout.write(self.style.SUCCESS(f"✅ Model training complete."))
        self.stdout.write(self.style.SUCCESS(f"✅ Model Accuracy: {accuracy:.4f}"))
