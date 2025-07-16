import pandas as pd
import joblib
import numpy as np
from django.core.management.base import BaseCommand
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

class Command(BaseCommand):
    help = 'Calibrate a trained ML model to output proper probability scores'

    def handle(self, *args, **options):
        # === 🔧 Hardcoded paths ===
        model_path = "three_model.joblib"  # change this to your model file
        dataset_path = "two_training.csv"  # full training set with features + label
        feature_list_path = "two_selected_features.joblib"  # selected feature names
        scaler_path = "two_feature_scaler.joblib"  # same scaler used for original model

        output_path = model_path.replace(".joblib", "_calibrated.joblib")

        # === 🧠 Load components ===
        print("📦 Loading model and dataset...")
        model = joblib.load(model_path)
        df = pd.read_csv(dataset_path, parse_dates=["timestamp"])
        selected_features = joblib.load(feature_list_path)
        scaler = joblib.load(scaler_path)

        df = df.dropna()
        X = df[selected_features]
        y = df["label"]

        # === ⚖️ Scale and split ===
        X_scaled = scaler.transform(X)
        X_train, X_calib, y_train, y_calib = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

        # === ✅ Wrap with calibrated classifier ===
        print("🔧 Calibrating model...")
        calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
        calibrated_model.fit(X_calib, y_calib)

        # === 💾 Save calibrated model ===
        joblib.dump(calibrated_model, output_path)
        print(f"✅ Calibrated model saved to: {output_path}")
