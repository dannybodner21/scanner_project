import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Test the saved XGBoost model on 2025 test data, print metrics, and save predictions"

    def handle(self, *args, **options):
        test_csv_path = 'six_long_testing_data.csv'
        model_path = 'six_long_xgb_model.bin'
        output_csv_path = 'six_long_predictions.csv'

        self.stdout.write(f"Loading test data from {test_csv_path} ...")
        df_test = pd.read_csv(test_csv_path, parse_dates=['timestamp'])

        self.stdout.write("Preparing test features and labels ...")
        X_test = df_test.drop(columns=['label', 'coin', 'timestamp'], errors='ignore')
        y_test = df_test['label']

        self.stdout.write(f"Loading model from {model_path} ...")
        model = xgb.Booster()
        model.load_model(model_path)

        self.stdout.write("Creating DMatrix for test data ...")
        dtest = xgb.DMatrix(X_test)

        self.stdout.write("Predicting probabilities on test data ...")
        y_pred_prob = model.predict(dtest)

        self.stdout.write("Generating predicted labels with 0.8 threshold ...")
        y_pred = (y_pred_prob >= 0.97).astype(int)

        self.stdout.write("Classification Report on 2025 test data:")
        report = classification_report(y_test, y_pred)
        self.stdout.write(report)

        self.stdout.write(f"Saving predictions to {output_csv_path} ...")
        df_test['prediction'] = y_pred
        df_test['prediction_prob'] = y_pred_prob
        df_test.sort_values('timestamp', inplace=True)

        df_test.to_csv(output_csv_path, index=False)
        self.stdout.write("Done.")
