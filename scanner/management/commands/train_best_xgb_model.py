import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from django.core.management.base import BaseCommand

# python manage.py train_best_xgb_model

class Command(BaseCommand):
    help = 'Train XGBoost model for LONG trades with diagnostics'

    def handle(self, *args, **options):
        train_file = 'five_long_training_data.csv'
        model_output = 'five_long_xgb_model.bin'
        importance_csv = 'five_long_feature_importance.csv'
        test_size = 0.1  # 10% validation split

        self.stdout.write(f"Loading training data from {train_file} ...")
        df = pd.read_csv(train_file, index_col=0, parse_dates=True)
        self.stdout.write(f"Loaded {len(df)} rows.")

        drop_cols = [col for col in ['coin', 'timestamp'] if col in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
            self.stdout.write(f"Dropped columns: {drop_cols}")

        X = df.drop(columns=['label'])
        y = df['label']

        self.stdout.write("Splitting data (90% train, 10% validation)...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        self.stdout.write(f"Train set size: {len(X_train)}")
        self.stdout.write(f"Validation set size: {len(X_val)}")

        ratio = (y_train == 0).sum() / (y_train == 1).sum()
        self.stdout.write(f"Scale_pos_weight (neg/pos): {ratio:.2f}")

        self.stdout.write("Training label distribution:\n" + str(y_train.value_counts()))
        self.stdout.write("Validation label distribution:\n" + str(y_val.value_counts()))

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'max_depth': 6,
            'eta': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': ratio,
            'seed': 42,
        }

        evallist = [(dtrain, 'train'), (dval, 'eval')]

        self.stdout.write("Starting training with early stopping...")
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=evallist,
            early_stopping_rounds=30,
            verbose_eval=True,
        )

        y_pred_prob = bst.predict(dval)
        y_pred = (y_pred_prob >= 0.5).astype(int)

        report = classification_report(y_val, y_pred)
        self.stdout.write("Validation classification report:\n" + report)

        importance = bst.get_score(importance_type='gain')
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        self.stdout.write("Top 10 features by gain:")
        for feat, score in sorted_importance[:10]:
            self.stdout.write(f"{feat}: {score:.4f}")

        pd.DataFrame(sorted_importance, columns=["feature", "gain"]).to_csv(importance_csv, index=False)
        self.stdout.write(f"Full feature importance saved to {importance_csv}")

        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(X_train, y_train)
        y_dummy_pred = dummy.predict(X_val)
        dummy_report = classification_report(y_val, y_dummy_pred)
        self.stdout.write("Dummy classifier classification report on validation:\n" + dummy_report)

        bst.save_model(model_output)
        self.stdout.write(f"Model saved to {model_output}")
