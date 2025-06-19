import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Train XGBoost model for SHORT trades with diagnostics'

    def add_arguments(self, parser):
        parser.add_argument(
            '--train-file',
            type=str,
            default='short_training_data.csv',
            help='Path to the SHORT trade training CSV file',
        )
        parser.add_argument(
            '--model-output',
            type=str,
            default='best_short_xgb_model.bin',
            help='Filename to save the trained SHORT model',
        )
        parser.add_argument(
            '--split-date',
            type=str,
            default='2024-07-01',
            help='YYYY-MM-DD date to split training/validation',
        )

    def handle(self, *args, **options):
        train_file = options['train_file']
        model_output = options['model_output']
        split_date_str = options['split_date']

        self.stdout.write(f"Loading training data from {train_file} ...")
        df = pd.read_csv(train_file, index_col=0, parse_dates=True)
        self.stdout.write(f"Loaded {len(df)} rows.")

        self.stdout.write(f"Index is unique? {df.index.is_unique}")
        duplicate_indices_count = df.index.duplicated().sum()
        self.stdout.write(f"Number of duplicate indices: {duplicate_indices_count}")
        self.stdout.write(f"Index timezone info: {df.index.tz}")

        split_date = pd.Timestamp(split_date_str)
        if df.index.tz is not None:
            split_date = split_date.tz_localize(df.index.tz)
        self.stdout.write(f"Split date with timezone: {split_date}")

        drop_cols = [col for col in ['coin', 'timestamp'] if col in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
            self.stdout.write(f"Dropped columns: {drop_cols}")

        X = df.drop(columns=['label'])
        y = df['label']

        train_mask = df.index < split_date
        val_mask = df.index >= split_date

        overlap = set(df.loc[train_mask].index).intersection(set(df.loc[val_mask].index))
        self.stdout.write(f"Overlap rows count between train and val: {len(overlap)} (should be 0)")

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]
        X_val = X.loc[val_mask]
        y_val = y.loc[val_mask]

        self.stdout.write(f"Full dataset label distribution:\n{df['label'].value_counts(normalize=True)}")
        self.stdout.write(f"Train label distribution:\n{y_train.value_counts(normalize=True)}")
        self.stdout.write(f"Validation label distribution:\n{y_val.value_counts(normalize=True)}")

        self.stdout.write("Sample TRAIN rows (features + label):")
        train_sample = df.loc[train_mask].head(10)
        self.stdout.write(str(train_sample))

        self.stdout.write("Sample VALIDATION rows (features + label):")
        val_sample = df.loc[val_mask].head(10)
        self.stdout.write(str(val_sample))

        ratio = (y_train == 0).sum() / (y_train == 1).sum()
        self.stdout.write(f"Scale_pos_weight (neg/pos): {ratio:.2f}")

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

        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(X_train, y_train)
        y_dummy_pred = dummy.predict(X_val)
        dummy_report = classification_report(y_val, y_dummy_pred)
        self.stdout.write("Dummy classifier classification report on validation:\n" + dummy_report)

        bst.save_model(model_output)
        self.stdout.write(f"Model saved to {model_output}")
