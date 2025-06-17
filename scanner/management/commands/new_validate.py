import pandas as pd
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Validate generated long model CSV data for nulls, zeros, and basic stats'

    def handle(self, *args, **options):
        file_path = 'new_long_training_data.csv'

        print(f"Loading {file_path}...")
        df = pd.read_csv(file_path, index_col=0)

        print(f"Data shape: {df.shape}")
        print("\nNull values per column:")
        print(df.isnull().sum())

        print("\nZero value counts per numeric column:")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            zero_count = (df[col] == 0).sum()
            print(f"  {col}: {zero_count}")

        print("\nBasic stats summary:")
        print(df.describe())

        print("\nSample rows:")
        print(df.head(5))
