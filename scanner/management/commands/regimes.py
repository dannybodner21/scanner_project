import pandas as pd
from datetime import datetime, timezone
from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice

class Command(BaseCommand):
    help = 'Calculate market regime (bull, bear, sideways) on BTC 5-min data'

    def handle(self, *args, **options):
        print("Loading BTC data...")
        df = self.load_data('BTCUSDT')
        print(f"Loaded {len(df)} rows.")

        print("Calculating market regimes...")
        df = self.calculate_regimes(df)
        print("Market regimes added.")

        df.to_csv('btc_market_regimes.csv')
        print("Saved data with regimes to btc_market_regimes.csv")

    def load_data(self, coin):
        start_date = datetime(2019, 1, 1, tzinfo=timezone.utc)
        end_date = datetime.now(timezone.utc)

        queryset = CoinAPIPrice.objects.filter(
            coin=coin,
            timestamp__gte=start_date,
            timestamp__lte=end_date
        ).order_by('timestamp')

        df = pd.DataFrame(list(queryset.values('timestamp', 'open', 'high', 'low', 'close', 'volume')))
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        df = df.set_index('timestamp').sort_index()
        return df

    def calculate_regimes(self, df):
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df['ma_200'] = df['close'].rolling(200).mean()

        median_vol = df['volatility'].median()

        df['bull_regime'] = ((df['close'] > df['ma_200']) & (df['volatility'] < median_vol)).astype(int)
        df['bear_regime'] = ((df['close'] < df['ma_200']) & (df['volatility'] > median_vol)).astype(int)
        df['sideways_regime'] = 1 - df['bull_regime'] - df['bear_regime']

        return df.dropna()
