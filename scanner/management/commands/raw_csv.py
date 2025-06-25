import pandas as pd
from scanner.models import CoinAPIPrice

# Pull all rows for all coins you want
qs = CoinAPIPrice.objects.filter(
    timestamp__gte=datetime.datetime(2019, 1, 1, tzinfo=datetime.timezone.utc),
    timestamp__lt=datetime.datetime(2025, 6, 13, tzinfo=datetime.timezone.utc)
)

# Build list of dicts
data = list(qs.values('timestamp', 'coin', 'open', 'high', 'low', 'close', 'volume'))

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Sort to ensure proper order
df = df.sort_values(by=["coin", "timestamp"])

# Now save to CSV
df.to_csv("coinapi_raw_data.csv", index=False)
