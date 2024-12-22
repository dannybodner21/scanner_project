from django.db import models


class Coin(models.Model):
    name = models.CharField(max_length=100)
    symbol = models.CharField(max_length=10, unique=True)
    market_cap_rank = models.IntegerField(null=True, blank=True)
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name}"

class HistoricalData(models.Model):
    coin = models.ForeignKey(Coin, on_delete=models.CASCADE, related_name="historical_data")
    date = models.DateField()  # Store daily data
    price = models.DecimalField(max_digits=20, decimal_places=8)
    volume_24h = models.DecimalField(max_digits=20, decimal_places=2)

    def __str__(self):
        return f"Historical data for {self.coin.name} at {self.date}"

class ShortIntervalData(models.Model):
    coin = models.ForeignKey(Coin, on_delete=models.CASCADE, related_name="short_interval_data")
    timestamp = models.DateTimeField()
    price = models.DecimalField(max_digits=20, decimal_places=8)
    volume_5min = models.DecimalField(max_digits=20, decimal_places=2)

    def __str__(self):
        return f"Short interval for {self.coin.name} at {self.timestamp}"

class Metrics(models.Model):
    coin = models.ForeignKey(Coin, on_delete=models.CASCADE, related_name="metrics")
    timestamp = models.DateTimeField()  # When the metric was calculated
    relative_volume = models.FloatField(null=True, blank=True)
    price_change_5min = models.FloatField(null=True, blank=True)
    price_change_10min = models.FloatField(null=True, blank=True)
    price_change_1hr = models.FloatField(null=True, blank=True)
    price_change_24hr = models.FloatField(null=True, blank=True)
    price_change_7d = models.FloatField(null=True, blank=True)
    circulating_supply = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)
    volume_24h = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)
    last_price = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    market_cap = models.DecimalField(max_digits=20, decimal_places=2)


    def __str__(self):
        return f"Metrics for {self.coin.name} at {self.timestamp}"
