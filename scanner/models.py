from django.db import models


class Coin(models.Model):
    cmc_id = models.IntegerField(unique=True)
    name = models.CharField(max_length=100)
    symbol = models.CharField(max_length=10)
    market_cap_rank = models.IntegerField(null=True, blank=True)
    last_updated = models.DateTimeField(auto_now=True)
    exchange = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.name}"


class MemeCoin(models.Model):
    cmc_id = models.IntegerField(unique=True)
    name = models.CharField(max_length=100)
    symbol = models.CharField(max_length=10)
    market_cap_rank = models.IntegerField(null=True, blank=True)
    date_added = models.DateTimeField(auto_now=True)

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
    circulating_supply = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)

    def __str__(self):
        return f"Short interval for {self.coin.name} at {self.timestamp}"


class MemeShortIntervalData(models.Model):
    coin = models.ForeignKey(MemeCoin, on_delete=models.CASCADE, related_name="meme_short_interval_data")
    timestamp = models.DateTimeField()
    price = models.DecimalField(max_digits=20, decimal_places=8)
    volume_5min = models.DecimalField(max_digits=20, decimal_places=2)
    circulating_supply = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)

    def __str__(self):
        return f"Short interval for {self.coin.name} at {self.timestamp}"


class Metrics(models.Model):
    coin = models.ForeignKey(Coin, on_delete=models.CASCADE, related_name="metrics")
    timestamp = models.DateTimeField()
    daily_relative_volume = models.FloatField(null=True, blank=True)
    rolling_relative_volume = models.FloatField(null=True, blank=True)
    five_min_relative_volume = models.FloatField(null=True, blank=True)
    twenty_min_relative_volume = models.FloatField(null=True, blank=True)
    price_change_5min = models.FloatField(null=True, blank=True)
    price_change_10min = models.FloatField(null=True, blank=True)
    price_change_1hr = models.FloatField(null=True, blank=True)
    price_change_24hr = models.FloatField(null=True, blank=True)
    price_change_7d = models.FloatField(null=True, blank=True)
    circulating_supply = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)
    volume_24h = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)
    last_price = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    market_cap = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)

    def __str__(self):
        return f"Metrics for {self.coin.name} at {self.timestamp}"


class MemeMetric(models.Model):
    coin = models.ForeignKey(MemeCoin, on_delete=models.CASCADE, related_name="memeMetrics")
    timestamp = models.DateTimeField()
    five_min_relative_volume = models.FloatField(null=True, blank=True)
    price_change_5min = models.FloatField(null=True, blank=True)
    price_change_10min = models.FloatField(null=True, blank=True)
    price_change_1hr = models.FloatField(null=True, blank=True)
    circulating_supply = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)
    volume_24h = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)
    last_price = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    market_cap = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)

    def __str__(self):
        return f"Metrics for {self.coin.name} at {self.timestamp}"


class Triggers(models.Model):
    trigger = models.CharField(max_length=200)
    timestamp = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.trigger}"
