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


class ShortIntervalData(models.Model):
    coin = models.ForeignKey(Coin, on_delete=models.CASCADE, related_name="short_interval_data")
    timestamp = models.DateTimeField()
    price = models.DecimalField(max_digits=20, decimal_places=8)
    volume_5min = models.DecimalField(max_digits=20, decimal_places=2)
    circulating_supply = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)

    def __str__(self):
        return f"Short interval for {self.coin.name} at {self.timestamp}"


class HistoricalData(models.Model):
    coin = models.ForeignKey(Coin, on_delete=models.CASCADE, related_name="historical_data")
    date = models.DateField()  # Store daily data
    price = models.DecimalField(max_digits=20, decimal_places=8)
    volume_24h = models.DecimalField(max_digits=20, decimal_places=2)

    def __str__(self):
        return f"Historical data for {self.coin.name} at {self.date}"


class HighLowData(models.Model):
    coin = models.ForeignKey(Coin, on_delete=models.CASCADE, related_name="high_low_data")
    daily_high = models.DecimalField(max_digits=20, decimal_places=8)
    daily_low = models.DecimalField(max_digits=20, decimal_places=8)
    timestamp = models.DateTimeField()

    def __str__(self):
        return f"{self.coin.symbol}, high: {self.daily_high}, low: {self.daily_low}, {self.timestamp}"


class SupportResistance(models.Model):
    coin = models.ForeignKey(Coin, on_delete=models.CASCADE, related_name="support_resistance")
    level_one = models.DecimalField(max_digits=20, decimal_places=8, null=True)
    level_two = models.DecimalField(max_digits=20, decimal_places=8, null=True)
    level_three = models.DecimalField(max_digits=20, decimal_places=8, null=True)
    level_four = models.DecimalField(max_digits=20, decimal_places=8, null=True)
    level_five = models.DecimalField(max_digits=20, decimal_places=8, null=True)
    level_six = models.DecimalField(max_digits=20, decimal_places=8, null=True)
    timestamp = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.coin.symbol}, {self.timestamp}"


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


class Trigger(models.Model):
    trigger_name = models.CharField(max_length=200)
    timestamp = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.trigger_name} at {self.timestamp}"


class Pattern(models.Model):
    coin = models.ForeignKey(Coin, on_delete=models.CASCADE, related_name="pattern")
    symbol = models.CharField(max_length=200, null=True, blank=True)
    name = models.CharField(max_length=200, null=True, blank=True)
    patterntype = models.CharField(max_length=200, null=True, blank=True)
    status = models.CharField(max_length=200)
    entry = models.DecimalField(max_digits=20, decimal_places=8)
    takeprofit = models.DecimalField(max_digits=20, decimal_places=8)
    stoploss = models.DecimalField(max_digits=20, decimal_places=8)
    five_min_signal = models.CharField(max_length=200, null=True, blank=True)
    fifteen_min_signal = models.CharField(max_length=200, null=True, blank=True)
    one_hour_signal = models.CharField(max_length=200, null=True, blank=True)
    five_min_adx = models.DecimalField(max_digits=20, decimal_places=8, null=True)
    fifteen_min_adx = models.DecimalField(max_digits=20, decimal_places=8, null=True)
    one_hour_adx = models.DecimalField(max_digits=20, decimal_places=8, null=True)
    timestamp = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.symbol}, {self.timestamp}"


'''
class TriggerCombination(models.Model):
    coin = models.ForeignKey(Coin, on_delete=models.CASCADE, related_name="trigger_combination")
    timestamp = models.DateTimeField(auto_now=True)
    daily_relative_volume = models.FloatField(null=True, blank=True)
    rolling_relative_volume = models.FloatField(null=True, blank=True)
    five_min_relative_volume = models.FloatField(null=True, blank=True)
    twenty_min_relative_volume = models.FloatField(null=True, blank=True)
    price_change_5min = models.FloatField(null=True, blank=True)
    price_change_10min = models.FloatField(null=True, blank=True)
    price_change_1hr = models.FloatField(null=True, blank=True)
    price_change_24hr = models.FloatField(null=True, blank=True)
    price_change_7d = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.coin.symbol} at {self.timestamp}"
'''





#
