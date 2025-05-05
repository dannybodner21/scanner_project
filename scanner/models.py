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

    class Meta:
        indexes = [
            models.Index(fields=['coin', 'timestamp'], name='coin_timestamp_idx'),
        ]

    def __str__(self):
        return f"Short interval for {self.coin.name} at {self.timestamp}"


class HistoricalData(models.Model):
    coin = models.ForeignKey(Coin, on_delete=models.CASCADE, related_name="historical_data")
    date = models.DateField()
    price = models.DecimalField(max_digits=20, decimal_places=8)
    volume_24h = models.DecimalField(max_digits=20, decimal_places=2)

    def __str__(self):
        return f"Historical data for {self.coin.name} at {self.date}"


class FiredSignal(models.Model):
    coin = models.ForeignKey(Coin, on_delete=models.CASCADE, related_name="fired_signals")
    fired_at = models.DateTimeField()
    price_at_fired = models.DecimalField(max_digits=20, decimal_places=8)
    metrics = models.JSONField()
    take_profit_pct = models.FloatField(default=5.0)
    stop_loss_pct = models.FloatField(default=2.0)
    result = models.CharField(
        max_length=10,
        choices=[("win", "Win"), ("loss", "Loss"), ("unknown", "Unknown")],
        default="unknown"
    )
    checked_at = models.DateTimeField(null=True, blank=True)
    signal_type = models.CharField(
        max_length=10,
        choices=[("long", "Long"), ("short", "Short")],
        default="long"
    )
    closed_at = models.DateTimeField(null=True, blank=True)


    def __str__(self):
        return f"{self.coin.symbol} at {self.fired_at} — {self.result}"


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
    volatility_5min = models.FloatField(null=True, blank=True)
    volume_marketcap_ratio = models.FloatField(null=True, blank=True)
    trend_slope_30min = models.FloatField(null=True, blank=True)
    change_since_low = models.FloatField(null=True, blank=True)
    change_since_high = models.FloatField(null=True, blank=True)

    def __str__(self):
        coin_name = self.coin.name if self.coin else "Unknown"
        return f"Metrics for {coin_name} at {self.timestamp}"



class SuccessfulMove(models.Model):
    coin = models.ForeignKey(Coin, on_delete=models.CASCADE)
    timestamp = models.DateTimeField()
    entry_price = models.DecimalField(max_digits=20, decimal_places=8)
    move_type = models.CharField(max_length=10, choices=[("long", "Long"), ("short", "Short")])
    metrics = models.JSONField()
    entry_metrics = models.ForeignKey(Metrics, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return f"{self.move_type.upper()} — {self.coin.symbol} at {self.timestamp}"


class BacktestResult(models.Model):
    coin = models.ForeignKey(Coin, on_delete=models.CASCADE)
    timestamp = models.DateTimeField()
    entry_price = models.DecimalField(max_digits=20, decimal_places=10)
    exit_price = models.DecimalField(max_digits=20, decimal_places=10, null=True, blank=True)
    success = models.BooleanField()
    confidence = models.FloatField()
    entry_metrics = models.ForeignKey(Metrics, on_delete=models.SET_NULL, null=True, blank=True)
    trade_type = models.CharField(max_length=10, choices=[("long", "Long"), ("short", "Short")], default="long")


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


class RickisMetrics(models.Model):

    coin = models.ForeignKey(Coin, on_delete=models.CASCADE)
    timestamp = models.DateTimeField()
    price = models.DecimalField(max_digits=20, decimal_places=10)

    # High of Day Momentum
    high_24h = models.DecimalField(max_digits=20, decimal_places=10, null=True)
    low_24h = models.DecimalField(max_digits=20, decimal_places=10, null=True)
    open = models.DecimalField(max_digits=20, decimal_places=10, null=True)
    close = models.DecimalField(max_digits=20, decimal_places=10, null=True)

    # Top Gainers
    change_5m = models.FloatField(null=True)
    change_1h = models.FloatField(null=True)
    change_24h = models.FloatField(null=True)

    # Volume Spike
    volume = models.DecimalField(max_digits=30, decimal_places=2)
    avg_volume_1h = models.DecimalField(max_digits=30, decimal_places=2, null=True)

    # Reversal
    rsi = models.FloatField(null=True)
    macd = models.FloatField(null=True)
    macd_signal = models.FloatField(null=True)
    stochastic_k = models.FloatField(null=True)
    stochastic_d = models.FloatField(null=True)

    # Support/Resistance
    support_level = models.DecimalField(max_digits=20, decimal_places=10, null=True)
    resistance_level = models.DecimalField(max_digits=20, decimal_places=10, null=True)

    # Other
    price_slope_1h = models.FloatField(null=True)
    relative_volume = models.FloatField(null=True)
    sma_5 = models.DecimalField(max_digits=20, decimal_places=10, null=True)
    sma_20 = models.DecimalField(max_digits=20, decimal_places=10, null=True)
    ema_12 = models.DecimalField(max_digits=20, decimal_places=10, null=True)
    ema_26 = models.DecimalField(max_digits=20, decimal_places=10, null=True)
    stddev_1h = models.FloatField(null=True)
    atr_1h = models.DecimalField(max_digits=20, decimal_places=10, null=True)

    # fear / greed index score
    market_sentiment_score = models.IntegerField(null=True)
    market_sentiment_label = models.CharField(max_length=32, null=True)

    change_since_high = models.FloatField(null=True)  # price vs high_24h
    change_since_low = models.FloatField(null=True)   # price vs low_24h
    volume_mc_ratio = models.FloatField(null=True)

    obv = models.FloatField(null=True)
    adx = models.FloatField(null=True)
    bollinger_upper = models.FloatField(null=True)
    bollinger_middle = models.FloatField(null=True)
    bollinger_lower = models.FloatField(null=True)
    fib_distance_0_236 = models.FloatField(null=True)
    fib_distance_0_382 = models.FloatField(null=True)
    fib_distance_0_5   = models.FloatField(null=True)
    fib_distance_0_618 = models.FloatField(null=True)
    fib_distance_0_786 = models.FloatField(null=True)

    long_result = models.BooleanField(null=True)  # True = win, False = loss
    short_result = models.BooleanField(null=True)  # True = win, False = loss

    class Meta:

        indexes = [
            models.Index(fields=['coin', 'timestamp'], name='rickismetrics_idx'),
        ]

    def __str__(self):
        return f"{self.coin.symbol} @ {self.timestamp}"


class ModelTrade(models.Model):
    TRADE_TYPE_CHOICES = [
        ('long', 'Long'),
        ('short', 'Short'),
    ]

    coin = models.ForeignKey(Coin, on_delete=models.CASCADE)
    trade_type = models.CharField(max_length=5, choices=TRADE_TYPE_CHOICES)
    entry_timestamp = models.DateTimeField()
    exit_timestamp = models.DateTimeField(null=True, blank=True)

    entry_price = models.DecimalField(max_digits=20, decimal_places=10)
    exit_price = models.DecimalField(max_digits=20, decimal_places=10, null=True, blank=True)

    model_confidence = models.FloatField()
    take_profit_percent = models.FloatField()
    stop_loss_percent = models.FloatField()

    duration_minutes = models.IntegerField(null=True, blank=True)  # optional: calculated later
    result = models.BooleanField(null=True, blank=True)  # True = win, False = loss

    def __str__(self):
        return f"{self.coin.symbol} | {self.trade_type.upper()} | {self.entry_timestamp.strftime('%Y-%m-%d %H:%M')}"




#
