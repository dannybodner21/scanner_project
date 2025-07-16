from django.db import models
from django.db.models import JSONField


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
    coin = models.ForeignKey(Coin, on_delete=models.CASCADE, related_name="patterns")
    symbol = models.CharField(max_length=200, null=True, blank=True)
    resolution = models.IntegerField(null=True, blank=True)  # 5, 15, or 60
    patterntype = models.CharField(max_length=200, null=True, blank=True)
    patternname = models.CharField(max_length=200, null=True, blank=True)
    status = models.CharField(max_length=200, null=True, blank=True)
    entry = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    takeprofit = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    stoploss = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    adx = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    timestamp = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.symbol} | {self.resolution}min | {self.patternname} | {self.timestamp}"



class RickisMetrics(models.Model):

    coin = models.ForeignKey(Coin, on_delete=models.CASCADE)
    timestamp = models.DateTimeField()
    price = models.DecimalField(max_digits=20, decimal_places=10) # good
    high_24h = models.DecimalField(max_digits=20, decimal_places=10, null=True) # good
    low_24h = models.DecimalField(max_digits=20, decimal_places=10, null=True) # good
    open = models.DecimalField(max_digits=20, decimal_places=10, null=True) # good
    close = models.DecimalField(max_digits=20, decimal_places=10, null=True) # good
    change_5m = models.FloatField(null=True) # 9554 zeros - leave it
    change_1h = models.FloatField(null=True) # good
    change_24h = models.FloatField(null=True) # good
    volume = models.DecimalField(max_digits=30, decimal_places=2) # good
    avg_volume_1h = models.DecimalField(max_digits=30, decimal_places=2, null=True) # good
    rsi = models.FloatField(null=True) # good
    macd = models.FloatField(null=True) # good
    macd_signal = models.FloatField(null=True) # good
    stochastic_k = models.FloatField(null=True) # good
    stochastic_d = models.FloatField(null=True) # good
    support_level = models.DecimalField(max_digits=20, decimal_places=10, null=True) # good
    resistance_level = models.DecimalField(max_digits=20, decimal_places=10, null=True) # good
    relative_volume = models.FloatField(null=True) # good
    sma_5 = models.DecimalField(max_digits=20, decimal_places=10, null=True) # good
    sma_20 = models.DecimalField(max_digits=20, decimal_places=10, null=True) # good
    stddev_1h = models.FloatField(null=True) # 60 zeros
    atr_1h = models.DecimalField(max_digits=20, decimal_places=10, null=True) # 538 zeros
    change_since_high = models.FloatField(null=True) # good
    change_since_low = models.FloatField(null=True) # good
    fib_distance_0_236 = models.FloatField(null=True) # 18
    fib_distance_0_382 = models.FloatField(null=True) # 18
    fib_distance_0_5   = models.FloatField(null=True) # 18
    fib_distance_0_618 = models.FloatField(null=True) # 18
    fib_distance_0_786 = models.FloatField(null=True) # 18
    adx = models.FloatField(null=True) # good
    bollinger_upper = models.FloatField(null=True) # good
    bollinger_middle = models.FloatField(null=True) # good
    bollinger_lower = models.FloatField(null=True) # good

    chart_pattern_5m = models.CharField(max_length=100, null=True, blank=True)
    chart_pattern_15m = models.CharField(max_length=100, null=True, blank=True)
    chart_pattern_60m = models.CharField(max_length=100, null=True, blank=True)

    fear_greed = models.FloatField(null=True)


    long_result = models.BooleanField(null=True)
    short_result = models.BooleanField(null=True)




    obv = models.FloatField(null=True) # 46k zeros - don't use
    price_slope_1h = models.FloatField(null=True) # bad
    ema_12 = models.DecimalField(max_digits=20, decimal_places=10, null=True)
    ema_26 = models.DecimalField(max_digits=20, decimal_places=10, null=True)
    volume_mc_ratio = models.FloatField(null=True)
    market_sentiment_score = models.IntegerField(null=True)
    market_sentiment_label = models.CharField(max_length=32, null=True)

    class Meta:

        indexes = [
            models.Index(fields=['coin', 'timestamp'], name='rickismetrics_idx'),
        ]

    def __str__(self):
        return f"{self.coin.symbol} @ {self.timestamp}"









class CoinAPIPrice(models.Model):
    coin = models.CharField(max_length=20)  # e.g. 'BTCUSDT'
    timestamp = models.DateTimeField()  # 5-min candle open time (UTC)

    open = models.DecimalField(max_digits=20, decimal_places=10, null=True)
    high = models.DecimalField(max_digits=20, decimal_places=10, null=True)
    low = models.DecimalField(max_digits=20, decimal_places=10, null=True)
    close = models.DecimalField(max_digits=20, decimal_places=10, null=True)
    volume = models.DecimalField(max_digits=30, decimal_places=15, null=True)

    class Meta:
        indexes = [
            models.Index(fields=['coin', 'timestamp']),
        ]
        unique_together = ('coin', 'timestamp')



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
    duration_minutes = models.IntegerField(null=True, blank=True)
    result = models.BooleanField(null=True, blank=True)
    confidence_trade = models.FloatField(null=True, blank=True)
    model_name = models.CharField(max_length=100, null=True, blank=True)
    recent_confidences = JSONField(null=True, blank=True, help_text="Last 6 confidence scores")

    def __str__(self):
        return f"{self.coin.symbol} | {self.trade_type.upper()} | {self.entry_timestamp.strftime('%Y-%m-%d %H:%M')}"



class ConfidenceHistory(models.Model):
    coin = models.ForeignKey(Coin, on_delete=models.CASCADE)
    model_name = models.CharField(max_length=100)
    timestamp = models.DateTimeField(auto_now_add=True)
    confidence = models.FloatField()

    class Meta:
        ordering = ['-timestamp']




class RealTrade(models.Model):

    coin = models.ForeignKey(Coin, on_delete=models.CASCADE)
    trade_type = models.CharField(max_length=10)
    entry_timestamp = models.DateTimeField()
    exit_timestamp = models.DateTimeField(null=True, blank=True)
    entry_price = models.DecimalField(max_digits=20, decimal_places=10)
    exit_price = models.DecimalField(max_digits=20, decimal_places=10, null=True, blank=True)
    model_confidence = models.FloatField()
    take_profit_percent = models.FloatField()
    stop_loss_percent = models.FloatField()
    duration_minutes = models.IntegerField(null=True, blank=True)
    result = models.BooleanField(null=True, blank=True)
    exit_timestamp = models.DateTimeField(null=True, blank=True)
    entry_usd_amount = models.DecimalField(max_digits=20, decimal_places=10)
    exit_usd_amount = models.DecimalField(max_digits=20, decimal_places=10, null=True, blank=True)
    account_balance_before = models.DecimalField(max_digits=20, decimal_places=10)
    account_balance_after = models.DecimalField(max_digits=20, decimal_places=10, null=True, blank=True)

    def __str__(self):
        return f"{self.coin.symbol} | {self.trade_type.upper()} | {self.entry_timestamp.strftime('%Y-%m-%d %H:%M')}"













class LiveModelMetrics(models.Model):
    coin = models.ForeignKey(Coin, on_delete=models.CASCADE)
    timestamp = models.DateTimeField()

    # Core fields
    open = models.DecimalField(max_digits=20, decimal_places=10)
    high = models.DecimalField(max_digits=20, decimal_places=10)
    low = models.DecimalField(max_digits=20, decimal_places=10)
    close = models.DecimalField(max_digits=20, decimal_places=10)
    volume = models.DecimalField(max_digits=30, decimal_places=2)

    # All the enriched metrics you just trained on:
    sma_5 = models.DecimalField(max_digits=20, decimal_places=10, null=True)
    sma_20 = models.DecimalField(max_digits=20, decimal_places=10, null=True)
    ema_12 = models.DecimalField(max_digits=20, decimal_places=10, null=True)
    ema_26 = models.DecimalField(max_digits=20, decimal_places=10, null=True)
    ema_crossover_flag = models.BooleanField(null=True)
    rsi = models.FloatField(null=True)
    macd = models.FloatField(null=True)
    macd_signal = models.FloatField(null=True)
    stochastic_k = models.FloatField(null=True)
    stochastic_d = models.FloatField(null=True)
    bollinger_upper = models.FloatField(null=True)
    bollinger_middle = models.FloatField(null=True)
    bollinger_lower = models.FloatField(null=True)
    adx = models.FloatField(null=True)
    atr_1h = models.DecimalField(max_digits=20, decimal_places=10, null=True)
    stddev_1h = models.FloatField(null=True)
    momentum_10 = models.FloatField(null=True)
    momentum_50 = models.FloatField(null=True)
    roc = models.FloatField(null=True)
    rolling_volatility_5h = models.FloatField(null=True)
    rolling_volatility_24h = models.FloatField(null=True)
    high_low_ratio = models.FloatField(null=True)
    price_position = models.FloatField(null=True)
    candle_body_size = models.FloatField(null=True)
    candle_body_pct = models.FloatField(null=True)
    wick_upper = models.FloatField(null=True)
    wick_lower = models.FloatField(null=True)
    slope_5h = models.FloatField(null=True)
    slope_24h = models.FloatField(null=True)
    trend_acceleration = models.FloatField(null=True)
    fib_distance_0_236 = models.FloatField(null=True)
    fib_distance_0_382 = models.FloatField(null=True)
    fib_distance_0_618 = models.FloatField(null=True)
    vwap = models.FloatField(null=True)
    volume_price_ratio = models.FloatField(null=True)
    volume_change_5m = models.FloatField(null=True)
    volume_surge = models.FloatField(null=True)
    overbought_rsi = models.BooleanField(null=True)
    oversold_rsi = models.BooleanField(null=True)
    upper_bollinger_break = models.BooleanField(null=True)
    lower_bollinger_break = models.BooleanField(null=True)
    atr_normalized = models.FloatField(null=True)
    short_vs_long_strength = models.FloatField(null=True)

    class Meta:
        unique_together = ('coin', 'timestamp')



#
