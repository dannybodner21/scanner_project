from django.contrib import admin
from scanner.models import Coin, RealTrade, MemoryTrade, LiveChart, LivePriceSnapshot, ModelTrade, RickisMetrics, BacktestResult, SuccessfulMove, FiredSignal, SupportResistance, Pattern, HighLowData, HistoricalData, ShortIntervalData, Metrics, Trigger
from import_export.admin import ExportMixin
from django.utils.html import format_html

admin.site.register(Coin)
admin.site.register(HistoricalData)
admin.site.register(ShortIntervalData)
admin.site.register(Trigger)
admin.site.register(HighLowData)
admin.site.register(Pattern)
admin.site.register(SupportResistance)
admin.site.register(RickisMetrics)
admin.site.register(ModelTrade)
admin.site.register(RealTrade)
admin.site.register(LivePriceSnapshot)
admin.site.register(MemoryTrade)


@admin.register(BacktestResult)
class BacktestResultAdmin(admin.ModelAdmin):
    list_display = ("coin", "timestamp", "entry_price", "success")
    raw_id_fields = ("coin", "entry_metrics")  # ✅ Critical

@admin.register(Metrics)
class MetricsAdmin(admin.ModelAdmin):
    list_display = (
        'coin', 'timestamp', 'last_price', 'price_change_5min',
        'price_change_10min', 'price_change_1hr', 'price_change_24hr',
        'price_change_7d', 'five_min_relative_volume',
        'rolling_relative_volume', 'twenty_min_relative_volume', 'volume_24h',
        'volatility_5min', 'volume_marketcap_ratio', 'trend_slope_30min',
        'change_since_low', 'change_since_high')
    raw_id_fields = ("coin",)

@admin.register(FiredSignal)
class FiredSignalAdmin(admin.ModelAdmin):
    list_display = ("coin", "fired_at", "result")
    raw_id_fields = ("coin",)

@admin.register(SuccessfulMove)
class SuccessfulMoveAdmin(admin.ModelAdmin):
    list_display = ("coin", "timestamp", "move_type", "entry_price")
    raw_id_fields = ("coin", "entry_metrics")

@admin.register(LiveChart)
class LiveChartAdmin(admin.ModelAdmin):
    list_display = ('coin', 'timestamp', 'chart_preview')

    def chart_preview(self, obj):
        if obj.image:
            return format_html('<img src="{}" width="200"/>', obj.image.url)
        return "-"
    chart_preview.short_description = "Chart Image"
