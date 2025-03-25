from django.contrib import admin
from scanner.models import Coin, SuccessfulMove, FiredSignal, SupportResistance, Pattern, HighLowData, HistoricalData, ShortIntervalData, Metrics, Trigger
from import_export.admin import ExportMixin

admin.site.register(Coin)
admin.site.register(HistoricalData)
admin.site.register(ShortIntervalData)
admin.site.register(Trigger)
admin.site.register(HighLowData)
admin.site.register(Pattern)
admin.site.register(SupportResistance)
admin.site.register(FiredSignal)

@admin.register(Metrics)
class MetricsAdmin(ExportMixin, admin.ModelAdmin):
    list_display = (
        'coin', 'timestamp', 'last_price', 'price_change_5min',
        'price_change_10min', 'price_change_1hr', 'price_change_24hr',
        'price_change_7d', 'five_min_relative_volume',
        'rolling_relative_volume', 'twenty_min_relative_volume', 'volume_24h')
    list_filter = ('coin',)
    search_fields = ('coin__symbol',)

@admin.register(SuccessfulMove)
class SuccessfulMoveAdmin(ExportMixin, admin.ModelAdmin):
    list_display = ('coin', 'timestamp', 'entry_price', 'move_type','metrics')
    list_filter = ('coin',)
    search_fields = ('coin__symbol',)
