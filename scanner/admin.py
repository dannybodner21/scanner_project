from django.contrib import admin
from scanner.models import Coin, Pattern, HighLowData, HistoricalData, ShortIntervalData, Metrics, MemeCoin, MemeMetric, MemeShortIntervalData, Trigger

admin.site.register(Coin)
admin.site.register(HistoricalData)
admin.site.register(ShortIntervalData)
admin.site.register(Metrics)
admin.site.register(MemeCoin)
admin.site.register(MemeMetric)
admin.site.register(MemeShortIntervalData)
admin.site.register(Trigger)
admin.site.register(HighLowData)
admin.site.register(Pattern)
