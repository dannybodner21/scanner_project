from django.contrib import admin
from scanner.models import Coin, HistoricalData, ShortIntervalData, Metrics

admin.site.register(Coin)
admin.site.register(HistoricalData)
admin.site.register(ShortIntervalData)
admin.site.register(Metrics)
