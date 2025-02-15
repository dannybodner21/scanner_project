from django.contrib import admin
from scanner.models import Coin, TriggerCombination, SupportResistance, Pattern, HighLowData, HistoricalData, ShortIntervalData, Metrics, Trigger

admin.site.register(Coin)
admin.site.register(HistoricalData)
admin.site.register(ShortIntervalData)
admin.site.register(Metrics)
admin.site.register(Trigger)
admin.site.register(HighLowData)
admin.site.register(Pattern)
admin.site.register(SupportResistance)
admin.site.register(TriggerCombination)
