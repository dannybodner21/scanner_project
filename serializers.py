from rest_framework import serializers
from .models import RealTrade

class RealTradeLiveSerializer(serializers.ModelSerializer):
    coin = serializers.CharField(source='coin.symbol')
    pnl = serializers.SerializerMethodField()
    current = serializers.SerializerMethodField()

    class Meta:
        model = RealTrade
        fields = ['coin', 'trade_type', 'entry_price', 'current', 'pnl']

    def get_pnl(self, obj):
        # No exit price, so return 0 or placeholder
        return 0.0

    def get_current(self, obj):
        # Return entry price as current until live price integration
        return float(obj.entry_price)
