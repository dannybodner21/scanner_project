from django.core.management.base import BaseCommand
from scanner.models import Coin, ShortIntervalData
from django.utils.timezone import make_aware
from datetime import datetime, timedelta

class Command(BaseCommand):
    help = "Check number of ShortIntervalData entries per coin for a specific UTC day"

    def add_arguments(self, parser):
        parser.add_argument('--date', type=str, help='Date in YYYY-MM-DD format (UTC)')

    def handle(self, *args, **kwargs):
        date_str = kwargs['date']
        if not date_str:
            self.stdout.write(self.style.ERROR("Please provide a --date argument (e.g., 2024-03-30)"))
            return

        start = make_aware(datetime.strptime(date_str, "%Y-%m-%d"))
        end = start + timedelta(days=1)

        target_symbols = ["BTC","ETH","XRP","BNB","SOL","TRX","DOGE","ADA","LEO","LINK",
                          "AVAX","XLM","TON","SHIB","SUI","HBAR","BCH","DOT","LTC",
                          "HYPE","BGB","DAI","PI","XMR","UNI","PEPE","OKB","APT","GT",
                          "NEAR","ONDO","TAO","ICP","ETC","RENDER","MNT","KAS","CRO",
                          "AAVE","POL","VET","FIL","TRUMP","ALGO","ENA","ATOM","TIA",
                          "FET","ARB","S","KCS","DEXE","OP","JUP","MKR","XDC","STX",
                          "FLR","EOS","WLD","IP","BONK","FARTCOIN","SEI","INJ","IMX",
                          "GRT","FORM","QNT","PAXG","CRV","JASMY","SAND","GALA",
                          "NEXO","CORE","RAY","KAIA","LDO","THETA","IOTA","HNT",
                          "MANA","FLOW","CAKE","MOVE","FLOKI","XCN"]

        coins = Coin.objects.filter(symbol__in=target_symbols)

        for coin in coins:
            count = ShortIntervalData.objects.filter(
                coin=coin,
                timestamp__gte=start,
                timestamp__lt=end
            ).count()
            print(f"{coin.symbol}: {count} entries on {date_str}")
