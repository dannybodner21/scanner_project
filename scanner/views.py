import json
import time
import urllib.request
import os
import requests
import asyncio

from django.shortcuts import render
from zoneinfo import ZoneInfo
from django.http import HttpResponseRedirect
from scanner.models import Coin, HistoricalData, ShortIntervalData, Metrics
from datetime import datetime, timedelta, timezone
from django.utils.timezone import now
from django.http import JsonResponse


def create_temporary_data():

    target_time = "2024-11-23 16:00:00"
    target_datetime = datetime.strptime(target_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    # Step 1: Create Bitcoin Entry
    bitcoin, created = Coin.objects.get_or_create(
        name="BitcoinTest",
        symbol="BTCTEST",
        market_cap_rank=1
    )
    if created:
        print("Bitcoin test coin entry created.")
    else:
        print("Bitcoin test coin entry already exists.")

    # Step 2: Populate Historical Data (30 Days)
    historical_data = []
    start_date = target_datetime - timedelta(days=30)
    for i in range(30):
        date = start_date + timedelta(days=i)
        price = 30000 + (i * 100)  # Simulated price
        volume_24h = 1000000000 + (i * 5000000)  # Simulated volume
        historical_data.append(
            HistoricalData(coin=bitcoin, date=date, price=price, volume_24h=volume_24h)
        )
    HistoricalData.objects.bulk_create(historical_data, ignore_conflicts=True)
    print("Historical data populated.")

    # Step 3: Populate 5-Minute Interval Data (24 Hours)
    short_interval_data = []
    start_time = target_datetime - timedelta(hours=24)
    for i in range(288):  # 24 hours * 60 minutes / 5 minutes
        timestamp = start_time + timedelta(minutes=5 * i)
        price = 31000 + (i * 10)
        volume_5min = 5000000 + (i * 10000)
        short_interval_data.append(
            ShortIntervalData(coin=bitcoin, timestamp=timestamp, price=price, volume_5min=volume_5min)
        )
    ShortIntervalData.objects.bulk_create(short_interval_data, ignore_conflicts=True)
    print("Short interval data populated.")

    # Step 4: Populate Metrics
    relative_volume = 1.5
    price_change_24hr = ((31000 - 30000) / 30000) * 100
    price_change_5min = ((31050 - 31000) / 31000) * 100
    price_change_10min = ((31100 - 31000) / 31000) * 100
    volume_24h = 1500000000
    circulating_supply = 19000000
    market_cap = circulating_supply * 31000

    Metrics.objects.create(
        coin=bitcoin,
        timestamp=target_datetime,
        relative_volume=relative_volume,
        price_change_24hr=price_change_24hr,
        price_change_5min=price_change_5min,
        price_change_10min=price_change_10min,
        circulating_supply=circulating_supply,
        volume_24h=volume_24h,
        last_price=31000,
        market_cap=market_cap
    )

    coin = Coin.objects.get(symbol="BTCTEST")
    coin.last_updated = now()
    coin.save()

    print("Metrics data populated.")


def index(request):

    create_temporary_data()

    top_cryptos = []
    coin = Coin.objects.get(symbol="BTCTEST")
    metrics = coin.metrics.all()

    metric = metrics[0]

    current_time = coin.last_updated

    top_cryptos.append({
        "time": current_time,
        "name": coin.name,
        "symbol": coin.symbol,
        "price": metric.last_price,
        "market_cap": metric.market_cap,
        "volume_24h_USD": metric.volume_24h,
        "volume_24h_percentage": 0000000,
        "price_change_24h_percentage": metric.price_change_24hr,
        "circulating_supply": metric.circulating_supply,
        "relative_volume": metric.relative_volume,
        "five_min_relative_volume": 00000000,
        "price_change_5min": metric.price_change_5min,
        "price_change_10min": metric.price_change_10min,

        "triggerOne": False,
        "triggerTwo": False,
        "triggerThree": False,
        "triggerFour": False,
        "triggerFive": False,
        "triggerSix": False,
        "triggerSeven": False,
    })

    # Render data to the HTML template
    return render(request, "index.html", {"top_cryptos": top_cryptos})
