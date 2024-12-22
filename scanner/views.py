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
from scanner.tasks import setup_schedule
from django.http import HttpResponse



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


def gather_historical_data():

    API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
    BASE_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"

    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": API_KEY,
    }

    coins = Coin.objects.all()

    # put all the coins in a group of ten because of api call limits
    coins_in_group_of_ten = []
    coin_group = []
    count = 0

    for coin in coins:

        if count < 10:
            coin_group.append(coin)
            count += 1

        else:

            count = 0
            coins_in_group_of_ten.append(coin_group)
            coin_group = []


    #coins = Coin.objects.all().order_by("market_cap_rank")[11:20]

    for coin_group in coins_in_group_of_ten:

        for coin in coin_group:
            try:

                # Use LA timezone - using UCT time now
                #la_timezone = ZoneInfo("America/Los_Angeles")
                end_time = datetime.now()
                start_time = end_time - timedelta(days=30)

                params = {
                    "symbol": coin.symbol,
                    "time_start": start_time.isoformat(),
                    "time_end": end_time.isoformat(),
                    "interval": "1d",
                }

                response = requests.get(BASE_URL, headers=headers, params=params)
                data = response.json()

                if "data" in data and "quotes" in data["data"]:
                    for quote in data["data"]["quotes"]:
                        HistoricalData.objects.update_or_create(
                            coin=coin,
                            date=quote["timestamp"].split("T")[0],
                            defaults={
                                "price": quote["quote"]["USD"]["price"],
                                "volume_24h": quote["quote"]["USD"]["volume_24h"],
                            },
                        )
                else:
                    print("FAILED TO FETCH")
                    print(coin.symbol)

                    HistoricalData.objects.update_or_create(
                        coin=coin,
                        date=end_time,
                        defaults={
                            "price": 1,
                            "volume_24h": 1,
                        },
                    )

                import time

                # Pause for 60 seconds
                time.sleep(60)

            except Exception as e:
                print(f"Error fetching historical data for {coin.symbol}: {e}")


def fetch_short_interval_data():

    API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
    BASE_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"

    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": API_KEY,
    }

    coins = Coin.objects.all()
    #coins = Coin.objects.order_by('market_cap_rank')[:200]

    # put all the coins in a group of five because of api call limits
    coins_in_group_of_five = []
    coin_group = []
    count = 0

    for coin in coins:

        if count < 5:
            coin_group.append(coin)
            count += 1

        else:
            count = 1
            coins_in_group_of_five.append(coin_group)
            coin_group = []
            coin_group.append(coin)


    for coin_group in coins_in_group_of_five:
        for coin in coin_group:
            try:
                #la_timezone = ZoneInfo("America/Los_Angeles")
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=24)

                params = {
                    "symbol": coin.symbol,
                    "time_start": start_time.isoformat(),
                    "time_end": end_time.isoformat(),
                    "interval": "5m",
                }

                response = requests.get(BASE_URL, headers=headers, params=params)
                data = response.json()

                if "data" in data and "quotes" in data["data"]:
                    for quote in data["data"]["quotes"]:
                        ShortIntervalData.objects.update_or_create(
                            coin=coin,
                            timestamp=quote["timestamp"],
                            defaults={
                                "price": quote["quote"]["USD"]["price"],
                                "volume_5min": quote["quote"]["USD"]["volume_24h"],  # Volume for the interval
                            },
                        )

                else:
                    print('==============')
                    print(' short term data ')
                    print(coin.symbol)
                    print(data)

                    ShortIntervalData.objects.update_or_create(
                        coin=coin,
                        timestamp=end_time,
                        defaults={
                            "price": 1,
                            "volume_5min": 1,
                        },
                    )

            except Exception as e:
                print(f"Error fetching short interval data for {coin.symbol}: {e}")

        import time

        # Pause for 60 seconds
        print("pausing for 60 seconds")
        time.sleep(60)
        print("resuming")


def load_coins():

    API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'

    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": API_KEY,
    }

    params = {
        "start": "1",  # Start at the first cryptocurrency
        "limit": "200",  # Fetch the top 200 cryptocurrencies
        "convert": "USD",  # Convert prices to USD
    }

    try:

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        # Extract relevant information
        top_cryptos = []
        for crypto in data["data"]:

            Coin.objects.update_or_create(
                symbol=crypto["symbol"],
                defaults={
                    "name": crypto["name"],
                    "market_cap_rank": crypto["cmc_rank"],
                    "last_updated": datetime.strptime(
                        crypto["last_updated"], "%Y-%m-%dT%H:%M:%S.%fZ"
                    ),
                },
            )

        print("Top coins fetched and updated successfully.")

    except Exception as e:
        print(f"Error fetching data: {e}")


# once a day delete unneeded data from database
def delete_old_data():

    # delete any data from HistoricalData that is older than 30 days
    #threshold_date = now() - timedelta(days=30)
    #deleted_count, _ = HistoricalData.objects.filter(date__lt=threshold_date).delete()
    #print(f"Deleted {deleted_count} old records from HistoricalData.")

    # delete any data from ShortIntervalData that is older than 30 days
    threshold_date = now() - timedelta(days=30)
    deleted_count, _ = ShortIntervalData.objects.filter(timestamp__lt=threshold_date).delete()
    print(f"Deleted {deleted_count} old records from ShortIntervalData.")

    # delete any data from Metrics that is older than 24 hours
    #threshold_date = now() - timedelta(hours=24)
    #deleted_count, _ = Metrics.objects.filter(timestamp__lt=threshold_date).delete()
    #print(f"Deleted {deleted_count} old records from Metrics.")


def calculate_relative_volume(coin):

    # volume over 24 hours / average 24 hour volume

    relative_volume = 1

    try:

        current_time = now()
        threshold_time_24h = current_time - timedelta(hours=24)
        threshold_time_30d = current_time - timedelta(days=30)

        # Query the 5-minute volumes for the last 24 hours
        #last_24_hours_data = ShortIntervalData.objects.filter(coin=coin, timestamp__gte=threshold_time_24h)

        latest_interval = ShortIntervalData.objects.filter(coin=coin).order_by('-timestamp').first()

        # Query the 5-minute volumes for the last 30 days
        last_30_days_data = ShortIntervalData.objects.filter(coin=coin, timestamp__gte=threshold_time_30d)

        sum_volume_30_days = sum(data.volume_5min for data in last_30_days_data)

        if last_30_days_data != 0:
            average_volume_30_days = sum_volume_30_days / len(last_30_days_data)

        else:
            average_volume_30_days = 1
            return average_volume_30_days

        # Calculate relative volume
        if average_volume_30_days != 0:
            relative_volume = latest_interval.volume_5min / average_volume_30_days

        else:
            relative_volume = 1
            print("Couldn't calculate relative volume")
            print(coin.symbol)

    except Exception as e:
        print(f"There was a problem calculating relative volume V2 for: {e}")
        print(coin.symbol)

    return relative_volume


def calculate_price_change_five_min(coin):

    # (price change over 5 min / price 5 min ago) * 100

    price_change_5min = 1

    # Get the current time
    current_time = now()
    time_5_min_ago = current_time - timedelta(minutes=5)

    # Get the latest price
    current_data = ShortIntervalData.objects.filter(coin=coin).order_by('-timestamp').first()

    # Get the price 5 minutes ago
    data_5_min_ago = ShortIntervalData.objects.filter(coin=coin, timestamp__lte=time_5_min_ago).order_by('-timestamp').first()

    if current_data and data_5_min_ago:
        current_price = current_data.price
        price_5_min_ago = data_5_min_ago.price

        # Calculate price change
        price_change = current_price - price_5_min_ago

        # Calculate percentage change
        price_change_5min = (price_change / price_5_min_ago) * 100 if price_5_min_ago != 0 else 1

    return price_change_5min


def calculate_price_change_ten_min(coin):

    # (price change over 10 min / price 10 min ago) * 100

    price_change_10min = 1

    # Get the current time
    current_time = now()
    time_10_min_ago = current_time - timedelta(minutes=10)

    # Get the latest price
    current_data = ShortIntervalData.objects.filter(coin=coin).order_by('-timestamp').first()

    # Get the price 10 minutes ago
    data_10_min_ago = ShortIntervalData.objects.filter(coin=coin, timestamp__lte=time_10_min_ago).order_by('-timestamp').first()

    if current_data and data_10_min_ago:
        current_price = current_data.price
        price_10_min_ago = data_10_min_ago.price

        # Calculate price change
        price_change = current_price - price_10_min_ago

        # Calculate percentage change
        price_change_10min = (price_change / price_10_min_ago) * 100 if price_10_min_ago != 0 else 1

    return price_change_10min


def five_min_update():

    # every 5 minutes run this function
    # it will go through all the coins in our database
    # in Coin - update market_cap_rank and last_updated
    # in ShortIntervalData - update timestamp, price, volume_5min
    # in Metrics - update timestamp, ...


    # if the time is 0000 delete old data
    now = datetime.now()
    if now.hour == 0 and now.minute < 6:
        delete_old_data()


    API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'

    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": API_KEY,
    }

    params = {
        "start": "1",  # Start at the first cryptocurrency
        "limit": "200",  # Fetch the top 200 cryptocurrencies
        "convert": "USD",  # Convert prices to USD
    }


    try:

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        # Extract relevant information
        top_cryptos = []
        for crypto in data["data"]:

            coin = Coin.objects.get(symbol=crypto["symbol"])

            #print(coin.name)

            ShortIntervalData.objects.create(
                coin=coin,
                timestamp=datetime.strptime(
                    crypto["last_updated"], "%Y-%m-%dT%H:%M:%S.%fZ"
                ),
                price=crypto["quote"]["USD"]["price"],
                volume_5min=crypto["quote"]["USD"]["volume_24h"]
            )

            Metrics.objects.filter(coin=coin).delete()
            metric = Metrics.objects.create(
                coin=coin,
                timestamp=datetime.strptime(
                    crypto["last_updated"], "%Y-%m-%dT%H:%M:%S.%fZ"
                ),
                relative_volume=calculate_relative_volume(coin),
                price_change_5min=calculate_price_change_five_min(coin),
                price_change_10min=calculate_price_change_ten_min(coin),
                price_change_1hr=crypto["quote"]["USD"]["percent_change_1h"],
                price_change_24hr=crypto["quote"]["USD"]["percent_change_24h"],
                price_change_7d=crypto["quote"]["USD"]["percent_change_7d"],
                circulating_supply=crypto["circulating_supply"],
                volume_24h=crypto["quote"]["USD"]["volume_24h"],
                last_price=crypto["quote"]["USD"]["price"],
                market_cap=crypto["quote"]["USD"]["market_cap"]
            )

        print("Update complete.")

    except Exception as e:
        print(f"Coin probably not found - Error fetching data: {e}")


def index(request):

    # get all the coins in our databse
    top_cryptos = []
    #coins = Coin.objects.order_by('market_cap_rank')[:50]
    coins = Coin.objects.all()

    for coin in coins:

        shortIntervalData = ShortIntervalData.objects.filter(coin=coin).order_by("-timestamp").first()
        metric = Metrics.objects.filter(coin=coin).order_by("-timestamp").first()

        # Go through the different triggers

        # TRIGGER ONE - price up 10% or more in last 24 hours
        triggerOne = False
        if metric.price_change_24hr >= 10:
            triggerOne = True

        # TRIGGER TWO - relative volume is 2.0 or higher
        triggerTwo = False
        if metric.relative_volume >= 2.0:
            triggerTwo = True

        # TRIGGER THREE - price is less than $50
        triggerThree = False
        if shortIntervalData.price < 50:
            triggerThree = True

        # TRIGGER FOUR - circulating supply is less than 100 million
        triggerFour = False
        if metric.circulating_supply < 100000000:
            triggerFour = True

        # TRIGGER FIVE - market cap between $10M and $1B
        triggerFive = False
        if metric.market_cap > 10000000 and metric.market_cap < 1000000000:
            triggerFive = True

        # TRIGGER SIX - volume today is at least 200000
        triggerSix = False
        if metric.volume_24h >= 200000:
            triggerSix = True

        try:
            top_cryptos.append({
                "time": coin.last_updated,
                "name": coin.name,
                "symbol": coin.symbol,
                "price": shortIntervalData.price,
                "market_cap": metric.market_cap,
                "volume_24h_USD": metric.volume_24h,
                #"volume_24h_percentage": 1,
                "price_change_1h": metric.price_change_1hr,
                "price_change_24h_percentage": metric.price_change_24hr,
                "price_change_7d": metric.price_change_7d,
                "circulating_supply": metric.circulating_supply,
                "relative_volume": metric.relative_volume,
                #"five_min_relative_volume": ,
                "price_change_5min": metric.price_change_5min,
                "price_change_10min": metric.price_change_10min,
                "triggerOne": triggerOne,
                "triggerTwo": triggerTwo,
                "triggerThree": triggerThree,
                "triggerFour": triggerFour,
                "triggerFive": triggerFive,
                "triggerSix": triggerSix,
            })

        except:
            print("Couldn't fetch all the data...")

    # sort by top gainers: price change over the last 24 hours
    sorted_coins = sorted(top_cryptos, key=lambda x: x["price_change_24h_percentage"], reverse=True)

    # Render data to the HTML template
    return render(request, "index.html", {"top_cryptos": sorted_coins})



def setup_scheduler(request):
    setup_schedule()
    return HttpResponse("Schedule created successfully!")
