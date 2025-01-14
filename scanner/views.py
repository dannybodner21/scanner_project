import json
import time
import urllib.request
import os
import requests
import asyncio

from django.shortcuts import render
from zoneinfo import ZoneInfo
from django.http import HttpResponseRedirect
from scanner.models import Coin, HistoricalData, ShortIntervalData, Metrics, MemeCoin, MemeMetric, MemeShortIntervalData
from datetime import datetime, timedelta, timezone
from django.utils.timezone import now
from django.http import JsonResponse
from django.http import HttpResponse




# bot message notificagtions
def send_text(true_triggers_two):

    if len(true_triggers_two) > 0:

        # telegram bot information
        chat_id = '1077594551'
        #chat_id_ricki = '1054741134,'
        #chat_ids = [chat_id, chat_id_ricki]
        chat_ids = [chat_id]
        bot_token = '7672687080:AAFWvkwzp-LQE92XdO9vcVa5yWJDUxO17yE'
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"


    # send message to myself and Ricki
    message = ""
    for chat_id in chat_ids:
        for trigger in true_triggers_two:

            message += trigger + " "

            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown",
            }

            response = requests.post(url, data=payload)

            if response.status_code == 200:
                print("Message sent successfully.")
            else:
                print(f"Failed to send message: {response.content}")

    return


def fetch_short_interval_data():

    API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
    BASE_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"

    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": API_KEY,
    }

    coins = Coin.objects.all()
    #coins = Coin.objects.order_by('market_cap_rank')[:500]

    # put all the coins in a group of 15 because of api call limits
    coins_in_group_of_fifteen = []
    coin_group = []
    count = 0

    for coin in coins:

        if count < 20:
            coin_group.append(coin)
            count += 1

        else:
            count = 1
            coins_in_group_of_fifteen.append(coin_group)
            coin_group = []
            coin_group.append(coin)


    for coin_group in coins_in_group_of_fifteen:
        for coin in coin_group:
            try:
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=24)

                params = {
                    "id": coin.cmc_id,
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
                                "volume_5min": quote["quote"]["USD"]["volume_24h"],
                                "circulating_supply": quote["quote"]["USD"]["circulating_supply"]
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
                            "price": None,
                            "volume_5min": None,
                        },
                    )

            except Exception as e:
                print(f"Error fetching short interval data for {coin.symbol}: {e}")

        # Pause for 60 seconds
        print("pausing for 60 seconds")
        time.sleep(60)
        print("resuming")


def load_coins():

    API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'

    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": API_KEY,
    }

    specific_coins = ["PHA","GMT","BAN","BLUE","FWOG","VIRTUAL","TOKEN","W","ORDER","NS","FORTH","GRASS","ACX","F","SSV","THE","USTC","SUN","MERL","PUFFER","COOK","MORPHO","L3","ZBCN","FIRE","AI","HIPPO","DAR","POWRBOBA","SUNDOG","LUNA2","ETHFI","REZ","REN","LDO","MDT","AUDIO","FXS","1CAT","PENDLE","XRD","STEEM","TOSHI","AIDOGE","UNI","PORTAL","PEOPLE","LUNC","CLOUD","DYDX","GIGA","XEM","ID","PRIME","ETHW","EDU","PHB","CATI","MEW","APE","RPL","LEVER","XCN","PEAQ","JST","BONK","REEF","ICX","XAI","PIXEL","TRX","CATS","IOTX","BOME","POPCAT","DOP1","RATS","OGN","JTO","IMX","MAVIA","MAV","UXLINK","CTC","METIS","HNT","MAJOR","LIT","BENDOG","KNC","BRETT","MYRO","KAVA","FARTCOIN","OP","KDA","MKR","GMX","ASTR","IO","SATS","REQ","GODS","MOCA","DBR","RAYDIUM","AIOZ","SYN","ZRX","MANEKI","SWELL","UMA","IDEX","LINA","BADGER","PEIPEI","RUNE","SNT","SYS","ZRC","GLM","ENS","STMX","CFX","KEY","BAKE","BNX","OM","AUCTION","GAS","SAGA","VOXEL","LISTA","LSK","SC","OL","KAS","TNSR","ALT","SCRT","FLOKI","MTL","AMB","ARB","CORE","XMR","ORDI","STRK","RSS3","BCH","CYBER","ALPHA","CRV","RARE","DODO","YGG","MEME","VRA","ONG","NFP","LAI","NYAN","SPELL","ARK","BIGTIME","POLYX","OMNI","WOO","HOT","PERP","ACH","DYM","FLOW","BICO","ADA","C98","HIFI","MAGIC","CTK","BSW","ARPA","BLUR","DATA","ZETA","AR","CVX","COMBO","FLUX","SXP","AXS","MINA","WLD","DOGS","CHILLGUY","MASK","FIDA","TLM","BANANA","DOG","JOE","HOOK","CAKE","QI","COS","TRB","XVS","MANTA","NULS","DEGEN","A8","CELO","AVAIL","API3","NTRN","RDNT","YFI","NOT","EIGEN","SLF","SNX","MNT","FTN","POL","CVC","WAXP","CKB","SILLY","BAL","FLM","RIF","ETC","ORBS","CHZ","SLERF","IOTA","ZIL","NEO","OXT","MBL","STX","KSM","1INCH","ILV","MAX","RON","VANRY","CRO","ACE","TAI","AGLD","NEAR","EGLD","T","ANKR","ZK","NKN","GTC","CTSI","NMR","PYTH","CHESS","TON","BNB","XION","ALICE","ARKM","PAXG","ONT","QTUM","FOXY","OMG","OSMO","TAO","NEIROETH","HFT","MANA","GLMR","ROSE","TWT","QUICK","RVN","IOST","SKL","AEVO","ETH","SEP","WAVES","WIF","THETA","COMP","BEL","STORJ","EOS","LRC","GRIFFAIN","GRT","ATOM","GALA","SEND","COTI","AGI","ENJ","G","HMSTR","DENT","DUSK","RSR","CHR","BAND","FIL","XRP","DOGE","KAIA","TRU","DOT","SLP","BSV","TAIKO","STG","VTHO","MOVR","ONDO","BTC","LUMIA","FB","LOOKS","CELR","DGB","SUSHI","LTC","AXL","BEAM","SAND","SEI","MYRIA","ENA","XTZ","LINK","VELODROME","INJ","APT","SOL","TIA","ICP","KMNO","AKT","RENDER","LUCE","VET","AVAX","XLM","SUI","STPT","MOBILE","BLAST","PNUT","SPEC","RAD","BAT","SUPER","ACT","JUP","SAFE","ALEO","PIRATE","FTM","DASH","ZRO","CETUS","ALGO","AAVE","TROY","ONE","XNO","DEEP","ZEUS","MOODENG","HBAR","PRCL","CARV","ATH","JASMY","GEMS","GME","GOAT","AIXBT","LQTY","MON","DRIFT","XVG","MOVE","PENGU","ZEC","SPX","LPT","MOTHER","COW","VELO","ZEN","URO","RIFSOL","DEXE","MASA","PEPE","BTT","XEC","SHIB","LADYS","X","BABYDOGE","NEIROCTO","WEN","MOG","CAT","TURBO"]


    chunk_size = 100

    for i in range(0, len(specific_coins), chunk_size):

        symbol_batch = specific_coins[i:i + chunk_size]
        params = {
            "symbol": ",".join(symbol_batch),
            "convert": "USD",
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            #data = response.json()

            data = response.json().get("data", {})

            top_cryptos = []

            for symbol in specific_coins:
                coin_data = data.get(symbol)
                if coin_data:
                    name = coin_data.get("name")
                    price = coin_data["quote"]["USD"].get("price")
                    market_cap = coin_data["quote"]["USD"].get("market_cap")

                    Coin.objects.update_or_create(
                        name=name,
                        defaults={
                            "symbol": coin_data.get("symbol"),
                            "market_cap_rank": coin_data.get("cmc_rank"),
                            "last_updated": datetime.strptime(
                                coin_data.get("last_updated"), "%Y-%m-%dT%H:%M:%S.%fZ"
                            ),
                            "cmc_id": coin_data.get("id"),
                        },
                    )

            print("Top coins fetched and updated successfully.")

        except Exception as e:
            print(f"Error fetching data: {e}")


# once a day delete unneeded data from database
def delete_old_data():

    # delete any data from ShortIntervalData that is older than 30 days
    threshold_date = now() - timedelta(days=30)
    deleted_count, _ = ShortIntervalData.objects.filter(timestamp__lt=threshold_date).delete()
    print(f"Deleted {deleted_count} old records from ShortIntervalData.")


def calculate_daily_relative_volume(coin):

    relative_volume = None

    try:
        current_time = now()
        threshold_time_30d = current_time - timedelta(days=30)

        latest_interval = HistoricalData.objects.filter(coin=coin).order_by('-date').first()
        last_30_days_data = HistoricalData.objects.filter(coin=coin, date__gte=threshold_time_30d)

        remaining_metrics = last_30_days_data[1:]

        sum_volume_30_days = sum(data.volume_24h for data in remaining_metrics)

        if len(remaining_metrics) != 0:
            average_volume_30_days = sum_volume_30_days / len(remaining_metrics)

        else:
            average_volume_30_days = None
            return average_volume_30_days

        # Calculate relative volume
        if average_volume_30_days != 0:
            relative_volume = latest_interval.volume_24h / average_volume_30_days

    except Exception as e:
        print(f"There was a problem calculating daily relative volume: {e}")
        print(coin.symbol)

    return relative_volume


def calculate_relative_volume(coin):

    # volume over 24 hours / average 24 hour volume

    relative_volume = None

    try:

        current_time = now()
        threshold_time_24h = current_time - timedelta(hours=24)
        threshold_time_30d = current_time - timedelta(days=30)

        latest_interval = ShortIntervalData.objects.filter(coin=coin).order_by('-timestamp').first()
        last_30_days_data = ShortIntervalData.objects.filter(coin=coin, timestamp__gte=threshold_time_30d)

        remaining_metrics = last_30_days_data[1:]

        sum_volume_30_days = sum(data.volume_5min for data in remaining_metrics)

        if len(remaining_metrics) != 0:
            average_volume_30_days = sum_volume_30_days / len(remaining_metrics)

        else:
            average_volume_30_days = None
            return average_volume_30_days

        # Calculate relative volume
        if average_volume_30_days != 0:
            relative_volume = latest_interval.volume_5min / average_volume_30_days

        else:
            relative_volume = None
            print("Couldn't calculate relative volume")
            print(coin.symbol)

    except Exception as e:
        print(f"There was a problem calculating relative volume V2 for: {e}")
        print(coin.symbol)

    return relative_volume


def calculate_price_change_five_min(coin):

    # (price change over 5 min / price 5 min ago) * 100

    price_change_5min = None

    prices = ShortIntervalData.objects.filter(coin=coin).order_by('-timestamp')[:2]

    # Extract the most recent and second most recent, if available
    if len(prices) < 2:
        return None

    price_now, price_five_min_ago = prices[0].price, prices[1].price
    if price_five_min_ago == 0:
        return None

    price_change_5min = ((price_now - price_five_min_ago) / price_five_min_ago) * 100
    return price_change_5min


def calculate_price_change_ten_min(coin):

    # (price change over 10 min / price 10 min ago) * 100

    price_change_10min = None

    prices = ShortIntervalData.objects.filter(coin=coin).order_by('-timestamp')[:4]

    price_now = prices[0].price if len(prices) > 0 else None
    price_ten_min_ago = prices[3].price if len(prices) > 3 else None

    if price_now != None and price_ten_min_ago != None:

        price_difference = price_now - price_ten_min_ago
        price_change_10min = (price_difference / price_ten_min_ago) * 100 if price_ten_min_ago != 0 else None
        return price_change_10min

    else:
        return None


def calculate_twenty_min_relative_volume(coin):

    twenty_min_relative_volume = None

    # volume now / volume 20 min ago - trying this instead of average volume over last 20 min

    try:

        volumes = ShortIntervalData.objects.filter(coin=coin).order_by('-timestamp')[:8]

        volume_now = volumes[0].volume_5min if len(volumes) > 0 else None

        remaining_volumes = volumes[1:]

        sum = 0
        for volume in remaining_volumes:
            sum += volume.volume_5min

        if len(remaining_volumes) != 0:
            average = sum / len(remaining_volumes)

        else:
            average = None


        #volume_twenty_min_ago = volumes[20].volume_5min if len(volumes) > 20 else None

        if volume_now != None and average != None:

            twenty_min_relative_volume = (volume_now / average) if average != 0 else None
            return twenty_min_relative_volume

        else:
            return None

    except Exception as e:
        print(f"There was a problem calculating 20 min relative volume for: {e}")
        print(coin.symbol)

    return twenty_min_relative_volume


def calculate_five_min_relative_volume(coin):

    five_min_relative_volume = None

    # volume now / volume 5 min ago - trying this instead of average volume over last 5 min

    try:

        volumes = ShortIntervalData.objects.filter(coin=coin).order_by('-timestamp')[:4]
        #volumes = ShortIntervalData.objects.filter(coin=coin).order_by('-timestamp')

        # Extract the most recent and second most recent, if available
        volume_now = volumes[0].volume_5min if len(volumes) > 0 else None

        remaining_volumes = volumes[1:]

        sum = 0
        for volume in remaining_volumes:
            sum += volume.volume_5min

        if len(remaining_volumes) != 0:
            average = sum / len(volumes)

        else:
            average = None

        #volume_five_min_ago = volumes[200].volume_5min if len(volumes) > 60 else None

        if volume_now != None and average != None and average != 0:

            five_min_relative_volume = (volume_now / average)
            return five_min_relative_volume

        else:
            print("problem in five min relative volume")
            print(coin.symbol)
            print(volume_now)
            print(average)
            return None

    except Exception as e:
        print(f"There was a problem calculating 5 min relative volume for: {e}")
        print(coin.symbol)

    return five_min_relative_volume




def calculate_meme_price_change_five_min(coin):

    # (price change over 5 min / price 5 min ago) * 100

    price_change_5min = None

    prices = MemeShortIntervalData.objects.filter(coin=coin).order_by('-timestamp')[:2]

    # Extract the most recent and second most recent, if available
    if len(prices) < 2:
        return None

    price_now, price_five_min_ago = prices[0].price, prices[1].price
    if price_five_min_ago == 0:
        return None

    price_change_5min = ((price_now - price_five_min_ago) / price_five_min_ago) * 100
    return price_change_5min

def calculate_meme_price_change_ten_min(coin):

    # (price change over 10 min / price 10 min ago) * 100

    price_change_10min = None

    prices = MemeShortIntervalData.objects.filter(coin=coin).order_by('-timestamp')[:4]

    price_now = prices[0].price if len(prices) > 0 else None
    price_ten_min_ago = prices[3].price if len(prices) > 3 else None

    if price_now != None and price_ten_min_ago != None:

        price_difference = price_now - price_ten_min_ago
        price_change_10min = (price_difference / price_ten_min_ago) * 100 if price_ten_min_ago != 0 else None
        return price_change_10min

    else:
        return None

def calculate_meme_five_min_relative_volume(coin):

    five_min_relative_volume = None

    # volume now / volume 5 min ago - trying this instead of average volume over last 5 min

    try:

        volumes = MemeShortIntervalData.objects.filter(coin=coin).order_by('-timestamp')[:4]
        #volumes = ShortIntervalData.objects.filter(coin=coin).order_by('-timestamp')

        # Extract the most recent and second most recent, if available
        volume_now = volumes[0].volume_5min if len(volumes) > 0 else None

        remaining_volumes = volumes[1:]

        sum = 0
        for volume in remaining_volumes:
            sum += volume.volume_5min

        if len(remaining_volumes) != 0:
            average = sum / len(volumes)

        else:
            average = None

        #volume_five_min_ago = volumes[200].volume_5min if len(volumes) > 60 else None

        if volume_now != None and average != None and average != 0:

            five_min_relative_volume = (volume_now / average)
            return five_min_relative_volume

        else:
            print("problem in five min relative volume")
            print(coin.symbol)
            print(volume_now)
            print(average)
            return None

    except Exception as e:
        print(f"There was a problem calculating 5 min relative volume for: {e}")
        print(coin.symbol)

    return five_min_relative_volume










# =======================================================================

# stop everything
# python3 manage.py makemigrations
# python3 manage.py migrate
# python3 manage.py scheduled_task


# Function to fetch all Solana meme coins and save to the Coins model
def fetch_solana_meme_coins():

    CMC_API_KEY = "7dd5dd98-35d0-475d-9338-407631033cd9"
    CMC_API_BASE_URL = "https://pro-api.coinmarketcap.com/v1/"

    HEADERS = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": CMC_API_KEY,
    }

    url = f"{CMC_API_BASE_URL}cryptocurrency/listings/latest"
    params = {
        "start": 1,
        "limit": 3000,
        "convert": "USD",
    }

    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        coins_data = response.json().get("data", [])

        for coin in coins_data:

            platform = coin.get("platform")
            tags = coin.get("tags", [])

            # Check for Solana platform and meme-related tags
            if platform and platform.get("name") == "Solana" and any(tag in tags for tag in ["memes", "cat-themed", "animal-memes"]):

                MemeCoin.objects.update_or_create(
                    cmc_id=coin.get("id"),
                    defaults={
                        "name": coin.get("name"),
                        "symbol": coin.get("symbol"),
                        "market_cap_rank": coin.get("cmc_rank"),
                        "last_updated": datetime.strptime(coin.get("last_updated"), "%Y-%m-%dT%H:%M:%S.%fZ"),
                    },
                )

        print("Solana meme coins fetched and updated.")
    except Exception as e:
        print(f"Error fetching Solana meme coins: {e}")


# Function to check for new Solana meme coin listings and get metrics
def check_new_solana_listings():

    print("looking for new solana meme coins")

    CMC_API_KEY = "7dd5dd98-35d0-475d-9338-407631033cd9"
    CMC_API_BASE_URL = "https://pro-api.coinmarketcap.com/v1/"

    HEADERS = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": CMC_API_KEY,
    }

    url = f"{CMC_API_BASE_URL}cryptocurrency/listings/new"

    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        new_listings = response.json().get("data", {})

        for coin in new_listings:
            # Filter for Solana-based meme coins

            for tag in coin.get("tags"):
                if tag['slug'] == 'solana-ecosystem':

                    exists = MemeCoin.objects.filter(cmc_id=coin.get("id")).exists()

                    if exists == False:
                        coin_obj, created = MemeCoin.objects.update_or_create(
                            cmc_id=coin.get("id"),
                            defaults={
                                "name": coin.get("name"),
                                "symbol": coin.get("symbol"),
                                "market_cap_rank": coin.get("cmc_rank"),
                                "date_added": datetime.strptime(coin.get("last_updated"), "%Y-%m-%dT%H:%M:%S.%fZ"),
                            },
                        )

                        #print("new meme coin created")
                        #print(coin_obj.symbol)

                        # If the coin is new, fetch its metrics
                        if created:
                            print("MEME CREATED")
                            text_message = f"new meme coin created: {coin_obj.symbol}"
                            text_to_send = [text_message]
                            send_text(text_to_send)
                            #fetch_memecoin_metrics(coin_obj)

        print("New Solana meme coins checked and updated.")
    except Exception as e:
        print(f"Error checking new Solana listings: {e}")


def fetch_memecoin_metrics(coin=None):
    COINMARKETCAP_API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
    API_URL = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'

    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": COINMARKETCAP_API_KEY,
    }

    if coin:
        params = {
            "id": coin.cmc_id,
            "convert": "USD",
        }

    else:
        meme_coins = MemeCoin.objects.all()
        cmc_ids = [coin.cmc_id for coin in meme_coins]

        # API limit: Up to 100 IDs per call
        batch_size = 100
        for i in range(0, len(cmc_ids), batch_size):
            cmc_id_batch = cmc_ids[i:i + batch_size]
            params = {
                "id": ",".join(map(str, cmc_id_batch)),
                "convert": "USD",
            }


    try:
        response = requests.get(API_URL, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        for cmc_id in cmc_id_batch:
            if str(cmc_id) in data["data"]:

                coin = MemeCoin.objects.get(cmc_id=cmc_id)

                coin_data = data["data"][str(cmc_id)]

                try:
                    MemeShortIntervalData.objects.update_or_create(
                        coin=coin,
                        timestamp=datetime.strptime(
                            coin_data["last_updated"], "%Y-%m-%dT%H:%M:%S.%fZ"
                        ),
                        price=coin_data["quote"]["USD"]["price"],
                        volume_5min=coin_data["quote"]["USD"]["volume_24h"],
                        circulating_supply=coin_data["circulating_supply"]
                    )

                except:
                    print("failed fetching meme shortIntervalData")


                try:
                    meme_metric = MemeMetric.objects.update_or_create(
                        coin=coin,
                        timestamp=datetime.strptime(
                            coin_data["last_updated"], "%Y-%m-%dT%H:%M:%S.%fZ"
                        ),
                        defaults={
                            "five_min_relative_volume": calculate_meme_five_min_relative_volume(coin),
                            "price_change_5min": calculate_meme_price_change_five_min(coin),
                            "price_change_10min": calculate_meme_price_change_ten_min(coin),
                            "price_change_1hr": coin_data["quote"]["USD"]["percent_change_1h"],
                            "circulating_supply": coin_data["circulating_supply"],
                            "volume_24h": coin_data["quote"]["USD"]["volume_24h"],
                            "last_price": coin_data["quote"]["USD"]["price"],
                            "market_cap": coin_data["quote"]["USD"]["market_cap"],
                        },
                    )

                except Exception as e:
                    print(f"failed fetching meme metrics: {e}")

    except Exception as e:
        print(f"Error fetching metrics for batch starting with: {e}")


def meme_coin_triggers():

    meme_coins = MemeCoin.objects.all()

    for coin in meme_coins:

        meme_triggers = []

        # check for an increase in volume
        meme_trigger_one = False
        meme_trigger_two = False
        meme_trigger_three = False
        meme_trigger_four = False

        meme_volumes = MemeMetric.objects.filter(coin=coin).order_by('-timestamp').all()

        if len(meme_volumes) > 2:

            current_meme_metric = meme_volumes[0]

            if hasattr(meme_volumes[0], 'volume_24h') and meme_volumes[0].volume_24h != None:

                current_meme_volume = meme_volumes[0].volume_24h
                remaining_meme_volumes = meme_volumes[1:]

                # (current volume - avg volume 30 min ago) / avg volume 30 min age * 100
                sum_meme_volume = sum(data.volume_24h for data in remaining_meme_volumes)
                average_meme_volume = sum_meme_volume / len(remaining_meme_volumes)
                temp = current_meme_volume - average_meme_volume
                percentage_change_meme_volume = (temp / average_meme_volume) * 100
                if percentage_change_meme_volume > 10:
                    meme_trigger_one = True

                    #trigger = "MEME | " + coin.symbol + " : relative volume > 10"
                    #meme_triggers.append(trigger)

                if percentage_change_meme_volume > 3:
                    meme_trigger_four = True

            # price increased 5% over last 5 min
            if hasattr(current_meme_metric, 'price_change_5min') and current_meme_metric.price_change_5min != None:
                if current_meme_metric.price_change_5min > 10:
                    meme_trigger_two = True

                    #trigger = "MEME | " + coin.symbol + " : price change 5 min > 5"
                    #meme_triggers.append(trigger)

                if current_meme_metric.price_change_5min < 5:
                    meme_trigger_four = False

            # price increased 5% over last 10 min
            if hasattr(current_meme_metric, 'price_change_10min') and current_meme_metric.price_change_10min != None:
                if current_meme_metric.price_change_10min > 15:
                    meme_trigger_three = True

                    #trigger = "MEME | " + coin.symbol + " : price change 10 min > 10"
                    #meme_triggers.append(trigger)

            if meme_trigger_one == True and meme_trigger_two == True:
                trigger = "MEME | " + coin.symbol + " : relative volume > 12 and price change 5 min > 10"
                meme_triggers.append(trigger)

            '''
            if meme_trigger_one == True and meme_trigger_three == True:
                trigger = "MEME | " + coin.symbol + " : relative volume > 10 and price change 10 min > 10"
                meme_triggers.append(trigger)
            '''

            # trigger 4
            # 1. coins that were listed in the last 3 hours
            # 2. average true volume is 3x
            # 3. has consistent price increase over 5 intervals of 1 minute ( so 5 green candle sticks in a row on the 1 minute)
            # 4. has atleast a volume of 200K

            current_time = now()

            if hasattr(coin, 'date_added') and coin.date_added != None:
                time_difference = current_time - coin.date_added
                if time_difference >= timedelta(hours=3):
                    meme_trigger_four = False

            if meme_trigger_four == True:
                if hasattr(current_meme_metric, 'volume_24h') and current_meme_metric.volume_24h != None:
                    if current_meme_metric.volume_24h < 300000:
                        meme_trigger_four = False

            if meme_trigger_four == True:
                trigger = "MEME | " + coin.symbol + " : RICKI'S CRITERIA HIT"
                meme_triggers.append(trigger)



        if len(meme_triggers) > 0:
            send_text(meme_triggers)

    return


# ====================================================================




def five_min_update(request=None):

    # if the time is ~0000 delete old data
    now = datetime.now()
    if now.hour == 0 and now.minute <= 5:
        #manually_clean_database()
        print("not deleting right now...")


    API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'

    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": API_KEY,
    }

    coins = Coin.objects.all()
    cmc_ids = [coin.cmc_id for coin in coins]

    # API limit: Up to 100 IDs per call
    batch_size = 100
    for i in range(0, len(cmc_ids), batch_size):
        cmc_id_batch = cmc_ids[i:i + batch_size]  # Create batches of 100 IDs
        params = {
            "id": ",".join(map(str, cmc_id_batch)),  # Pass CMC IDs as a comma-separated string
            "convert": "USD",
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            for cmc_id in cmc_id_batch:
                if str(cmc_id) in data["data"]:
                    crypto_data = data["data"][str(cmc_id)]

                    coin = Coin.objects.get(cmc_id=cmc_id)

                    try:
                        coin.market_cap_rank = crypto_data["cmc_rank"]
                        coin.last_updated = datetime.strptime(
                            crypto_data["last_updated"], "%Y-%m-%dT%H:%M:%S.%fZ"
                        )
                        coin.save()
                    except:
                        print("FAILED IN GROUP 1")

                    try:
                        ShortIntervalData.objects.create(
                            coin=coin,
                            timestamp=datetime.strptime(
                                crypto_data["last_updated"], "%Y-%m-%dT%H:%M:%S.%fZ"
                            ),
                            price=crypto_data["quote"]["USD"]["price"],
                            volume_5min=crypto_data["quote"]["USD"]["volume_24h"],
                            circulating_supply=crypto_data["circulating_supply"]
                        )
                    except:
                        print("FAILED IN GROUP 2")

                    try:
                        metrics = Metrics.objects.filter(coin=coin).order_by('-timestamp')

                        #if metrics.count() > 122:
                            #metrics_to_delete = metrics[122:]  # Slice to get metrics beyond the 100th
                            #metrics_to_delete.delete()

                        #Metrics.objects.filter(coin=coin).delete()
                        metric = Metrics.objects.create(
                            coin=coin,
                            timestamp=datetime.strptime(
                                crypto_data["last_updated"], "%Y-%m-%dT%H:%M:%S.%fZ"
                            ),
                            daily_relative_volume=calculate_daily_relative_volume(coin),
                            rolling_relative_volume=calculate_relative_volume(coin),
                            five_min_relative_volume=calculate_five_min_relative_volume(coin),
                            twenty_min_relative_volume=calculate_twenty_min_relative_volume(coin),
                            price_change_5min=calculate_price_change_five_min(coin),
                            price_change_10min=calculate_price_change_ten_min(coin),
                            price_change_1hr = crypto_data["quote"]["USD"]["percent_change_1h"],
                            price_change_24hr = crypto_data["quote"]["USD"]["percent_change_24h"],
                            price_change_7d = crypto_data["quote"]["USD"]["percent_change_7d"],
                            circulating_supply=crypto_data["circulating_supply"],
                            volume_24h = crypto_data["quote"]["USD"]["volume_24h"],
                            last_price = crypto_data["quote"]["USD"]["price"],
                            market_cap = crypto_data["quote"]["USD"]["market_cap"]
                        )

                    except Exception as e:
                        print("FAILED IN GROUP 3")
                        print(e)

                    if now.hour == 0 and now.minute <= 5:

                        timestamp = datetime.strptime(crypto_data["last_updated"], "%Y-%m-%dT%H:%M:%S")
                        date = timestamp.date()

                        try:
                            HistoricalData.objects.create(
                                coin=coin,
                                date=date,
                                price=crypto_data["quote"]["USD"]["price"],
                                volume_24h=crypto_data["quote"]["USD"]["volume_24h"],
                            )

                        except:
                            print("Couldn't create new historical data")

        except Exception as e:
            print(f"Error updating tracked coins for batch {cmc_id_batch}: {e}")


    # wait 30 seconds before checking solana
    print("pausing for 30 seconds before solana check")
    time.sleep(30)
    print("checking solana")
    check_new_solana_listings()
    fetch_memecoin_metrics()
    print("done fetching solana data")
    print("checking meme triggers")
    meme_coin_triggers()
    print("done checking meme triggers")

    if request:
        return JsonResponse({"status": "success", "message": "Update triggered successfully"})


def index(request):

    # get all the coins in our databse
    top_cryptos = []
    coins = Coin.objects.all()

    # try storing multiple daily relative volumes to see the progression
    daily_relative_volumes = []

    sorted_volumes = []

    for coin in coins:

        # use the following to see metrics on a coin that pumped recently
        '''
        if coin.symbol == 'G':
            metrics = Metrics.objects.filter(coin=coin).order_by("-timestamp").all()
            for metric in metrics:
                print("Time: " + str(metric.timestamp))
                print("daily_relative_volume:" + str(metric.daily_relative_volume))
                print("rolling relative volume: " + str(metric.rolling_relative_volume))
                print("five_min_relative_volume: " + str(metric.five_min_relative_volume))
                print("twenty_min_relative_volume: " + str(metric.twenty_min_relative_volume))
                print("price_change_5min: " + str(metric.price_change_5min))
                print("price_change_10min: " + str(metric.price_change_10min))
                print("price_change_1hr: " + str(metric.price_change_1hr))
                print("price_change_24hr: " + str(metric.price_change_24hr))
                print("circulating_supply: " + str(metric.circulating_supply))
                print("volume_24h: " + str(metric.volume_24h))
                print("last_price: " + str(metric.last_price))
                print("market_cap: " + str(metric.market_cap))
        '''



        shortIntervalData = ShortIntervalData.objects.filter(coin=coin).order_by("-timestamp").first()
        #metric = Metrics.objects.get(coin=coin)
        metric = Metrics.objects.filter(coin=coin).order_by("-timestamp").first()

        if hasattr(shortIntervalData, 'timestamp'):
            coin_time = shortIntervalData.timestamp
        else:
            coin_time = None

        coin_name = coin.name
        coin_symbol = coin.symbol

        if hasattr(shortIntervalData, 'price') and shortIntervalData.price != None:
            coin_price = round(shortIntervalData.price, 7)
        else:
            coin_price = None
        if hasattr(metric, 'market_cap'):
            coin_market_cap = metric.market_cap
        else:
            coin_market_cap = None
        if hasattr(metric, 'volume_24h') and metric.volume_24h != None:
            coin_volume_24h_USD = round(metric.volume_24h, 2)
        else:
            coin_volume_24h_USD = None
        if hasattr(metric, 'price_change_1hr') and metric.price_change_1hr != None:
            coin_price_change_1h = round(metric.price_change_1hr, 2)
        else:
            coin_price_change_1h = None
        if hasattr(metric, 'price_change_24hr') and metric.price_change_24hr != None:
            coin_price_change_24h_percentage = round(metric.price_change_24hr, 2)
        else:
            coin_price_change_24h_percentage = None
        if hasattr(metric, 'price_change_7d') and metric.price_change_7d != None:
            coin_price_change_7d = round(metric.price_change_7d, 2)
        else:
            coin_price_change_7d = None
        if hasattr(metric, 'circulating_supply'):
            coin_circulating_supply = metric.circulating_supply
        else:
            coin_circulating_supply = None
        if hasattr(metric, 'rolling_relative_volume') and metric.rolling_relative_volume != None:
            coin_rolling_relative_volume = round(metric.rolling_relative_volume, 2)
        else:
            coin_rolling_relative_volume = None
        if hasattr(metric, 'daily_relative_volume') and metric.daily_relative_volume != None:
            coin_daily_relative_volume = round(metric.daily_relative_volume, 2)
        else:
            coin_daily_relative_volume = None
        if hasattr(metric, 'twenty_min_relative_volume') and metric.twenty_min_relative_volume != None:
            coin_twenty_min_relative_volume = round(metric.twenty_min_relative_volume, 2)
        else:
            coin_twenty_min_relative_volume = None
        if hasattr(metric, 'five_min_relative_volume') and metric.five_min_relative_volume != None:
            coin_five_min_relative_volume = round(metric.five_min_relative_volume, 2)
        else:
            coin_five_min_relative_volume = None
        if hasattr(metric, 'price_change_5min') and metric.price_change_5min != None:
            coin_price_change_5min = round(metric.price_change_5min, 2)
        else:
            coin_price_change_5min = None
        if hasattr(metric, 'price_change_10min') and metric.price_change_10min != None:
            coin_price_change_10min = round(metric.price_change_10min, 2)
        else:
            coin_price_change_10min = None

        # Go through the different triggers
        true_triggers = []
        true_triggers_two = []

        # TRIGGER ONE - price up 10% or more in last 24 hours
        triggerOne = False
        if coin_price_change_24h_percentage != None:
            if coin_price_change_24h_percentage >= 10:
                triggerOne = True
                trigger = coin.symbol + " : Price Change > 10% in 24 hours"
                true_triggers.append(trigger)

        # TRIGGER TWO - rolling relative volume is 2.0 or higher
        triggerTwo = False
        if coin_rolling_relative_volume != None:
            if coin_rolling_relative_volume >= 2.0:
                triggerTwo = True
                trigger = coin.symbol + " : Relative Volume >= 2.0"
                true_triggers.append(trigger)
                #true_triggers_two.append(trigger)

        # TRIGGER THREE - price is less than $50
        triggerThree = False
        if hasattr(shortIntervalData, 'price') and shortIntervalData.price != None:
            if shortIntervalData.price < 50:
                triggerThree = True

        # TRIGGER FOUR - circulating supply is less than 100 million
        triggerFour = False
        if coin_circulating_supply != None:
            if coin_circulating_supply < 100000000:
                triggerFour = True

        # TRIGGER FIVE - market cap between $10M and $1B
        triggerFive = False
        if coin_market_cap != None:
            if coin_market_cap > 10000000 and coin_market_cap < 1000000000:
                triggerFive = True

        # TRIGGER SIX - volume today is at least 200000
        triggerSix = False
        if coin_volume_24h_USD != None:
            if coin_volume_24h_USD >= 200000:
                triggerSix = True

        # TRIGGER SEVEN - 20 min relative volume > 2.0
        triggerSeven = False
        if coin_twenty_min_relative_volume != None:
            if coin_twenty_min_relative_volume > 2.0:
                triggerSeven = True
                trigger = coin.symbol + " : 20 min Relative Volume > 2.0"
                true_triggers.append(trigger)

        # TRIGGER EIGHT - circulating supply down >5% in last 24 hours
        triggerEight = False
        circulating_supplies = ShortIntervalData.objects.filter(coin=coin).order_by('-timestamp')
        remaining_circulating_supplies = circulating_supplies[1:]
        circulating_supply_now = circulating_supplies[0].circulating_supply
        sum_circulating_supplies = sum(data.circulating_supply for data in remaining_circulating_supplies)

        if len(remaining_circulating_supplies) != 0:
            average_circulating_supply = sum_circulating_supplies / len(remaining_circulating_supplies)

            if average_circulating_supply != 0:
                circulating_supply_change = ((average_circulating_supply - circulating_supply_now) / average_circulating_supply) * 100

                if circulating_supply_change > 50:
                    #triggerEight = True
                    triggerEight = False
                    #trigger = coin.symbol + " : Circulating Supply down > 50% in last 24 hours"
                    #true_triggers_two.append(trigger)

        # TRIGGER NINE - price is up 5%+ in last hour
        triggerNine = False
        if coin_price_change_1h != None and coin_price_change_1h >= 5:
            triggerNine = True

            if triggerTwo == True:
                trigger = coin.symbol + " : Price > 5% in last hour and RVOL > 2"
                true_triggers_two.append(trigger)


        # TRIGGER TEN -
        # rolling_relative_volume > 2.5 and increasing over 2 intervals
        # five_min_relative_volume > 1.7 for 2 consecutive intervals
        # price_change_10min > 3%
        # volume_24h increases by 10% within the last 30 minutes
        triggerTen = False
        if coin_rolling_relative_volume > 2:
            if coin_five_min_relative_volume > 1.5:
                if coin_price_change_10min > 3:

                    # (current volume - volume 30 min ago) / volume 30 min age * 100
                    trigger_ten_volumes = Metrics.objects.filter(coin=coin).order_by('-timestamp')[:8]
                    if len(trigger_ten_volumes) > 6:
                        trigger_ten_volume = trigger_ten_volumes[0].volume_24h
                        trigger_ten_30_min_volume = trigger_ten_volumes[6].volume_24h
                        temp = (trigger_ten_volume - trigger_ten_30_min_volume)
                        trigger_ten_percent_change = (temp / trigger_ten_30_min_volume) * 100

                        if trigger_ten_percent_change >= 5:
                            triggerTen = True

                            trigger = coin.symbol + " : TRIGGER TEN HIT !!!"
                            true_triggers_two.append(trigger)

        if len(true_triggers_two) > 0:
            send_text(true_triggers_two)

        try:
            top_cryptos.append({
                "time": coin_time,
                "name": coin_name,
                "symbol": coin_symbol,
                "price": coin_price,
                "market_cap": coin_market_cap,
                "volume_24h_USD": coin_volume_24h_USD,
                "price_change_1h": coin_price_change_1h,
                "price_change_24h_percentage": coin_price_change_24h_percentage,
                "price_change_7d": coin_price_change_7d,
                "circulating_supply": coin_circulating_supply,
                "circulating_supply_change": circulating_supply_change,
                "daily_relative_volume": coin_daily_relative_volume,
                "rolling_relative_volume": coin_rolling_relative_volume,
                "five_min_relative_volume": coin_five_min_relative_volume,
                "twenty_min_relative_volume": coin_twenty_min_relative_volume,
                "price_change_5min": coin_price_change_5min,
                "price_change_10min": coin_price_change_10min,
                "triggerOne": triggerOne,
                "triggerTwo": triggerTwo,
                "triggerThree": triggerThree,
                "triggerFour": triggerFour,
                "triggerFive": triggerFive,
                "triggerSix": triggerSix,
                "triggerSeven": triggerSeven,
                "triggerEight": triggerEight,
                "true_triggers": true_triggers,
            })

        except Exception as e:
            print(f"Couldn't fetch all the data... - Error fetching data: {e}")
            print(coin)
            print(coin.symbol)


        # gather relative volume data
        try:

            relative_volumes = Metrics.objects.filter(coin=coin).order_by("-timestamp")[:73]
            #relative_volumes = Metrics.objects.filter(coin=coin).order_by("-timestamp")

            # get every 12th
            relative_volumes = relative_volumes[::6]


            volumes = []

            for volume in relative_volumes:

                if volume.rolling_relative_volume != None:
                    volumes.append(round(volume.rolling_relative_volume, 2))

            is_descending = all(volumes[i] >= volumes[i + 1] for i in range(len(volumes) - 1))

            #is_descending = True

            not_all_same = len(set(volumes)) > 1

            if is_descending and not_all_same:

                if coin_price_change_24h_percentage != None:
                    coin_price_change_24h_percentage = round(coin_price_change_24h_percentage, 2)

                daily_relative_volumes.append({
                    "rank": coin.market_cap_rank,
                    "symbol": coin_symbol,
                    "price_change_24h_percentage": coin_price_change_24h_percentage,
                    "volumes": volumes,
                    "is_descending": is_descending,
                    "daily_relative_volume": coin_daily_relative_volume,
                    "price": coin_price,
                })


        except Exception as e:
            print(f"Couldn't fetch RELATIVE VOLUME DATAS: {e}")
            print(coin)
            print(coin.symbol)

        try:
            sorted_volumes = sorted(
                daily_relative_volumes,
                key=lambda x: (x["price_change_24h_percentage"] is None, x["price_change_24h_percentage"] if x["price_change_24h_percentage"] is not None else 0),
                reverse=True
            )
        except:
            sorted_volumes = []

    # sort by top gainers: price change over the last 24 hours
    try:
        sorted_coins = sorted(
            top_cryptos,
            key=lambda x: (x["daily_relative_volume"] is None, x["daily_relative_volume"] if x["daily_relative_volume"] is not None else 0),
            reverse=True
        )

    except:
        print("--------------------------Couldn't sort coins...")
        sorted_coins = top_cryptos



    # If this is an Ajax automatic refresh:
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':  # Check if it's an AJAX request

        data = {
            "top_cryptos": sorted_coins,  # Ensure this is serializable
            "sorted_volumes": sorted_volumes,  # Ensure this is serializable
        }
        return JsonResponse(data, safe=False)


    # Render data to the HTML template
    return render(request, "index.html", {
        "top_cryptos": sorted_coins,
        "sorted_volumes": sorted_volumes
    })


def gather_daily_historical_data():

    API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
    URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"

    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": API_KEY,
    }

    coins = Coin.objects.all()
    coins_in_group_of_fifteen = []
    coin_group = []
    count = 0

    for coin in coins:

        if count < 15:
            coin_group.append(coin)
            count += 1

        else:
            count = 1
            coins_in_group_of_fifteen.append(coin_group)
            coin_group = []
            coin_group.append(coin)


    for coin_group in coins_in_group_of_fifteen:
        for coin in coin_group:
            try:
                end_time = datetime.now()
                start_time = end_time - timedelta(days=30)

                params = {
                    "id": coin.cmc_id,
                    "time_start": start_time.isoformat(),
                    "time_end": end_time.isoformat(),
                    "interval": "1d",
                }

                response = requests.get(URL, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()

                if "data" in data and "quotes" in data["data"]:
                    historical_data = data["data"]["quotes"]

                    for quote in historical_data:

                        HistoricalData.objects.update_or_create(
                            coin=coin,
                            date=quote["timestamp"].split("T")[0],
                            defaults={
                                "price": quote["quote"]["USD"]["price"],
                                "volume_24h": quote["quote"]["USD"]["volume_24h"],
                            },
                        )

                else:
                    print('==============')
                    print(' Historical Data error with:')
                    print(coin.symbol)
                    print(data)

                    HistoricalData.objects.update_or_create(
                        coin=coin,
                        date=end_time,
                        defaults={
                            "price": None,
                            "volume_24h": None,
                        },
                    )

            except Exception as e:
                print(f"Error fetching historical data for {coin.symbol}: {e}")

        # Pause for 60 seconds
        print("pausing for 60 seconds")
        time.sleep(60)
        print("resuming")


def analyze_historical_metrics():

    API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
    URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"

    target_time = "2025-01-25 14:45:00"
    target_datetime = datetime.strptime(target_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    time_end = target_datetime.isoformat()
    time_start = (target_datetime - timedelta(days=30)).isoformat()



    # UTC TIME ZONE------------------------------------------------
    # XRP at Dec 20, 2024 12:30:00 - right before 16% up
    # XRP at Dec 10, 2024 19:00:00 - 16% up
    # XRP at Dec 02, 2024 12:00:00 - 18% up
    # XRP at Dec 01, 2024 17:00:00 - 28% up
    # XRP at Nov 09, 2024 21:00:00 - 120% up

    # KAS on Jan 13, 2025 at 1445 - about to go up 20%








    symbol = "XRP"

    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": API_KEY,
    }

    params = {
        "symbol": symbol,
        "time_start": time_start,
        "time_end": time_end,
        "interval": "1d",
    }

    try:
        response = requests.get(URL, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        if "data" in data and "quotes" in data["data"]:
            historical_data = data["data"]["quotes"]

            if historical_data:

                volumes = [point["quote"]["USD"]["volume_24h"] for point in historical_data]
                current_volume = volumes[-1]
                average_volume = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else None

                if average_volume and average_volume != 0:
                    relative_volume = current_volume / average_volume

                    print(f"Historical Analysis for {symbol}:")
                    print(f"Price at Specific Time: {historical_data[-1]['quote']['USD']['price']}")
                    print(f"Relative Volume: {relative_volume}")
                    print(f"1 hour percent change: {historical_data[-1]['quote']['USD']['percent_change_1h']}")
                    print(f"24 hour percent change: {historical_data[-1]['quote']['USD']['percent_change_24h']}")
                    print(f"7 day percent change: {historical_data[-1]['quote']['USD']['percent_change_7d']}")

                    return None

                else:
                    print("Average volume is zero or not available.")
                    return None

            else:
                print("No historical data found.")

        else:
            print(f"Error in response data: {data}")
            return []

    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return []


def manually_clean_database():

    # Calculate cutoff times
    thirty_days_ago = now() - timedelta(days=31)
    thirty_six_hours_ago = now() - timedelta(hours=25)

    # Delete HistoricalData older than 30 days
    historical_deleted, _ = HistoricalData.objects.filter(date__lt=thirty_days_ago).delete()

    # Delete ShortIntervalData older than 36 hours
    short_interval_deleted, _ = ShortIntervalData.objects.filter(timestamp__lt=thirty_six_hours_ago).delete()

    # Delete Metrics older than 36 hours
    metrics_deleted, _ = Metrics.objects.filter(timestamp__lt=thirty_six_hours_ago).delete()

    print(f"Cleaned database:")
    print(f" - {historical_deleted} HistoricalData entries older than 30 days")
    print(f" - {short_interval_deleted} ShortIntervalData entries older than 36 hours")
    print(f" - {metrics_deleted} Metrics entries older than 36 hours")


def test(request):
    return render(request, "test.html")




def analyze_recent_metrics(event_time, coin_symbol):

    # UTC TIME ZONE------------------------------------------------
    # XRP at Dec 20, 2024 12:30:00 - right before 16% up
    # XRP at Dec 10, 2024 19:00:00 - 16% up
    # XRP at Dec 02, 2024 12:00:00 - 18% up
    # XRP at Dec 01, 2024 17:00:00 - 28% up
    # XRP at Nov 09, 2024 21:00:00 - 120% up

    # KAS on Jan 13, 2025 at 1445 - about to go up 20%
    # FARTCOIN on Jan 13, 2025 at 1215 - about to go up 60%
    # TURBO on Jan 13, 2025 at 1845 - about to go up 14%
    # ACT on Jan 13, 2025 at 1900 - about to go up 16%
    # URO on Jan 13, 2025 at 2030 - about to go up 24%
    # STMX on Jan 13, 2025 at 1900 - about to go up 17%
    # PRIME on Jan 13, 2025 at 1845 - about to go up 22%


    # Define the event time

    # Jan 13, 2025, 1445 UTC
    #event_time = datetime(2025, 1, 13, 14, 45, tzinfo=timezone.utc)

    # Jan 13, 2025, 1215 UTC
    #event_time = datetime(2025, 1, 13, 12, 15, tzinfo=timezone.utc)

    # Jan 13, 2025, 1845 UTC
    #event_time = datetime(2025, 1, 13, 18, 45, tzinfo=timezone.utc)


    #coin_symbol = "KAS"
    #python manage.py analyze_recent_metrics --coin_symbol="KAS" --event_time="2025-01-13 14:45:00"

    #coin_symbol = "FARTCOIN"
    #python manage.py analyze_recent_metrics --coin_symbol="FARTCOIN" --event_time="2025-01-13 12:15:00"

    #coin_symbol = "TURBO"
    #python manage.py analyze_recent_metrics --coin_symbol="TURBO" --event_time="2025-01-13 18:45:00"

    #coin_symbol = "ACT"
    #python manage.py analyze_recent_metrics --coin_symbol="ACT" --event_time="2025-01-13 19:00:00"

    #coin_symbol = "URO"
    #python manage.py analyze_recent_metrics --coin_symbol="URO" --event_time="2025-01-13 18:45:00"

    #coin_symbol = "STMX"
    #python manage.py analyze_recent_metrics --coin_symbol="STMX" --event_time="2025-01-13 19:00:00"

    #coin_symbol = "PRIME"
    #python manage.py analyze_recent_metrics --coin_symbol="PRIME" --event_time="2025-01-13 18:45:00"






    # Calculate the time range: 1 hour before and 5 hours after
    start_time = event_time - timedelta(hours=1)
    end_time = event_time + timedelta(hours=5)

    try:
        coin = Coin.objects.get(symbol=coin_symbol)

        metrics = coin.metrics.filter(timestamp__gte=start_time, timestamp__lte=end_time)

        # Prepare the data for JSON response
        metrics_data = [
            {
                "Metric Name": metric.coin.symbol,
                "Timestamp": metric.timestamp,
                "daily rvol": metric.daily_relative_volume,
                "rolling rvol": metric.rolling_relative_volume,
                "five min rvol": metric.five_min_relative_volume,
                "20 min rvol": metric.twenty_min_relative_volume,
                "5 min price change": metric.price_change_5min,
                "10 min price change": metric.price_change_10min,
                "1hr price change": metric.price_change_1hr,
                "24hr price change": metric.price_change_24hr,
                "7day price change": metric.price_change_7d,
                "circulaating supply": metric.circulating_supply,
                "24hr volume": metric.volume_24h,
                "last price": metric.last_price,
                "market cap": metric.market_cap,
            }

            for metric in metrics

        ]

        for metric in metrics_data:
            print(metric)

        # Return the data as JSON
        return JsonResponse({"status": "success", "coin": coin_name, "metrics": metrics_data})

    except Coin.DoesNotExist:
        # Handle case where the coin does not exist
        return JsonResponse({"status": "error", "message": f"Coin '{coin_name}' not found."})

    except Exception as e:
        # Handle other exceptions
        return JsonResponse({"status": "error", "message": str(e)})














#
