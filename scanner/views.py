import json
import time
import urllib.request
import os
import requests
import asyncio
import decimal
from django.shortcuts import render
from zoneinfo import ZoneInfo
from django.http import HttpResponseRedirect
from scanner.models import Coin, HistoricalData, ShortIntervalData, Metrics, MemeCoin, MemeMetric, MemeShortIntervalData, Trigger
from datetime import datetime, timedelta, timezone
from django.utils.timezone import now
from django.http import JsonResponse
from django.http import HttpResponse
from django.http import FileResponse, Http404
from django.db.models import Prefetch, OuterRef, Subquery


# used to test out a trigger combination against the data we have in the db
def check_trigger(symbol):

    #coin = Coin.objects.get(symbol=symbol)

    coins = Coin.objects.all()

    amount_of_trades = 0
    successful_trades = 0
    failed_trades = 0
    trigger_one_trades = 0
    trigger_one_success = 0
    trigger_two_trades = 0
    trigger_two_success = 0
    trigger_three_trades = 0
    trigger_three_success = 0
    trigger_short_trades = 0
    trigger_short_success = 0
    trigger_five_trades = 0
    trigger_five_success = 0

    count_15 = 0
    count_16 = 0
    count_17 = 0
    count_18 = 0
    count_19 = 0
    count_20 = 0
    count_21 = 0
    count_22 = 0
    count_23 = 0

    for coin in coins:

        metrics = Metrics.objects.filter(coin=coin).order_by('timestamp')

        average_volume = 0
        for metric in metrics:
            average_volume += metric.volume_24h

        average_volume = average_volume / len(metrics)
        average_volume = average_volume * decimal.Decimal(1.15)

        for x in range(6, len(metrics)):

            if (metrics[x].rolling_relative_volume != None and
                metrics[x].price_change_5min != None and
                metrics[x].price_change_10min != None and
                metrics[x].price_change_1hr != None and
                metrics[x].price_change_24hr != None and
                metrics[x].daily_relative_volume != None and
                metrics[x].five_min_relative_volume != None and
                metrics[x].twenty_min_relative_volume != None):

                # 24 hour volume growth
                # current volume - volume 5 min ago / volume 5 min ago * 100
                current_volume = metrics[x].volume_24h
                previous_volume = metrics[x-1].volume_24h
                volume_growth = (current_volume - previous_volume) / previous_volume * 100

                # 5 min relative volume progression
                # current and previous 2 are increasing or equivalent
                rvol_progression = False
                current_rvol = metrics[x].five_min_relative_volume
                one_previous_rvol = metrics[x-1].five_min_relative_volume
                two_previous_rvol = metrics[x-2].five_min_relative_volume
                if (two_previous_rvol <= one_previous_rvol <= current_rvol):
                    rvol_progression = True

                # 5 min price change is greater than previous
                five_min_price_increase = False
                current_five_min = metrics[x].price_change_5min
                previous_five_min = metrics[x-1].price_change_5min
                previous_five_min_two = metrics[x-2].price_change_5min
                if (previous_five_min < current_five_min and
                    previous_five_min < 0 and
                    current_five_min > 0):
                    five_min_price_increase = True

                # 5 min and 10 min price changes go negative, positive, positive
                ten_min_price_increase = False
                current_ten_min = metrics[x].price_change_10min
                previous_ten_min = metrics[x-1].price_change_10min
                previous_ten_min_two = metrics[x-2].price_change_10min
                if (previous_ten_min < current_ten_min and
                    previous_ten_min < 0):
                    ten_min_price_increase = True


                if (
                    metrics[x].daily_relative_volume >= 1.6 and
                    metrics[x].rolling_relative_volume >= 1.6 and
                    metrics[x].five_min_relative_volume >= 1.3 and
                    metrics[x].price_change_5min >= 0 and
                    metrics[x].price_change_24hr < 0 and
                    metrics[x].twenty_min_relative_volume >= 1 and
                    five_min_price_increase == True and
                    rvol_progression == True

                    #metrics[x].volume_24h_growth >= 10 and
                    #metrics[x].price_change_1hr > 0 and
                    #metrics[x].price_change_7d >= 15 and
                    #metric.market_cap_growth_10min >= 0.5 and
                    #3 <= metrics[x].price_change_24hr <= 10 and
                    #0.8 <= metrics[x].daily_relative_volume <= 1.5 and
                    #abs(metrics[x].five_min_relative_volume - metrics[x].twenty_min_relative_volume) <= 0.1
                    #abs(metric.last_price - metric.recent_local_low) <= 0.01 * metric.recent_local_low
                ):

                    #print("-------TRIGGER ONE-----------")
                    #print(coin.symbol)
                    #print(metrics[x].timestamp)

                    amount_of_trades += 1
                    trigger_one_trades += 1

                    # check if the trigger was right
                    # right means the price went up over 5% in the next 5 hours
                    # and it didnt go below 2% from the trigger price
                    trigger_price = metrics[x].last_price
                    stop_loss_price = trigger_price - (trigger_price * decimal.Decimal(0.02))
                    take_profit_price = trigger_price + (trigger_price * decimal.Decimal(0.05))

                    # try to go through remaining metrics
                    take_profit_hit = False
                    stop_loss_hit = False
                    take_profit_timestamp = None
                    stop_loss_timestamp = None
                    try:
                        for y in range(x, len(metrics)):
                            if (metrics[y].last_price >= take_profit_price):
                                take_profit_hit = True
                                take_profit_timestamp = metrics[y].timestamp

                            if (metrics[y].last_price <= stop_loss_price):
                                stop_loss_hit = True
                                stop_loss_timestamp = metrics[y].timestamp

                        if (take_profit_hit == True):
                            if (stop_loss_hit == True):
                                # compare timestamps
                                if (take_profit_timestamp < stop_loss_timestamp):
                                    # successful trade
                                    successful_trades += 1
                                    trigger_one_success += 1
                                else:
                                    # failed trade
                                    failed_trades += 1
                            else:
                                # successful trade
                                successful_trades += 1
                                trigger_one_success += 1

                        if (take_profit_hit == False and stop_loss_hit == True):
                            # failed trade
                            failed_trades += 1

                        if (take_profit_hit == False and stop_loss_hit == False):
                            amount_of_trades -= 1
                            trigger_one_trades -= 1

                    except:
                        print("failed in trigger 1")

                    day = metrics[x].timestamp.day
                    if day == 15:
                        count_15 += 1
                    elif day == 16:
                        count_16 += 1
                    elif day == 17:
                        count_17 += 1
                    elif day == 18:
                        count_18 += 1
                    elif day == 19:
                        count_19 += 1
                    elif day == 20:
                        count_20 += 1
                    elif day == 21:
                        count_21 += 1
                    elif day == 22:
                        count_22 += 1
                    elif day == 23:
                        count_23 += 1
                        print("-----TRIGGER ONE-------------")
                        print(coin.symbol)
                        print(metrics[x].timestamp)


                if (
                    metrics[x].daily_relative_volume >= 2 and
                    metrics[x].rolling_relative_volume >= 1.2 and
                    metrics[x].five_min_relative_volume >= 1.3 and
                    metrics[x].price_change_5min >= 0.7 and
                    metrics[x].price_change_24hr < -5 and
                    metrics[x].twenty_min_relative_volume >= 1 and
                    rvol_progression == True
                ):
                    #print("-----TRIGGER TWO-------------")
                    #print(coin.symbol)
                    #print(metrics[x].timestamp)

                    amount_of_trades += 1
                    trigger_two_trades += 1

                    trigger_price = metrics[x].last_price
                    stop_loss_price = trigger_price - (trigger_price * decimal.Decimal(0.02))
                    take_profit_price = trigger_price + (trigger_price * decimal.Decimal(0.05))
                    take_profit_hit = False
                    stop_loss_hit = False
                    take_profit_timestamp = None
                    stop_loss_timestamp = None
                    try:
                        for y in range(x, len(metrics)):
                            if (metrics[y].last_price >= take_profit_price):
                                take_profit_hit = True
                                take_profit_timestamp = metrics[y].timestamp

                            if (metrics[y].last_price <= stop_loss_price):
                                stop_loss_hit = True
                                stop_loss_timestamp = metrics[y].timestamp

                        if (take_profit_hit == True):
                            if (stop_loss_hit == True):
                                # compare timestamps
                                if (take_profit_timestamp < stop_loss_timestamp):
                                    # successful trade
                                    successful_trades += 1
                                    trigger_two_success += 1
                                else:
                                    # failed trade
                                    failed_trades += 1
                            else:
                                # successful trade
                                successful_trades += 1
                                trigger_two_success += 1

                        if (take_profit_hit == False and stop_loss_hit == True):
                            # failed trade
                            failed_trades += 1

                        if (take_profit_hit == False and stop_loss_hit == False):
                            amount_of_trades -= 1
                            trigger_two_trades -= 1

                    except:
                        print("failed in trigger 2")

                    day = metrics[x].timestamp.day
                    if day == 15:
                        count_15 += 1
                    elif day == 16:
                        count_16 += 1
                    elif day == 17:
                        count_17 += 1
                    elif day == 18:
                        count_18 += 1
                    elif day == 19:
                        count_19 += 1
                    elif day == 20:
                        count_20 += 1
                    elif day == 21:
                        count_21 += 1
                    elif day == 22:
                        count_22 += 1
                    elif day == 23:
                        count_23 += 1
                        print("-----TRIGGER TWO-------------")
                        print(coin.symbol)
                        print(metrics[x].timestamp)



                # rolling 5 min price change
                rolling_one = metrics[x].price_change_5min
                rolling_two = metrics[x-1].price_change_5min
                rolling_three = metrics[x-2].price_change_5min
                rolling_price_change_5min = (rolling_one+rolling_two+rolling_three) / 3

                # rate of change 5 min rvol
                rvol_one = metrics[x].five_min_relative_volume
                rvol_two = metrics[x-1].five_min_relative_volume
                rate_of_change_rvol = (rvol_one - rvol_two) / rvol_two * 100


                if (
                    # 59 trades at 72% success rate
                    #metrics[x].price_change_24hr < -5 and
                    #metrics[x].rolling_relative_volume >= 2.1 and
                    #metrics[x].price_change_5min < 0 and
                    #metrics[x].price_change_10min < 0 and
                    #metrics[x].price_change_1hr > 0 and
                    #metrics[x-1].price_change_1hr < metrics[x].price_change_1hr and
                    #metrics[x-2].price_change_1hr < metrics[x-1].price_change_1hr

                    # 122 trades at 59% success rate
                    #metrics[x].price_change_24hr < -5 and
                    #metrics[x].rolling_relative_volume >= 2.1 and
                    #metrics[x-1].price_change_5min < metrics[x].price_change_5min and
                    #metrics[x-2].price_change_5min < metrics[x-1].price_change_5min and
                    #metrics[x-1].price_change_10min < metrics[x].price_change_10min and
                    #metrics[x].price_change_1hr > 0 and
                    #metrics[x-1].price_change_1hr < metrics[x].price_change_1hr and
                    #metrics[x-2].price_change_1hr < metrics[x-1].price_change_1hr

                    # 145 trades at 61% success rate
                    #metrics[x].price_change_24hr < -5 and
                    #metrics[x].rolling_relative_volume >= 2.1 and
                    #metrics[x-1].price_change_5min < metrics[x].price_change_5min and
                    #metrics[x-2].price_change_5min < metrics[x-1].price_change_5min and
                    #metrics[x].price_change_1hr > 0 and
                    #metrics[x-1].price_change_1hr < metrics[x].price_change_1hr and
                    #metrics[x-2].price_change_1hr < metrics[x-1].price_change_1hr

                    # 117 trades at 63% success rate
                    metrics[x].price_change_24hr < -5 and
                    metrics[x].rolling_relative_volume >= 2.1 and
                    metrics[x-1].price_change_5min < metrics[x].price_change_5min and
                    metrics[x-2].price_change_5min < metrics[x-1].price_change_5min and
                    metrics[x].price_change_10min > 0 and
                    metrics[x].price_change_1hr > 0 and
                    metrics[x-1].price_change_1hr < metrics[x].price_change_1hr and
                    metrics[x-2].price_change_1hr < metrics[x-1].price_change_1hr

                ):
                    #print("-----TRIGGER THREE-------------")
                    #print(coin.symbol)
                    #print(metrics[x].timestamp)

                    amount_of_trades += 1
                    trigger_three_trades += 1

                    trigger_price = metrics[x].last_price
                    stop_loss_price = trigger_price - (trigger_price * decimal.Decimal(0.02))
                    take_profit_price = trigger_price + (trigger_price * decimal.Decimal(0.05))
                    take_profit_hit = False
                    stop_loss_hit = False
                    take_profit_timestamp = None
                    stop_loss_timestamp = None
                    success = False
                    try:
                        for y in range(x, len(metrics)):
                            if (metrics[y].last_price >= take_profit_price):
                                take_profit_hit = True
                                take_profit_timestamp = metrics[y].timestamp
                                break

                        for y in range(x, len(metrics)):
                            if (metrics[y].last_price <= stop_loss_price):
                                stop_loss_hit = True
                                stop_loss_timestamp = metrics[y].timestamp
                                break

                        if (take_profit_hit == True):
                            if (stop_loss_hit == True):
                                # compare timestamps
                                if (take_profit_timestamp < stop_loss_timestamp):
                                    # successful trade
                                    successful_trades += 1
                                    trigger_three_success += 1
                                    success = True
                                else:
                                    # failed trade
                                    failed_trades += 1
                            else:
                                # successful trade
                                successful_trades += 1
                                trigger_three_success += 1
                                success = True

                        if (take_profit_hit == False and stop_loss_hit == True):
                            # failed trade
                            failed_trades += 1

                        if (take_profit_hit == False and stop_loss_hit == False):
                            amount_of_trades -= 1
                            trigger_three_trades -= 1

                    except:
                        print("failed in trigger 3")

                    day = metrics[x].timestamp.day
                    if day == 15:
                        count_15 += 1
                    elif day == 16:
                        count_16 += 1
                    elif day == 17:
                        count_17 += 1
                    elif day == 18:
                        count_18 += 1
                    elif day == 19:
                        count_19 += 1
                    elif day == 20:
                        count_20 += 1
                    elif day == 21:
                        count_21 += 1
                    elif day == 22:
                        count_22 += 1
                    elif day == 23:
                        count_23 += 1
                        print("-----TRIGGER THREE-------------")
                        print(coin.symbol)
                        print(metrics[x].timestamp)


                # SHORT Trigger
                if (
                    metrics[x].daily_relative_volume > 1.2 and
                    metrics[x].rolling_relative_volume >= 1.85 and
                    metrics[x].price_change_5min < metrics[x-1].price_change_5min and
                    metrics[x-1].price_change_5min < metrics[x-2].price_change_5min and
                    metrics[x].price_change_1hr > 1 and
                    metrics[x].price_change_10min < metrics[x-1].price_change_10min and
                    metrics[x-1].price_change_10min < metrics[x-2].price_change_10min and
                    metrics[x].price_change_1hr < metrics[x-1].price_change_1hr and
                    metrics[x-1].price_change_1hr < metrics[x-2].price_change_1hr
                ):
                    #print("-----SHORT TRIGGER-------------")
                    #print(coin.symbol)
                    #print(metrics[x].timestamp)

                    amount_of_trades += 1
                    trigger_short_trades += 1
                    trigger_price = metrics[x].last_price
                    stop_loss_price = trigger_price + (trigger_price * decimal.Decimal(0.02))
                    take_profit_price = trigger_price - (trigger_price * decimal.Decimal(0.05))
                    take_profit_hit = False
                    stop_loss_hit = False
                    take_profit_timestamp = None
                    stop_loss_timestamp = None
                    try:
                        for y in range(x, len(metrics)):
                            if (metrics[y].last_price <= take_profit_price):
                                take_profit_hit = True
                                take_profit_timestamp = metrics[y].timestamp

                            if (metrics[y].last_price >= stop_loss_price):
                                stop_loss_hit = True
                                stop_loss_timestamp = metrics[y].timestamp

                        if (take_profit_hit == True):
                            if (stop_loss_hit == True):
                                # compare timestamps
                                if (take_profit_timestamp < stop_loss_timestamp):
                                    # successful trade
                                    successful_trades += 1
                                    trigger_short_success += 1
                                else:
                                    # failed trade
                                    failed_trades += 1
                            else:
                                # successful trade
                                successful_trades += 1
                                trigger_short_success += 1

                        if (take_profit_hit == False and stop_loss_hit == True):
                            # failed trade
                            failed_trades += 1

                        if (take_profit_hit == False and stop_loss_hit == False):
                            amount_of_trades -= 1
                            trigger_short_trades -= 1

                    except:
                        print("failed in short trigger")

                    day = metrics[x].timestamp.day
                    if day == 15:
                        count_15 += 1
                    elif day == 16:
                        count_16 += 1
                    elif day == 17:
                        count_17 += 1
                    elif day == 18:
                        count_18 += 1
                    elif day == 19:
                        count_19 += 1
                    elif day == 20:
                        count_20 += 1
                    elif day == 21:
                        count_21 += 1
                    elif day == 22:
                        count_22 += 1
                    elif day == 23:
                        count_23 += 1
                        print("-----TRIGGER SHORT-------------")
                        print(coin.symbol)
                        print(metrics[x].timestamp)


                if (
                    # 202 trades at 57% success rate
                    metrics[x].price_change_24hr < -5 and
                    (metrics[x].rolling_relative_volume >= 2.1 or metrics[x].daily_relative_volume >= 1.3) and
                    metrics[x].price_change_5min < 0 and
                    metrics[x].price_change_10min < 0 and
                    metrics[x].price_change_1hr > 0 and
                    metrics[x-1].price_change_1hr < metrics[x].price_change_1hr and
                    metrics[x-2].price_change_1hr < metrics[x-1].price_change_1hr
                ):

                    #print("-----TRIGGER FIVE-------------")
                    #print(coin.symbol)
                    #print(metrics[x].timestamp)

                    amount_of_trades += 1
                    trigger_five_trades += 1

                    trigger_price = metrics[x].last_price
                    stop_loss_price = trigger_price - (trigger_price * decimal.Decimal(0.02))
                    take_profit_price = trigger_price + (trigger_price * decimal.Decimal(0.05))
                    take_profit_hit = False
                    stop_loss_hit = False
                    take_profit_timestamp = None
                    stop_loss_timestamp = None
                    success = False
                    try:
                        for y in range(x, len(metrics)):
                            if (metrics[y].last_price >= take_profit_price):
                                take_profit_hit = True
                                take_profit_timestamp = metrics[y].timestamp
                                break

                        for y in range(x, len(metrics)):
                            if (metrics[y].last_price <= stop_loss_price):
                                stop_loss_hit = True
                                stop_loss_timestamp = metrics[y].timestamp
                                break

                        if (take_profit_hit == True):
                            if (stop_loss_hit == True):
                                # compare timestamps
                                if (take_profit_timestamp < stop_loss_timestamp):
                                    # successful trade
                                    successful_trades += 1
                                    trigger_five_success += 1
                                    success = True
                                else:
                                    # failed trade
                                    failed_trades += 1
                            else:
                                # successful trade
                                successful_trades += 1
                                trigger_five_success += 1
                                success = True

                        if (take_profit_hit == False and stop_loss_hit == True):
                            # failed trade
                            failed_trades += 1

                        if (take_profit_hit == False and stop_loss_hit == False):
                            amount_of_trades -= 1
                            trigger_five_trades -= 1

                    except:
                        print("failed in trigger 5")

                    day = metrics[x].timestamp.day
                    if day == 15:
                        count_15 += 1
                    elif day == 16:
                        count_16 += 1
                    elif day == 17:
                        count_17 += 1
                    elif day == 18:
                        count_18 += 1
                    elif day == 19:
                        count_19 += 1
                    elif day == 20:
                        count_20 += 1
                    elif day == 21:
                        count_21 += 1
                    elif day == 22:
                        count_22 += 1
                    elif day == 23:
                        count_23 += 1
                        print("-----TRIGGER FIVE-------------")
                        print(coin.symbol)
                        print(metrics[x].timestamp)


    print("Results: ")
    print(f"Amount of trades: {amount_of_trades}")
    print(f"Successful trades: {successful_trades}")
    print(f"Failed trades: {failed_trades}")
    print(f"Trigger One: {trigger_one_trades}")
    print(f"Trigger Two: {trigger_two_trades}")
    print(f"Trigger Three: {trigger_three_trades}")
    print(f"Trigger Short: {trigger_short_trades}")
    print(f"Trigger Five: {trigger_five_trades}")
    success_percentage = 0
    if (amount_of_trades != 0):
        success_percentage = (successful_trades / amount_of_trades) * 100
    print(f"Successful trade percentage: {success_percentage}%")

    trigger_one_success_percentage = 0
    if (trigger_one_trades != 0):
        trigger_one_success_percentage = (trigger_one_success / trigger_one_trades) * 100
    print(f"Trigger One Success: {trigger_one_success_percentage}%")

    trigger_two_success_percentage = 0
    if (trigger_two_trades != 0):
        trigger_two_success_percentage = (trigger_two_success / trigger_two_trades) * 100
    print(f"Trigger Two Success: {trigger_two_success_percentage}%")

    trigger_three_success_percentage = 0
    if (trigger_three_trades != 0):
        trigger_three_success_percentage = (trigger_three_success / trigger_three_trades) * 100
    print(f"Trigger Three Success: {trigger_three_success_percentage}%")

    trigger_short_success_percentage = 0
    if (trigger_short_trades != 0):
        trigger_short_success_percentage = (trigger_short_success / trigger_short_trades) * 100
    print(f"Trigger Short Success: {trigger_short_success_percentage}%")

    trigger_five_success_percentage = 0
    if (trigger_five_trades != 0):
        trigger_five_success_percentage = (trigger_five_success / trigger_five_trades) * 100
    print(f"Trigger Five Success: {trigger_five_success_percentage}%")

    print(f"Day 15: {count_15}")
    print(f"Day 16: {count_16}")
    print(f"Day 17: {count_17}")
    print(f"Day 18: {count_18}")
    print(f"Day 19: {count_19}")
    print(f"Day 20: {count_20}")
    print(f"Day 21: {count_21}")
    print(f"Day 22: {count_22}")
    print(f"Day 23: {count_23}")



def check_trigger_two():

    #coins = Coin.objects.all()
    coin = Coin.objects.get(symbol="XRP")

    metrics = Metrics.objects.filter(coin=coin).order_by('timestamp')

    variable = 0.5
    best_variable = variable
    best_success_rate = 0

    for i in range(1, 200):

        amount_of_trades = 0
        successful_trades = 0
        failed_trades = 0
        success_rate = 0

        average_volume = 0
        for metric in metrics:
            average_volume += metric.volume_24h

        average_volume = average_volume / len(metrics)
        average_volume = average_volume * decimal.Decimal(1.15)

        for x in range(6, len(metrics)):

            if (metrics[x].rolling_relative_volume != None and
                metrics[x].price_change_5min != None and
                metrics[x].price_change_10min != None and
                metrics[x].price_change_1hr != None and
                metrics[x].price_change_24hr != None and
                metrics[x].daily_relative_volume != None and
                metrics[x].five_min_relative_volume != None and
                metrics[x].twenty_min_relative_volume != None):

                # 24 hour volume growth
                # current volume - volume 5 min ago / volume 5 min ago * 100
                current_volume = metrics[x].volume_24h
                previous_volume = metrics[x-1].volume_24h
                volume_growth = (current_volume - previous_volume) / previous_volume * 100

                # 5 min relative volume progression
                # current and previous 2 are increasing or equivalent
                rvol_progression = False
                current_rvol = metrics[x].five_min_relative_volume
                one_previous_rvol = metrics[x-1].five_min_relative_volume
                two_previous_rvol = metrics[x-2].five_min_relative_volume
                if (two_previous_rvol <= one_previous_rvol <= current_rvol):
                    rvol_progression = True

                # 5 min price change is greater than previous
                five_min_price_increase = False
                current_five_min = metrics[x].price_change_5min
                previous_five_min = metrics[x-1].price_change_5min
                previous_five_min_two = metrics[x-2].price_change_5min
                if (previous_five_min < current_five_min and
                    previous_five_min < 0 and
                    current_five_min > 0):
                    five_min_price_increase = True

                # 5 min and 10 min price changes go negative, positive, positive
                ten_min_price_increase = False
                current_ten_min = metrics[x].price_change_10min
                previous_ten_min = metrics[x-1].price_change_10min
                previous_ten_min_two = metrics[x-2].price_change_10min
                if (previous_ten_min < current_ten_min and
                    previous_ten_min < 0):
                    ten_min_price_increase = True

                # rolling 5 min price change
                rolling_one = metrics[x].price_change_5min
                rolling_two = metrics[x-1].price_change_5min
                rolling_three = metrics[x-2].price_change_5min
                rolling_price_change_5min = (rolling_one+rolling_two+rolling_three) / 3

                # rate of change 5 min rvol
                rvol_one = metrics[x].five_min_relative_volume
                rvol_two = metrics[x-1].five_min_relative_volume
                rate_of_change_rvol = (rvol_one - rvol_two) / rvol_two * 100


                if (
                    rolling_price_change_5min > variable
                    #metrics[x].volume_24h > average_volume and
                    #metrics[x].five_min_relative_volume > 1.4 and
                    #rate_of_change_rvol > 1.6
                ):


                    amount_of_trades += 1

                    trigger_price = metrics[x].last_price
                    stop_loss_price = trigger_price - (trigger_price * decimal.Decimal(0.02))
                    take_profit_price = trigger_price + (trigger_price * decimal.Decimal(0.05))
                    take_profit_hit = False
                    stop_loss_hit = False
                    take_profit_timestamp = None
                    stop_loss_timestamp = None
                    try:
                        for y in range(x, len(metrics)):
                            if (metrics[y].last_price >= take_profit_price):
                                take_profit_hit = True
                                take_profit_timestamp = metrics[y].timestamp

                            if (metrics[y].last_price <= stop_loss_price):
                                stop_loss_hit = True
                                stop_loss_timestamp = metrics[y].timestamp

                        if (take_profit_hit == True):
                            if (stop_loss_hit == True):
                                # compare timestamps
                                if (take_profit_timestamp < stop_loss_timestamp):
                                    # successful trade
                                    successful_trades += 1
                                else:
                                    # failed trade
                                    failed_trades += 1
                            else:
                                # successful trade
                                successful_trades += 1

                        if (take_profit_hit == False and stop_loss_hit == True):
                            # failed trade
                            failed_trades += 1

                        if (take_profit_hit == False and stop_loss_hit == False):
                            amount_of_trades -= 1



                    except:
                        print("failed in trigger 3")


        success_percentage = 0
        if (amount_of_trades != 0):
            success_percentage = (successful_trades / amount_of_trades) * 100

        if (success_percentage > best_success_rate):
            best_variable = variable
            best_success_rate = success_percentage

        variable += 0.01

    print("Results: ")
    print(f"Amount of trades: {amount_of_trades}")
    print(f"Successful trades: {successful_trades}")
    print(f"Failed trades: {failed_trades}")
    print(f"Successful trade percentage: {best_success_rate}%")
    print(f"Final variable value: {best_variable}%")





# used to test if the telegram bot is sending messages properly
def test_message():

    chat_id_danny = '1077594551'
    chat_id_ricki = '1054741134'
    chat_ids = [chat_id_danny, chat_id_ricki]
    bot_token = '7672687080:AAFWvkwzp-LQE92XdO9vcVa5yWJDUxO17yE'
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    message = " test "

    for chat_id in chat_ids:

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


# bot message notificagtions
def send_text(true_triggers_two):

    if len(true_triggers_two) > 0:

        # telegram bot information
        chat_id_danny = '1077594551'
        chat_id_ricki = '1054741134'
        chat_ids = [chat_id_danny, chat_id_ricki]
        #chat_ids = [chat_id_danny]
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


# takes a trigger string and returns True if it already exists in the database
def check_duplicate_triggers(trigger):

    exists = Trigger.objects.filter(trigger_name=trigger).exists()

    if exists:
        return True
    else:
        return False


# deletes triggers in the database that have been there longer than an hour
def delete_old_triggers():

    # time an hour ago
    cutoff_time = now() - timedelta(hours=1)

    # delete records older than 1 hour
    Trigger.objects.filter(timestamp__lt=cutoff_time).delete()

    # log the information
    deleted_count, _ = Trigger.objects.filter(timestamp__lt=cutoff_time).delete()
    print(f"{deleted_count} triggers deleted.")

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


def load_coin_exchanges():

    URL = "https://pro-api.coinmarketcap.com/v1/exchange/info"
    API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'

    headers = {
        "X-CMC_PRO_API_KEY": API_KEY,
        "Accept": "application/json",
    }

    specific_coins = ["PHA","GMT","BAN","BLUE","FWOG","VIRTUAL","TOKEN","W","ORDER","NS","FORTH","GRASS","ACX","F","SSV","THE","USTC","SUN","MERL","PUFFER","COOK","MORPHO","L3","ZBCN","FIRE","AI","HIPPO","DAR","POWRBOBA","SUNDOG","LUNA2","ETHFI","REZ","REN","LDO","MDT","AUDIO","FXS","1CAT","PENDLE","XRD","STEEM","TOSHI","AIDOGE","UNI","PORTAL","PEOPLE","LUNC","CLOUD","DYDX","GIGA","XEM","ID","PRIME","ETHW","EDU","PHB","CATI","MEW","APE","RPL","LEVER","XCN","PEAQ","JST","BONK","REEF","ICX","XAI","PIXEL","TRX","CATS","IOTX","BOME","POPCAT","DOP1","RATS","OGN","JTO","IMX","MAVIA","MAV","UXLINK","CTC","METIS","HNT","MAJOR","LIT","BENDOG","KNC","BRETT","MYRO","KAVA","FARTCOIN","OP","KDA","MKR","GMX","ASTR","IO","SATS","REQ","GODS","MOCA","DBR","RAYDIUM","AIOZ","SYN","ZRX","MANEKI","SWELL","UMA","IDEX","LINA","BADGER","PEIPEI","RUNE","SNT","SYS","ZRC","GLM","ENS","STMX","CFX","KEY","BAKE","BNX","OM","AUCTION","GAS","SAGA","VOXEL","LISTA","LSK","SC","OL","KAS","TNSR","ALT","SCRT","FLOKI","MTL","AMB","ARB","CORE","XMR","ORDI","STRK","RSS3","BCH","CYBER","ALPHA","CRV","RARE","DODO","YGG","MEME","VRA","ONG","NFP","LAI","NYAN","SPELL","ARK","BIGTIME","POLYX","OMNI","WOO","HOT","PERP","ACH","DYM","FLOW","BICO","ADA","C98","HIFI","MAGIC","CTK","BSW","ARPA","BLUR","DATA","ZETA","AR","CVX","COMBO","FLUX","SXP","AXS","MINA","WLD","DOGS","CHILLGUY","MASK","FIDA","TLM","BANANA","DOG","JOE","HOOK","CAKE","QI","COS","TRB","XVS","MANTA","NULS","DEGEN","A8","CELO","AVAIL","API3","NTRN","RDNT","YFI","NOT","EIGEN","SLF","SNX","MNT","FTN","POL","CVC","WAXP","CKB","SILLY","BAL","FLM","RIF","ETC","ORBS","CHZ","SLERF","IOTA","ZIL","NEO","OXT","MBL","STX","KSM","1INCH","ILV","MAX","RON","VANRY","CRO","ACE","TAI","AGLD","NEAR","EGLD","T","ANKR","ZK","NKN","GTC","CTSI","NMR","PYTH","CHESS","TON","BNB","XION","ALICE","ARKM","PAXG","ONT","QTUM","FOXY","OMG","OSMO","TAO","NEIROETH","HFT","MANA","GLMR","ROSE","TWT","QUICK","RVN","IOST","SKL","AEVO","ETH","SEP","WAVES","WIF","THETA","COMP","BEL","STORJ","EOS","LRC","GRIFFAIN","GRT","ATOM","GALA","SEND","COTI","AGI","ENJ","G","HMSTR","DENT","DUSK","RSR","CHR","BAND","FIL","XRP","DOGE","KAIA","TRU","DOT","SLP","BSV","TAIKO","STG","VTHO","MOVR","ONDO","BTC","LUMIA","FB","LOOKS","CELR","DGB","SUSHI","LTC","AXL","BEAM","SAND","SEI","MYRIA","ENA","XTZ","LINK","VELODROME","INJ","APT","SOL","TIA","ICP","KMNO","AKT","RENDER","LUCE","VET","AVAX","XLM","SUI","STPT","MOBILE","BLAST","PNUT","SPEC","RAD","BAT","SUPER","ACT","JUP","SAFE","ALEO","PIRATE","FTM","DASH","ZRO","CETUS","ALGO","AAVE","TROY","ONE","XNO","DEEP","ZEUS","MOODENG","HBAR","PRCL","CARV","ATH","JASMY","GEMS","GME","GOAT","AIXBT","LQTY","MON","DRIFT","XVG","MOVE","PENGU","ZEC","SPX","LPT","MOTHER","COW","VELO","ZEN","URO","RIFSOL","DEXE","MASA","PEPE","BTT","XEC","SHIB","LADYS","X","BABYDOGE","NEIROCTO","WEN","MOG","CAT","TURBO"]

    #cmc_ids = Coin.objects.values_list('cmc_id', flat=True)
    #specific_coins.append(cmc_ids[0])

    chunk_size = 100

    for i in range(0, len(specific_coins), chunk_size):

        symbol_batch = specific_coins[i:i + chunk_size]
        params = {
            "symbol": ",".join(symbol_batch),
        }

        current_symbol = specific_coins[i]

        try:
            response = requests.get(URL, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()

            market_pairs = data.get("data", {})

            if len(market_pairs) > 0:
                market_pair = market_pairs[0]
                exchange = market_pair["name"]

                print(exchange)

            else:
                print("didn't find any exchange data")
                exchange = "BINANCE"

            exchange = exchange.upper()
            exchange_info = exchange + ":" + current_symbol + "USDT"

            coin = Coin.objects.get(symbol=current_symbol)

            # Update the exchange value
            coin.exchange = exchange_info
            coin.save()

            print("Top coins fetched and updated successfully.")

        except Exception as e:
            print(f"Error fetching data: {e}")


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

            data = response.json().get("data", {})

            top_cryptos = []

            for symbol in specific_coins:
                coin_data = data.get(symbol)
                if coin_data:
                    name = coin_data.get("name")
                    price = coin_data["quote"]["USD"].get("price")
                    market_cap = coin_data["quote"]["USD"].get("market_cap")

                    market_pairs = coin_data.get("market_pairs", [])

                    exchanges = [
                        pair["exchange"]["name"]
                        for pair in market_pairs
                        if pair.get("quote_currency", {}).get("symbol") == "USDT"
                    ]

                    if len(exchanges) > 0:
                        exchange = exchanges[0]
                    else:
                        exchange = "BINANCE"

                    exchange = exchange.upper()
                    exchange_info = exchange + ":" + coin_data.get("symbol") + "USDT"

                    Coin.objects.update_or_create(
                        name=name,
                        defaults={
                            "symbol": coin_data.get("symbol"),
                            "market_cap_rank": coin_data.get("cmc_rank"),
                            "last_updated": datetime.strptime(
                                coin_data.get("last_updated"), "%Y-%m-%dT%H:%M:%S.%fZ"
                            ),
                            "cmc_id": coin_data.get("id"),
                            "exchange": exchange_info,
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

    # DOING 30 MIN PRICE CHANGE NOT 10 MIN

    # (price now - price 30 min ago / price 30 min ago) * 100

    price_change_30min = None

    prices = ShortIntervalData.objects.filter(coin=coin).order_by('-timestamp')[:7]

    if (len(prices) == 7):

        price_now = prices[0].price
        price_thirty_min_ago = prices[6].price

        if price_now != None and price_thirty_min_ago != None:

            price_difference = price_now - price_thirty_min_ago
            price_change_30min = (price_difference / price_thirty_min_ago) * 100 if price_thirty_min_ago != 0 else None
            return price_change_30min

    else:
        return price_change_30min


def calculate_twenty_min_relative_volume(coin):

    twenty_min_relative_volume = None
    volumes = ShortIntervalData.objects.filter(coin=coin).order_by('-timestamp')[:5]

    if (len(volumes) == 5):

        volume_now = volumes[0].volume_5min
        remaining_volumes = volumes[1:]

        sum = 0
        for volume in remaining_volumes:
            sum += volume.volume_5min

        average = sum / len(remaining_volumes)
        twenty_min_relative_volume = (volume_now / average) if average != 0 else None
        return twenty_min_relative_volume

    else:
        return twenty_min_relative_volume


def calculate_five_min_relative_volume(coin):

    five_min_relative_volume = None

    volumes = ShortIntervalData.objects.filter(coin=coin).order_by('-timestamp')[:2]

    if len(volumes) == 2:

        volume_now = volumes[0].volume_5min
        previous_volume = volumes[1].volume_5min

        if (volume_now != None and previous_volume != None):
            five_min_relative_volume = (volume_now / previous_volume)
            return five_min_relative_volume

    else:
        print("problem in five min relative volume")
        print(coin.symbol)
        return None


# ======================================================================


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
    #now = datetime.now()
    #if now.hour == 0 and now.minute <= 5:
        #manually_clean_database()
        #print("not deleting right now...")

    # delete old Triggers
    delete_old_triggers()

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
        cmc_id_batch = cmc_ids[i:i + batch_size]
        params = {
            "id": ",".join(map(str, cmc_id_batch)),
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
                    except Exception as e:
                        print("FAILED IN GROUP 1")
                        print(e)

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
                    except Exception as e:
                        print("FAILED IN GROUP 2")
                        print(e)

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

                    now = datetime.now()

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
                            print("Created new historical data")

                        except Exception as e:
                            print("Couldn't create new historical data")
                            print(e)

        except Exception as e:
            print(f"Error updating tracked coins for batch {cmc_id_batch}: {e}")


    # wait 30 seconds before checking solana
    #print("pausing for 30 seconds before solana check")
    #time.sleep(30)
    #print("checking solana")
    #check_new_solana_listings()
    #fetch_memecoin_metrics()
    #print("done fetching solana data")
    #print("checking meme triggers")
    #meme_coin_triggers()
    #print("done checking meme triggers")


    if request:
        return JsonResponse({"status": "success", "message": "Update triggered successfully"})


def index_original(request):

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
        coin_exchange = coin.exchange

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

        triggerOne = False
        triggerTwo = False
        triggerThree = False
        triggerFour = False
        triggerFive = False
        triggerSix = False
        triggerSeven = False
        triggerEight = False
        triggerNine = False
        triggerTen = False

        '''
        # TRIGGER ONE - price up 10% or more in last 24 hours
        triggerOne = False
        if coin_price_change_24h_percentage != None:
            if coin_price_change_24h_percentage >= 10:
                triggerOne = True
                trigger = coin.symbol + " : Price Change > 10% in 24 hours"
                #true_triggers.append(trigger)

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
        remaining_circulating_supplies = []
        if len(circulating_supplies) > 2:
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
                #true_triggers_two.append(trigger)




        # TRIGGER TEN -
        # rolling_relative_volume > 2.5 and increasing over 2 intervals
        # five_min_relative_volume > 1.7 for 2 consecutive intervals
        # price_change_10min > 3%
        # volume_24h increases by 10% within the last 30 minutes
        triggerTen = False
        if coin_rolling_relative_volume != None and coin_five_min_relative_volume != None and coin_price_change_10min != None:
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

        '''





        # updated triggers ------------------------------------------

        # PRIMARY TRIGGER - HIGH CONFIDENCE
        # rolling rvol increased by 5-10% over 15-30 min period
        # 5 min rvol > 1.3 or 20 min rvol > 1.0
        # 5 min and 10 min price change are positive, while 1hr or
        # 24hr price change is negative -5% to -20%

        # rolling rvol change > 5% over 30 min
        # rvol now - rvol 30 min ago / rvol 30 min ago * 100
        primary_trigger_metrics = Metrics.objects.filter(coin=coin).order_by('-timestamp')[:8]

        # Ensure we have enough data points
        if len(primary_trigger_metrics) > 6:
            if hasattr(primary_trigger_metrics[0], 'rolling_relative_volume') and primary_trigger_metrics[0].rolling_relative_volume != None:
                if hasattr(primary_trigger_metrics[6], 'rolling_relative_volume') and primary_trigger_metrics[6].rolling_relative_volume != None:


                    primary_trigger_rvol_now = primary_trigger_metrics[0].rolling_relative_volume
                    primary_trigger_rvol_30min_ago = primary_trigger_metrics[6].rolling_relative_volume

                    # Calculate percentage change in rolling relative volume
                    if primary_trigger_rvol_30min_ago != 0:
                        primary_trigger_rvol_change = ((primary_trigger_rvol_now - primary_trigger_rvol_30min_ago) / primary_trigger_rvol_30min_ago) * 100
                    else:
                        primary_trigger_rvol_change = 0

                    # Trigger conditions

                    if (primary_trigger_rvol_change != None and
                        primary_trigger_metrics[0].five_min_relative_volume != None and
                        primary_trigger_metrics[0].twenty_min_relative_volume != None and
                        primary_trigger_metrics[0].price_change_5min != None and
                        primary_trigger_metrics[0].price_change_10min != None and
                        primary_trigger_metrics[0].price_change_1hr != None and
                        primary_trigger_metrics[0].price_change_24hr != None):


                        if primary_trigger_rvol_change > 5:
                            if (primary_trigger_metrics[0].five_min_relative_volume > 1.3 or
                                primary_trigger_metrics[0].twenty_min_relative_volume > 1.0):

                                if (primary_trigger_metrics[0].price_change_5min > 0.0 and
                                    primary_trigger_metrics[0].price_change_10min > 0.0):
                                    if (primary_trigger_metrics[0].price_change_1hr < -5 or
                                        primary_trigger_metrics[0].price_change_24hr < -5):

                                        # Primary trigger identified
                                        primary_trigger = f"{coin.symbol} : Primary Trigger Hit | "
                                        if primary_trigger_rvol_change > 10:
                                            primary_trigger += " (rvol > 10% !)"
                                        else:
                                            primary_trigger += " (rvol > 5% !)"

                                        exists = check_duplicate_triggers(primary_trigger)

                                        if exists == False:

                                            true_triggers_two.append(primary_trigger)

                                            # create and save the new Trigger element
                                            try:
                                                Trigger.objects.create(trigger_name=primary_trigger, timestamp=now())

                                            except Exception as e:
                                                print(f"Error creating new Trigger: {e}")







        # SECONDARY TRIGGER - MEDIUM CONFIDENCE
        # 24hr volume increasing steadily by 5-10% over 1-2 hours
        # market cap is increasing
        # 7day price change down 20% or more (oversold)
        secondary_trigger_metrics = Metrics.objects.filter(coin=coin).order_by('-timestamp')[:14]
        secondary_trigger_volumes = [metric.volume_24h for metric in secondary_trigger_metrics]
        tolerance = 0

        for i in range(1, len(secondary_trigger_metrics)):
            # Calculate time difference in hours
            time_diff = (secondary_trigger_metrics[i - 1].timestamp - secondary_trigger_metrics[i].timestamp).total_seconds() / 3600.0

            # Check if the time difference is 1 hour
            if time_diff >= 1:
                # Calculate percentage change
                prev_volume = secondary_trigger_metrics[i].volume_24h
                curr_volume = secondary_trigger_metrics[i - 1].volume_24h
                if prev_volume != 0:
                    percentage_change = (curr_volume - prev_volume) / prev_volume * 100
                else:
                    percentage_change = 0

                # If the increase is greater than the threshold
                if percentage_change > 5:
                    # Check for increasing market cap
                    secondary_trigger_market_caps = [metric.market_cap for metric in secondary_trigger_metrics]

                    # Check for steady increase
                    for j in range(1, len(secondary_trigger_market_caps)):
                        try:
                            market_cap_change = (secondary_trigger_market_caps[j] - secondary_trigger_market_caps[j - 1]) / secondary_trigger_market_caps[j - 1] * 100

                            if market_cap_change < -tolerance:
                                if secondary_trigger_metrics[0].price_change_7d < -20:
                                    secondary_trigger = coin.symbol + " : Secondary Trigger Hit !"

                                    exists = check_duplicate_triggers(secondary_trigger)

                                    if exists == False:

                                        true_triggers_two.append(secondary_trigger)

                                        # create and save the new Trigger element
                                        try:
                                            Trigger.objects.create(trigger_name=secondary_trigger, timestamp=now())

                                        except Exception as e:
                                            print(f"Error creating new Trigger: {e}")

                        except:
                            print("FAILED TO CALCULATE MARKET CAP PERCENTAGE CHANGE")


        # AMPLIFYING TRIGGER - HIGH CONFIDENCE
        # 5 min or 10 min price change is increasing
        # 5 min rvol > 1.3

        window_minutes = 30
        threshold = 0.2

        # Ensure metrics are ordered by timestamp (oldest to newest)
        metrics = Metrics.objects.filter(coin=coin).order_by('-timestamp')[:8][::-1]

        if len(metrics) > 0:
            latest_time = secondary_trigger_metrics[0].timestamp
            window_start = latest_time - timedelta(minutes=window_minutes)

            # Filter metrics within the time window
            filtered_metrics = [m for m in metrics if m.timestamp >= window_start]

            # Check for increasing 5-minute and 10-minute price changes
            for i in range(1, len(filtered_metrics)):
                prev = filtered_metrics[i - 1]
                curr = filtered_metrics[i]

                # Ensure both 5 and 10-minute price changes are increasing
                if (
                    prev.price_change_5min is not None and
                    curr.price_change_5min is not None and
                    prev.price_change_10min is not None and
                    curr.price_change_10min is not None and
                    curr.price_change_5min > prev.price_change_5min + threshold and
                    curr.price_change_10min > prev.price_change_10min + threshold
                ):

                    # Check for relative volume
                    if secondary_trigger_metrics[0].five_min_relative_volume > 1.5:
                        amplifying_trigger = coin.symbol + " : Amplifying Trigger Hit !"

                        exists = check_duplicate_triggers(amplifying_trigger)

                        if exists == False:

                            true_triggers_two.append(amplifying_trigger)

                            # create and save the new Trigger element
                            try:
                                Trigger.objects.create(trigger_name=amplifying_trigger, timestamp=now())

                            except Exception as e:
                                print(f"Error creating new Trigger: {e}")


        # MEGA TRIGGER 1 : Momentum + Volume Surge -----------------------------

        # Rolling RVOL > 1.3 (to confirm overall heightened trading activity).
        # Five-minute RVOL > 1.35 (short-term activity spike).
        # 5-minute price change > 0.3% (indicating immediate upward momentum).
        # 10-minute price change > 0.5% (momentum sustained over a slightly longer period).
        # 24-hour volume growth > 5% within the last hour (confirming increasing interest in the coin).

        # secondary_trigger_metrics has most recent 14 metrics, most recent to oldest

        if len(secondary_trigger_metrics) > 0:

            if (secondary_trigger_metrics[0].rolling_relative_volume != None and
                secondary_trigger_metrics[0].five_min_relative_volume != None and
                secondary_trigger_metrics[0].price_change_5min != None and
                secondary_trigger_metrics[0].price_change_10min != None):

                if (secondary_trigger_metrics[0].rolling_relative_volume > 1.3 and
                    secondary_trigger_metrics[0].five_min_relative_volume > 1.35 and
                    secondary_trigger_metrics[0].price_change_5min > 0.3 and
                    secondary_trigger_metrics[0].price_change_10min > 0.5):

                    # check 24 hour volume growth over last hour
                    # vol now - vol hour aga / vol hour ago * 100
                    volume_now = secondary_trigger_metrics[0].volume_24h
                    if len(secondary_trigger_metrics) > 12:
                        volume_an_hour_ago = secondary_trigger_metrics[12].volume_24h
                        x = volume_now - volume_an_hour_ago
                        volume_change = (x / volume_an_hour_ago) * 100

                        if volume_change > 5:
                            mega_trigger_1 = coin.symbol + " : Mega Trigger 1 Hit !"

                            exists = check_duplicate_triggers(mega_trigger_1)

                            if exists == False:

                                true_triggers_two.append(mega_trigger_1)

                                # create and save the new Trigger element
                                try:
                                    Trigger.objects.create(trigger_name=mega_trigger_1, timestamp=now())

                                except Exception as e:
                                    print(f"Error creating new Trigger: {e}")


                # MEGA TRIGGER 2 : Capital Flow + Momentum Confirmation ----------------

                # Rolling RVOL > 1.3 (to confirm sustained interest in the coin).
                # Market cap increases by ≥ 0.5% within 10 minutes (capital inflow validation).
                # 5-minute price change > 0.3% (indicating immediate bullish momentum).
                # 1-hour price change > 0.5% (indicating the trend is part of a larger movement).

                if (secondary_trigger_metrics[0].rolling_relative_volume > 1.3):

                    # market cap increase by over 0.5% in ten minutes
                    market_cap_now = secondary_trigger_metrics[0].market_cap
                    if len(secondary_trigger_metrics) > 2:
                        market_cap_ten_min_ago = secondary_trigger_metrics[2].market_cap
                        y = market_cap_now - market_cap_ten_min_ago
                        if market_cap_ten_min_ago and market_cap_ten_min_ago != 0:
                            market_cap_percent_change = (y / market_cap_ten_min_ago) * 100
                        else:
                            market_cap_percent_change = 0

                        if (market_cap_percent_change >= 0.5 and
                            secondary_trigger_metrics[0].price_change_5min > 0.3 and
                            secondary_trigger_metrics[0].price_change_1hr > 0.5):

                            mega_trigger_2 = coin.symbol + " : Mega Trigger 2 Hit !"

                            exists = check_duplicate_triggers(mega_trigger_2)

                            if exists == False:

                                true_triggers_two.append(mega_trigger_2)

                                # create and save the new Trigger element
                                try:
                                    Trigger.objects.create(trigger_name=mega_trigger_2, timestamp=now())

                                except Exception as e:
                                    print(f"Error creating new Trigger: {e}")


            # PRICE MOVEMENT TRIGGER
            # 5 min price change > 0.5
            # 10 min price change > 0.75
            # 1 hr price change > 1.0
            if (secondary_trigger_metrics[0].price_change_5min != None and
                secondary_trigger_metrics[0].price_change_10min != None and
                secondary_trigger_metrics[0].price_change_1hr != None):

                if (secondary_trigger_metrics[0].price_change_5min > 0.5 and
                    secondary_trigger_metrics[0].price_change_10min > 0.75 and
                    secondary_trigger_metrics[0].price_change_1hr > 1.0):

                    price_trigger = coin.symbol + " : Price Trigger Hit !"

                    exists = check_duplicate_triggers(price_trigger)

                    if exists == False:

                        true_triggers_two.append(price_trigger)

                        # create and save the new Trigger element
                        try:
                            Trigger.objects.create(trigger_name=price_trigger, timestamp=now())

                        except Exception as e:
                            print(f"Error creating new Trigger: {e}")




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
                #"circulating_supply_change": circulating_supply_change,
                "daily_relative_volume": coin_daily_relative_volume,
                "rolling_relative_volume": coin_rolling_relative_volume,
                "five_min_relative_volume": coin_five_min_relative_volume,
                "twenty_min_relative_volume": coin_twenty_min_relative_volume,
                "price_change_5min": coin_price_change_5min,
                "price_change_10min": coin_price_change_10min,
                "exchange": coin_exchange,
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
                    "exchange": coin_exchange,
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
            sorted_volumes = daily_relative_volumes

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


    # get all the triggers from the database
    current_triggers = Trigger.objects.all()
    triggers = []

    for trigger in current_triggers:
        triggers.append({
            "trigger_name": trigger.trigger_name,
            "timestampe": trigger.timestamp,
        })



    # If this is an Ajax automatic refresh:
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':  # Check if it's an AJAX request

        data = {
            "top_cryptos": sorted_coins,  # Ensure this is serializable
            "sorted_volumes": sorted_volumes,  # Ensure this is serializable
            "triggers": triggers,
        }
        return JsonResponse(data, safe=False)


    # Render data to the HTML template
    return render(request, "index.html", {
        "top_cryptos": sorted_coins,
        "sorted_volumes": sorted_volumes,
        "triggers": triggers,
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
    start_time = event_time - timedelta(minutes=15)
    end_time = event_time + timedelta(minutes=15)

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
            print("symbol: " + str(metric["Metric Name"]))
            print("timestamp: " + str(metric["Timestamp"]))
            print("daily rvol: " + str(metric["daily rvol"]))
            print("rolling rvol: " + str(metric["rolling rvol"]))
            print("five min rvol: " + str(metric["five min rvol"]))
            print("20 min rvol: " + str(metric["20 min rvol"]))
            print("5 min price change: " + str(metric["5 min price change"]))
            print("10 min price change: " + str(metric["10 min price change"]))
            print("1hr price change: " + str(metric["1hr price change"]))
            print("24hr price change: " + str(metric["24hr price change"]))
            print("7day price change: " + str(metric["7day price change"]))
            print("circulaating supply: " + str(metric["circulaating supply"]))
            print("24hr volume: " + str(metric["24hr volume"]))
            print("last price: " + str(metric["last price"]))
            print("market cap: " + str(metric["market cap"]))


        # Return the data as JSON
        return JsonResponse({"status": "success", "coin": coin_name, "metrics": metrics_data})

    except Coin.DoesNotExist:
        # Handle case where the coin does not exist
        return JsonResponse({"status": "error", "message": f"Coin '{coin_name}' not found."})

    except Exception as e:
        # Handle other exceptions
        return JsonResponse({"status": "error", "message": str(e)})


def find_metrics():

    try:

        # Get all coins
        coins = Coin.objects.all()

        for coin in coins:
            # Get metrics for the coin, ordered by timestamp
            metrics = Metrics.objects.filter(coin=coin).order_by('timestamp')

            for metric in metrics:

                #price_change = metric.price_change_10min
                price_change = metric.price_change_1hr

                if price_change >= 5:

                    earlier_metric = Metrics.objects.filter(
                        coin=coin,
                        timestamp=metric.timestamp - timedelta(minutes=15)
                    ).first()

                    relevant_metrics = [metric, earlier_metric]

                    for metric in relevant_metrics:

                        print("-------------------------------------------")
                        print(f"Price spike detected for {coin.symbol}")
                        print("timestamp: " + str(metric.timestamp))
                        print("daily rvol: " + str(round(metric.daily_relative_volume, 2)))
                        print("rolling rvol: " + str(round(metric.rolling_relative_volume, 2)))
                        print("five min rvol: " + str(round(metric.five_min_relative_volume, 2)))
                        print("20 min rvol: " + str(round(metric.twenty_min_relative_volume, 2)))
                        print("5 min price change: " + str(round(metric.price_change_5min, 2)))
                        print("10 min price change: " + str(round(metric.price_change_10min, 2)))
                        print("1hr price change: " + str(round(metric.price_change_1hr, 2)))
                        print("24hr price change: " + str(round(metric.price_change_24hr, 2)))
                        print("7day price change: " + str(round(metric.price_change_7d, 2)))
                        print("circulating supply: " + str(metric.circulating_supply))
                        print("24hr volume: " + str(round(metric.volume_24h, 2)))
                        print("last price: " + str(round(metric.last_price, 4)))
                        print("market cap: " + str(metric.market_cap))


        # Return the data as JSON
        return JsonResponse({"status": "success"})

    except Coin.DoesNotExist:
        # Handle case where the coin does not exist
        return JsonResponse({"status": "error", "message": f"Coin '{coin_name}' not found."})

    except Exception as e:
        # Handle other exceptions
        return JsonResponse({"status": "error", "message": str(e)})


import csv
def print_metrics(symbol):

    try:
        # Fetch coin by symbol
        coin = Coin.objects.get(symbol=symbol)

        # Fetch metrics ordered by timestamp
        metrics = Metrics.objects.filter(coin=coin).order_by('timestamp')

        csv_file_name = f"{symbol}_metrics.csv"
        with open(csv_file_name, mode='w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerow([
                "Timestamp", "Daily RVOL", "Rolling RVOL", "5-Min RVOL",
                "20-Min RVOL", "5-Min Price Change", "10-Min Price Change",
                "1-Hour Price Change", "24-Hour Price Change", "7-Day Price Change",
                "Circulating Supply", "24-Hour Volume", "Last Price", "Market Cap"
            ])

        #with open('output.txt', 'w') as f:
            for metric in metrics:
                csvwriter.writerow([
                    metric.timestamp,
                    round(metric.daily_relative_volume, 2),
                    round(metric.rolling_relative_volume, 2),
                    round(metric.five_min_relative_volume, 2),
                    round(metric.twenty_min_relative_volume, 2),
                    round(metric.price_change_5min, 2),
                    round(metric.price_change_10min, 2),
                    round(metric.price_change_1hr, 2),
                    round(metric.price_change_24hr, 2),
                    round(metric.price_change_7d, 2),
                    metric.circulating_supply,
                    round(metric.volume_24h, 2),
                    round(metric.last_price, 4),
                    metric.market_cap
                ])

        print(f"CSV file '{csv_file_name}' created successfully.")

    except Coin.DoesNotExist:
        print(f"Coin with symbol '{symbol}' does not exist.")

    except Exception as e:
        print(f"An error occurred: {e}")


def retrieve_metrics(symbol):

    api_key = '7dd5dd98-35d0-475d-9338-407631033cd9'
    base_url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"

    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": api_key,
    }

    #coins = ["ADA", "LINK", "HBAR", "SOL", "XRP"]
    coins = ["XRP"]

    batch_size = 5
    days = 5
    interval = 5
    price_change_threshold = 10

    for start in range(0, len(coins), batch_size):
        coins_batch = coins[start:start + batch_size]
        print(f"Processing batch {start + 1} to {min(start + batch_size, len(coins))}")

        for coin in coins_batch:
            print(f"Processing {coin}")

            # Define the time range for the last 10 days
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)

            # Fetch historical data
            params = {
                "symbol": coin,
                "interval": interval,
                "time_start": start_time.isoformat(),
                "time_end": end_time.isoformat(),
            }

            try:
                response = requests.get(base_url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()

            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for {coin}: {e}")
                continue

            # Process the historical data
            historical = data.get("data", {}).get("quotes", [])



            for i in range(len(historical) - 6):  # Compare points 30 minutes apart
                current = historical[i]
                future = historical[i + 6]
                # 5 min ago from future metric
                five_min_ago = historical[i + 5]
                # 10 min ago from future metric
                ten_min_ago = historical[i + 4]
                # 20 min ago from future metric
                twenty_min_ago = historical[i + 2]

                current_price = current["quote"]["USD"]["price"]
                future_price = future["quote"]["USD"]["price"]
                price_5min_ago = five_min_ago["quote"]["USD"]["price"]
                price_10min_ago = ten_min_ago["quote"]["USD"]["price"]
                volume_now = future["quote"]["USD"]["volume_24h"]
                volume_5min_ago = five_min_ago["quote"]["USD"]["volume_24h"]
                volume_20min_ago = twenty_min_ago["quote"]["USD"]["volume_24h"]

                # Calculate price change
                price_change = ((future_price - current_price) / current_price) * 100

                if price_change > price_change_threshold:
                    print(f"Significant price change detected for {coin}:")
                    print(f"Future timestamp: {future['timestamp']}")
                    print(f"Current Price: {current_price}")
                    print(f"Future Price: {future_price}")
                    print(f"Future volume: {future['quote']['USD']['volume_24h']}")
                    print(f"Future market Cap: {future['quote']['USD']['market_cap']}")

                    # 5 min price change
                    # price now - price 5 min ago / price 5 min ago * 100
                    x = future_price - price_5min_ago
                    price_change_5min = (x / price_5min_ago) * 100
                    print(f"Future's 5 min price change: {price_change_5min}")

                    # 10 min price change
                    # price now - price 10 min ago / price 10 min ago * 100
                    x = future_price - price_10min_ago
                    price_change_10min = (x / price_10min_ago) * 100
                    print(f"Future's 10 min price change: {price_change_10min}")

                    # 5 min relative volume
                    x = volume_now - volume_5min_ago
                    five_min_rvol = (x / volume_5min_ago) * 100
                    print(f"Future's 5 min rvol: {five_min_rvol}")

                    # 20 min relative volume
                    x = volume_now - volume_20min_ago_min
                    twenty_min_rvol = (x / volume_20min_ago_min) * 100
                    print(f"Future's 20 min rvol: {twenty_min_rvol}")

                    print(f"Future's 1hr price change: {future['quote']['USD']['percent_change_1h']}")
                    print(f"Future's 24hr price change: {future['quote']['USD']['percent_change_24h']}")
                    print(f"Future's 7d price change: {future['quote']['USD']['percent_change_7d']}")
                    print(f"Future's 24hr volume: {future['quote']['USD']['volume_24h']}")
                    print(f"Future's market cap: {future['quote']['USD']['market_cap']}")


def download_file(request, filename):
    try:
        filepath = f'./{filename}'  # Adjust path if the file is stored in a subdirectory
        return FileResponse(open(filepath, 'rb'), as_attachment=True)
    except FileNotFoundError:
        raise Http404("File does not exist")


# used to view coins and then delete the unneccessary ones
def print_coins():

    coins = Coin.objects.filter(market_cap_rank__gt=200)

    for coin in coins:
        # Delete related HistoricalData, ShortIntervalData, Metrics
        HistoricalData.objects.filter(coin=coin).delete()
        ShortIntervalData.objects.filter(coin=coin).delete()
        Metrics.objects.filter(coin=coin).delete()

        # Finally, delete the coin itself
        coin.delete()

    print(f"Deleted {coins.count()} coins and their associated data.")


def check_triggers(metrics_queryset):

    true_triggers = []

    trigger_passed = False

    if (metrics_queryset[0].rolling_relative_volume != None and
        metrics_queryset[0].price_change_5min != None and
        metrics_queryset[0].price_change_10min != None and
        metrics_queryset[0].price_change_1hr != None and
        metrics_queryset[0].price_change_24hr != None and
        metrics_queryset[0].daily_relative_volume != None and
        metrics_queryset[0].five_min_relative_volume != None and
        metrics_queryset[0].twenty_min_relative_volume != None):

        # 24 hour volume growth
        # current volume - volume 5 min ago / volume 5 min ago * 100
        current_volume = metrics_queryset[0].volume_24h
        previous_volume = metrics_queryset[1].volume_24h
        volume_growth = (current_volume - previous_volume) / previous_volume * 100

        # 5 min relative volume progression
        # current and previous 2 are increasing or equivalent
        rvol_progression = False
        current_rvol = metrics_queryset[0].five_min_relative_volume
        one_previous_rvol = metrics_queryset[1].five_min_relative_volume
        two_previous_rvol = metrics_queryset[2].five_min_relative_volume
        if (two_previous_rvol <= one_previous_rvol <= current_rvol):
            rvol_progression = True

        # 5 min price change is greater than previous
        five_min_price_increase = False
        current_five_min = metrics_queryset[0].price_change_5min
        previous_five_min = metrics_queryset[1].price_change_5min
        previous_five_min_two = metrics_queryset[2].price_change_5min
        if (previous_five_min < current_five_min and
            previous_five_min < 0 and
            current_five_min > 0):
            five_min_price_increase = True

        # 5 min and 10 min price changes go negative, positive, positive
        ten_min_price_increase = False
        current_ten_min = metrics_queryset[0].price_change_10min
        previous_ten_min = metrics_queryset[1].price_change_10min
        previous_ten_min_two = metrics_queryset[2].price_change_10min
        if (previous_ten_min < current_ten_min and
            previous_ten_min < 0):
            ten_min_price_increase = True

        if (
            metrics_queryset[0].daily_relative_volume >= 1.1 and
            metrics_queryset[0].rolling_relative_volume >= 1.4 and
            metrics_queryset[0].five_min_relative_volume >= 1.3 and
            metrics_queryset[0].price_change_5min >= 0 and
            metrics_queryset[1].price_change_5min < 0 and
            metrics_queryset[0].price_change_24hr < 0 and
            five_min_price_increase == True
        ):
            print("TRIGGER 1 passed")
            trigger_passed = True
            updated_trigger = str(metrics_queryset[0].coin) + " : New Trigger 1 Hit !"
            exists = check_duplicate_triggers(updated_trigger)

            if exists == False:

                true_triggers.append(updated_trigger)

                try:
                    Trigger.objects.create(trigger_name=updated_trigger, timestamp=now())

                except Exception as e:
                    print(f"Error creating new Trigger: {e}")


        if (
            metrics_queryset[0].daily_relative_volume >= 2 and
            metrics_queryset[0].rolling_relative_volume >= 1.2 and
            metrics_queryset[0].five_min_relative_volume >= 1.3 and
            metrics_queryset[0].price_change_5min >= 0.7 and
            metrics_queryset[0].price_change_24hr < -5 and
            metrics_queryset[0].twenty_min_relative_volume >= 1 and
            rvol_progression == True
        ):
            print("TRIGGER 2 passed")
            trigger_passed = True
            updated_trigger_two = str(metrics_queryset[0].coin) + " : New Trigger 2 Hit !"
            exists = check_duplicate_triggers(updated_trigger_two)

            if exists == False:

                true_triggers.append(updated_trigger_two)

                try:
                    Trigger.objects.create(trigger_name=updated_trigger_two, timestamp=now())

                except Exception as e:
                    print(f"Error creating new Trigger: {e}")


        if (
            #metrics_queryset[0].price_change_24hr < -5 and
            #metrics_queryset[0].daily_relative_volume > 1.15 and
            #metrics_queryset[0].rolling_relative_volume >= 1.9 and
            #metrics_queryset[0].price_change_5min < 0 and
            #metrics_queryset[0].price_change_10min < 0 and
            #metrics_queryset[0].price_change_1hr > 0 and
            #metrics_queryset[1].price_change_1hr < metrics_queryset[0].price_change_1hr and
            #metrics_queryset[2].price_change_1hr < metrics_queryset[1].price_change_1hr

            metrics_queryset[0].price_change_24hr < -5 and
            metrics_queryset[0].rolling_relative_volume >= 2.1 and
            metrics_queryset[1].price_change_5min < metrics_queryset[0].price_change_5min and
            metrics_queryset[2].price_change_5min < metrics_queryset[1].price_change_5min and
            metrics_queryset[0].price_change_10min > 0 and
            metrics_queryset[0].price_change_1hr > 0 and
            metrics_queryset[1].price_change_1hr < metrics_queryset[0].price_change_1hr and
            metrics_queryset[2].price_change_1hr < metrics_queryset[1].price_change_1hr
        ):
            print("TRIGGER 3 passed")
            trigger_passed = True
            updated_trigger_three = str(metrics_queryset[0].coin) + " : New Trigger 3 Hit !"
            exists = check_duplicate_triggers(updated_trigger_three)

            if exists == False:

                true_triggers.append(updated_trigger_three)

                try:
                    Trigger.objects.create(trigger_name=updated_trigger_three, timestamp=now())

                except Exception as e:
                    print(f"Error creating new Trigger: {e}")


        if (
            metrics_queryset[0].daily_relative_volume > 1.2 and
            metrics_queryset[0].rolling_relative_volume >= 1.85 and
            metrics_queryset[0].price_change_5min < metrics_queryset[1].price_change_5min and
            metrics_queryset[1].price_change_5min < metrics_queryset[2].price_change_5min and
            metrics_queryset[0].price_change_1hr > 1 and
            metrics_queryset[0].price_change_10min < metrics_queryset[1].price_change_10min and
            metrics_queryset[1].price_change_10min < metrics_queryset[2].price_change_10min and
            metrics_queryset[0].price_change_1hr < metrics_queryset[1].price_change_1hr and
            metrics_queryset[1].price_change_1hr < metrics_queryset[2].price_change_1hr
        ):
            print("TRIGGER 4 passed")
            trigger_passed = True
            updated_trigger_four = str(metrics_queryset[0].coin) + " : SHORT Trigger Hit !"
            exists = check_duplicate_triggers(updated_trigger_four)

            if exists == False:

                true_triggers.append(updated_trigger_four)

                try:
                    Trigger.objects.create(trigger_name=updated_trigger_four, timestamp=now())

                except Exception as e:
                    print(f"Error creating new Trigger: {e}")


        if (
            metrics_queryset[0].price_change_24hr < -5 and
            (metrics_queryset[0].rolling_relative_volume >= 2.1 or metrics_queryset[0].daily_relative_volume >= 1.3) and
            metrics_queryset[0].price_change_5min < 0 and
            metrics_queryset[0].price_change_10min < 0 and
            metrics_queryset[0].price_change_1hr > 0 and
            metrics_queryset[1].price_change_1hr < metrics_queryset[0].price_change_1hr and
            metrics_queryset[2].price_change_1hr < metrics_queryset[1].price_change_1hr
        ):
            print("TRIGGER 5 passed")
            trigger_passed = True
            updated_trigger_five = str(metrics_queryset[0].coin) + " : Trigger Five Hit !"
            exists = check_duplicate_triggers(updated_trigger_five)

            if exists == False:

                true_triggers.append(updated_trigger_five)

                try:
                    Trigger.objects.create(trigger_name=updated_trigger_five, timestamp=now())

                except Exception as e:
                    print(f"Error creating new Trigger: {e}")


    if trigger_passed == True:
        print("at least one trigger passed ===================================")
    else:
        print("no triggers passed")

    if len(true_triggers) > 0:
        send_text(true_triggers)


    return


def index(request):

    top_cryptos = []
    daily_relative_volumes = []
    sorted_volumes = []

    coins = Coin.objects.prefetch_related(
        Prefetch(
            'short_interval_data',  # The related_name defined in ShortIntervalData
            queryset=ShortIntervalData.objects.order_by('-timestamp'),
            to_attr='prefetched_short_interval_data'  # Use a unique name
        ),
        Prefetch(
            'metrics',  # The related_name defined in Metrics
            queryset=Metrics.objects.order_by('-timestamp'),
            to_attr='prefetched_metrics'  # Use a unique name
        )
    )

    for coin in coins:

        short_interval_data = coin.prefetched_short_interval_data[0] if coin.prefetched_short_interval_data else None
        metric = coin.prefetched_metrics[0] if coin.prefetched_metrics else None

        # Extract fields with default values
        coin_time = getattr(short_interval_data, 'timestamp', None)
        coin_price = round(getattr(short_interval_data, 'price', 0) or 0, 7) if short_interval_data else None
        coin_market_cap = getattr(metric, 'market_cap', None)
        coin_volume_24h_USD = round(getattr(metric, 'volume_24h', 0) or 0, 2) if metric else None
        coin_price_change_1h = round(getattr(metric, 'price_change_1hr', 0) or 0, 2) if metric else None
        coin_price_change_24h_percentage = round(getattr(metric, 'price_change_24hr', 0) or 0, 2) if metric else None
        coin_price_change_7d = round(getattr(metric, 'price_change_7d', 0) or 0, 2) if metric else None
        coin_circulating_supply = getattr(metric, 'circulating_supply', None)
        coin_rolling_relative_volume = round(getattr(metric, 'rolling_relative_volume', 0) or 0, 2) if metric else None
        coin_daily_relative_volume = round(getattr(metric, 'daily_relative_volume', 0) or 0, 2) if metric else None
        coin_twenty_min_relative_volume = round(getattr(metric, 'twenty_min_relative_volume', 0) or 0, 2) if metric else None
        coin_five_min_relative_volume = round(getattr(metric, 'five_min_relative_volume', 0) or 0, 2) if metric else None
        coin_price_change_5min = round(getattr(metric, 'price_change_5min', 0) or 0, 2) if metric else None
        coin_price_change_10min = round(getattr(metric, 'price_change_10min', 0) or 0, 2) if metric else None

        # Calculate relative volume progression
        metrics_queryset = coin.prefetched_metrics
        relative_volumes = metrics_queryset[:73][::6] if metrics_queryset else []

        volumes = [
            round(volume.rolling_relative_volume, 2)
            for volume in relative_volumes if volume.rolling_relative_volume is not None
        ]

        # Check volume progression
        is_descending = all(volumes[i] >= volumes[i + 1] for i in range(len(volumes) - 1))
        not_all_same = len(set(volumes)) > 1


        # TRIGGER INFORMATION HERE ---------------------------------
        check_triggers(metrics_queryset[:6])

        top_cryptos.append({
            "time": coin_time,
            "name": coin.name,
            "symbol": coin.symbol,
            "price": coin_price,
            "market_cap": coin_market_cap,
            "volume_24h_USD": coin_volume_24h_USD,
            "price_change_1h": coin_price_change_1h,
            "price_change_24h_percentage": coin_price_change_24h_percentage,
            "price_change_7d": coin_price_change_7d,
            "circulating_supply": coin_circulating_supply,
            "daily_relative_volume": coin_daily_relative_volume,
            "rolling_relative_volume": coin_rolling_relative_volume,
            "five_min_relative_volume": coin_five_min_relative_volume,
            "twenty_min_relative_volume": coin_twenty_min_relative_volume,
            "price_change_5min": coin_price_change_5min,
            "price_change_10min": coin_price_change_10min,
            "exchange": coin.exchange,
        })

        if is_descending and not_all_same:
            daily_relative_volumes.append({
                "rank": coin.market_cap_rank,
                "symbol": coin.symbol,
                "price_change_24h_percentage": coin_price_change_24h_percentage,
                "volumes": volumes,
                "is_descending": is_descending,
                "daily_relative_volume": coin_daily_relative_volume,
                "price": coin_price,
                "exchange": coin.exchange,
            })

    sorted_coins = sorted(top_cryptos, key=lambda x: x["daily_relative_volume"] or 0, reverse=True)
    sorted_volumes = sorted(daily_relative_volumes, key=lambda x: x["price_change_24h_percentage"] or 0, reverse=True)

    triggers = list(Trigger.objects.values("trigger_name", "timestamp"))


    # Handle AJAX request for partial updates
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':

        data = {
            "top_cryptos": sorted_coins,
            "sorted_volumes": sorted_volumes,
            "triggers": triggers,
        }

        return JsonResponse(data, safe=False)

    # Render data to the HTML template
    return render(request, "index.html", {
        "top_cryptos": sorted_coins,
        "sorted_volumes": sorted_volumes,
        "triggers": triggers,
    })











#
