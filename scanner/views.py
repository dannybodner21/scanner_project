import json
import time
import urllib.request
import os
import requests
import asyncio
import decimal
import statistics
import csv
import pandas as pd
import finnhub
import numpy as np
# import matplotlib.pyplot as plt  # Commented out - causing architecture compatibility issues
# import mplfinance as mpf  # Commented out - causing architecture compatibility issues
import traceback
import threading
import google.auth

from django.shortcuts import render
from zoneinfo import ZoneInfo
from django.http import HttpResponseRedirect, HttpResponse, Http404
from scanner.models import Coin, ConfidenceHistory, LiveChart, MemoryTrade, LivePriceSnapshot, CoinAPIPrice, RealTrade, ModelTrade, RickisMetrics, BacktestResult, SuccessfulMove, FiredSignal, SupportResistance, Pattern, HighLowData, HistoricalData, ShortIntervalData, Metrics, Trigger
from datetime import datetime, timedelta, timezone, date
from django.utils.timezone import now
from django.http import JsonResponse
from django.http import HttpResponse
from django.http import FileResponse, Http404
from django.db.models import Prefetch, OuterRef, Subquery
from django.db.models import Max, Min
from django.core.management import call_command
from django.views.decorators.csrf import csrf_exempt
from scanner.utils import score_metrics
from scanner.utils import send_telegram_alert
from scanner.utils import score_metrics, score_metrics_short
# from sklearn.linear_model import LinearRegression  # Commented out - causing pyarrow architecture issues
from collections import defaultdict
from django.utils.timezone import make_aware, is_naive
from django.db.models import Sum
from django.utils.timezone import timedelta
from django.core.management.base import BaseCommand
from threading import Thread
from scanner.management.commands.run_five_min_update_logic import run_five_min_update_logic
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from django.db.models import Q

from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import RealTrade
from .serializers import RealTradeLiveSerializer



from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import RealTrade, Coin, CoinAPIPrice








# binance price pull
# scanner/views_import.py
import json, math
from datetime import datetime, timezone
from django.conf import settings
from django.db import transaction
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponseForbidden
from django.views.decorators.csrf import csrf_exempt
from scanner.models import CoinAPIPrice

EPS = 1e-9  # change threshold

def _parse_ts(ts: str):
    # Expect ISO8601; accept 'Z'
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

@csrf_exempt
def import_candles(request):
    if request.method != "POST":
        return HttpResponseBadRequest("POST only.")

    if request.headers.get("X-Auth") != getattr(settings, "INTERNAL_IMPORT_TOKEN", None):
        return HttpResponseForbidden("Bad token.")

    try:
        payload = json.loads(request.body.decode("utf-8"))
        coin = payload["coin"]
        rows = payload["rows"]
        if not isinstance(rows, list):
            return HttpResponseBadRequest("rows must be a list")
    except Exception:
        return HttpResponseBadRequest("Invalid JSON")

    # dedupe by timestamp within this batch
    parsed = {}
    for r in rows:
        try:
            ts = _parse_ts(r["timestamp"])
            parsed[ts] = {
                "o": float(r["o"]),
                "h": float(r["h"]),
                "l": float(r["l"]),
                "c": float(r["c"]),
                "v": float(r["v"]),
            }
        except Exception:
            return HttpResponseBadRequest("Bad row format")

    ts_list = list(parsed.keys())
    existing = CoinAPIPrice.objects.filter(coin=coin, timestamp__in=ts_list)
    existing_map = {e.timestamp: e for e in existing}

    to_create, to_update = [], []
    for ts, vals in parsed.items():
        obj = existing_map.get(ts)
        if obj is None:
            to_create.append(CoinAPIPrice(
                coin=coin,
                timestamp=ts,
                open=vals["o"], high=vals["h"], low=vals["l"], close=vals["c"], volume=vals["v"]
            ))
        else:
            changed = (
                not math.isclose(float(obj.open),  vals["o"], abs_tol=EPS) or
                not math.isclose(float(obj.high),  vals["h"], abs_tol=EPS) or
                not math.isclose(float(obj.low),   vals["l"], abs_tol=EPS) or
                not math.isclose(float(obj.close), vals["c"], abs_tol=EPS) or
                not math.isclose(float(obj.volume),vals["v"], abs_tol=EPS)
            )
            if changed:
                obj.open = vals["o"]
                obj.high = vals["h"]
                obj.low  = vals["l"]
                obj.close= vals["c"]
                obj.volume = vals["v"]
                to_update.append(obj)

    with transaction.atomic():
        if to_create:
            CoinAPIPrice.objects.bulk_create(to_create, ignore_conflicts=True)
        if to_update:
            CoinAPIPrice.objects.bulk_update(to_update, ["open","high","low","close","volume"])

    return JsonResponse({
        "created": len(to_create),
        "updated": len(to_update),
        "received": len(rows),
        "processed": len(parsed),
    })











@api_view(['GET'])
def live_trades(request):
    trades = RealTrade.objects.filter(exit_price__isnull=True).order_by('-entry_timestamp')

    results = []
    for trade in trades:
        symbol = trade.coin.symbol.upper()
        current_price = get_current_price(symbol)
        if not current_price:
            continue

        entry = trade.entry_price
        if trade.trade_type.lower() == "long":
            pnl = ((current_price - entry) / entry) * 100
        else:
            pnl = ((entry - current_price) / entry) * 100

        # Round entry and current prices appropriately
        if symbol.lower() == "shib":
            entry = round(entry, 8)
            current_price = round(current_price, 8)
        else:
            entry = round(entry, 6)
            current_price = round(current_price, 6)

        results.append({
            "coin": symbol,
            "trade_type": trade.trade_type.upper(),
            "entry_price": entry,
            "current": current_price,
            "pnl": round(pnl, 2)
        })

    return Response(results)




def get_current_price(symbol):

    try:
        symbol = symbol.upper()

        # Get latest close price from LivePriceSnapshot
        snapshot = LivePriceSnapshot.objects.filter(coin=symbol).first()
        return snapshot.close if snapshot else None

    except:
        return None



def open_trades_view(request):
    open_trades = RealTrade.objects.filter(exit_timestamp__isnull=True).order_by('-entry_timestamp')

    trades_data = []
    for trade in open_trades:
        symbol = trade.coin.symbol.upper()
        current_price = get_current_price(symbol)
        if not current_price:
            continue

        entry = trade.entry_price
        if trade.trade_type.lower() == "long":
            pnl = ((current_price - entry) / entry) * 100
        else:
            pnl = ((entry - current_price) / entry) * 100

        if symbol.lower() == "shib":
            entry = round(entry, 8)
            current_price = round(current_price, 8)
        else:
            entry = round(entry, 4)
            current_price = round(current_price, 4)

        trades_data.append({
            'coin': symbol,
            'type': trade.trade_type,
            'entry_price': entry,
            'current_price': current_price,
            'pnl': round(pnl, 2),
        })

    return render(request, 'live_trades.html', {'trades': trades_data})









@csrf_exempt
def five_min_update(request):
    Thread(target=run_five_min_update_logic).start()
    return JsonResponse({"status": "started"})




# views for Webflow ------------------------------------------------------------

# function to get trade data on Webflow
from scanner.models import ModelTrade

COINAPI_KEY = "01293e2a-dcf1-4e81-8310-c6aa9d0cb743"
COINAPI_SYMBOL_MAP = {
    "BTC": "BINANCE_SPOT_BTC_USDT",
    "ETH": "BINANCE_SPOT_ETH_USDT",
    "XRP": "BINANCE_SPOT_XRP_USDT",
    "LTC": "BINANCE_SPOT_LTC_USDT",
    "SOL": "BINANCE_SPOT_SOL_USDT",
    "DOGE": "BINANCE_SPOT_DOGE_USDT",
    "LINK": "BINANCE_SPOT_LINK_USDT",
    "DOT": "BINANCE_SPOT_DOT_USDT",
    "SHIB": "BINANCE_SPOT_SHIB_USDT",
    "ADA": "BINANCE_SPOT_ADA_USDT",
    "UNI": "BINANCE_SPOT_UNI_USDT",
    "AVAX": "BINANCE_SPOT_AVAX_USDT",
    "XLM": "BINANCE_SPOT_XLM_USDT",
    "TRX": "BINANCE_SPOT_TRX_USDT",
    "ATOM": "BINANCE_SPOT_ATOM_USDT",
}


def get_coinapi_price(symbol):
    coinapi_symbol = COINAPI_SYMBOL_MAP.get(symbol)
    if not coinapi_symbol:
        return 0
    url = f"https://rest.coinapi.io/v1/ohlcv/{coinapi_symbol}/latest?period_id=5MIN&limit=1"
    headers = {"X-CoinAPI-Key": COINAPI_KEY}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        r.raise_for_status()
        data = r.json()
        return float(data[0]["price_close"]) if data else 0
    except:
        return 0


def get_open_trades(request):

    open_trades = (
        ModelTrade.objects
        .filter(exit_timestamp__isnull=True)
        .select_related("coin")
        .order_by("-entry_timestamp")[:100]
    )

    data = []

    for trade in open_trades:

        coin_symbol = trade.coin.symbol
        current_price = float(get_current_price(coin_symbol))
        entry_price = float(trade.entry_price or 0)
        current_percentage = 0

        if entry_price and current_price:

            current_percentage = ((current_price - entry_price) / entry_price) * 100

        data.append({
            "coin": coin_symbol,
            "trade_type": trade.trade_type,
            "model_confidence": trade.model_confidence,
            "entry_timestamp": trade.entry_timestamp.isoformat(),
            "entry_price": entry_price,
            "take_profit_percent": trade.take_profit_percent,
            "stop_loss_percent": trade.stop_loss_percent,
            "duration_minutes": trade.duration_minutes,
            "current_price": current_price,
            "current_percentage": current_percentage,
            "model_name": trade.model_name,
        })

    return JsonResponse(data, safe=False)


def get_closed_trades(request):
    closed_trades = (
        ModelTrade.objects
        .filter(exit_timestamp__isnull=False)
        .select_related("coin")
        .order_by("-exit_timestamp")[:1000]
    )

    data = []
    for trade in closed_trades:

        trade_result = "✅"
        if trade.result == False:
            trade_result = "❌"

        data.append({
            "coin": trade.coin.symbol,
            "trade_type": trade.trade_type,
            "model_confidence": trade.model_confidence,
            "entry_timestamp": trade.entry_timestamp.isoformat(),
            "entry_price": float(trade.entry_price or 0),
            "exit_timestamp": trade.exit_timestamp.isoformat(),
            "exit_price": float(trade.exit_price or 0),
            "duration_minutes": trade.duration_minutes,
            "result": trade_result,
        })

    return JsonResponse(data, safe=False)


def get_memory_trades(request):
    memory_trades = (
        MemoryTrade.objects
        .order_by("-timestamp")[:1000]
    )

    data = []
    for trade in memory_trades:

        trade_result = "open"
        if trade.outcome == "loss":
            trade_result = "❌"
        elif trade.outcome == "win":
            trade_result = "✅"

        data.append({
            "coin": trade.coin,
            "trade_type": trade.trade_type,
            "ml_confidence": trade.ml_confidence,
            "gpt_confidence": trade.gpt_confidence or -1,
            "outcome": trade_result,
        })

    return JsonResponse(data, safe=False)


# display charts on Webflow
def serve_chart_image(request, coin):
    try:
        chart = LiveChart.objects.get(coin=coin)
        if not chart.image:
            raise Http404("No image found.")
        return HttpResponse(chart.image.read(), content_type="image/png")
    except LiveChart.DoesNotExist:
        raise Http404("Chart not found.")


# Return True if all 400 recent candles exist for a coin
def has_400_recent_candles(coin_symbol: str) -> bool:
    # Grab the latest 400 timestamps and verify contiguous 5-min spacing
    qs = (
        CoinAPIPrice.objects
        .filter(coin=coin_symbol)
        .order_by('-timestamp')
        .values_list('timestamp', flat=True)[:400]
    )
    ts = list(qs)
    if len(ts) < 400:
        return False

    ts = sorted(ts)  # ascending
    step = timedelta(minutes=5)
    for i in range(1, 400):
        if ts[i] - ts[i-1] != step:
            return False
    return True


def get_model_results(request):

    total_long_trades = ModelTrade.objects.filter(trade_type="long", exit_timestamp__isnull=False).count()
    total_short_trades = ModelTrade.objects.filter(trade_type="short", exit_timestamp__isnull=False).count()

    total_long_wins = ModelTrade.objects.filter(trade_type="long", result=True).count()
    total_short_wins = ModelTrade.objects.filter(trade_type="short", result=True).count()

    model_name = "two_long_hgb_model.joblib"
    short_model_name = "short_four_model.joblib"

    btc_model = "btc_rf_model.joblib"
    avax_model = "avax_lr_model.joblib"
    doge_model = "doge_lr_model.joblib"
    sol_model = "sol_rf_model.joblib"
    ltc_model = "ltc_rf_model.joblib"
    link_model = "link_rf_model.joblib"
    xrp_model = "xrp_rf_model.joblib"
    uni_model = "uni_rf_model.joblib"

    btc_long_trades = ModelTrade.objects.filter(trade_type="long", model_name=btc_model, exit_timestamp__isnull=False).count()
    avax_long_trades = ModelTrade.objects.filter(trade_type="long", model_name=avax_model, exit_timestamp__isnull=False).count()
    doge_long_trades = ModelTrade.objects.filter(trade_type="long", model_name=doge_model, exit_timestamp__isnull=False).count()
    sol_long_trades = ModelTrade.objects.filter(trade_type="long", model_name=sol_model, exit_timestamp__isnull=False).count()
    ltc_long_trades = ModelTrade.objects.filter(trade_type="long", model_name=ltc_model, exit_timestamp__isnull=False).count()
    link_long_trades = ModelTrade.objects.filter(trade_type="long", model_name=link_model, exit_timestamp__isnull=False).count()
    xrp_long_trades = ModelTrade.objects.filter(trade_type="long", model_name=xrp_model, exit_timestamp__isnull=False).count()
    uni_long_trades = ModelTrade.objects.filter(trade_type="long", model_name=uni_model, exit_timestamp__isnull=False).count()

    btc_long_wins = ModelTrade.objects.filter(trade_type="long", model_name=btc_model, result=True).count()
    avax_long_wins = ModelTrade.objects.filter(trade_type="long", model_name=avax_model, result=True).count()
    doge_long_wins = ModelTrade.objects.filter(trade_type="long", model_name=doge_model, result=True).count()
    sol_long_wins = ModelTrade.objects.filter(trade_type="long", model_name=sol_model, result=True).count()
    ltc_long_wins = ModelTrade.objects.filter(trade_type="long", model_name=ltc_model, result=True).count()
    link_long_wins = ModelTrade.objects.filter(trade_type="long", model_name=link_model, result=True).count()
    xrp_long_wins = ModelTrade.objects.filter(trade_type="long", model_name=xrp_model, result=True).count()
    uni_long_wins = ModelTrade.objects.filter(trade_type="long", model_name=uni_model, result=True).count()
    

    btc_history = ConfidenceHistory.objects.filter(
        model_name=btc_model,
        coin__symbol="BTC"
    ).order_by("timestamp")
    xrp_history = ConfidenceHistory.objects.filter(
        model_name=xrp_model,
        coin__symbol="XRP"
    ).order_by("timestamp")
    ltc_history = ConfidenceHistory.objects.filter(
        model_name=ltc_model,
        coin__symbol="LTC"
    ).order_by("timestamp")
    sol_history = ConfidenceHistory.objects.filter(
        model_name=sol_model,
        coin__symbol="SOL"
    ).order_by("timestamp")
    doge_history = ConfidenceHistory.objects.filter(
        model_name=doge_model,
        coin__symbol="DOGE"
    ).order_by("timestamp")
    link_history = ConfidenceHistory.objects.filter(
        model_name=link_model,
        coin__symbol="LINK"
    ).order_by("timestamp")
    uni_history = ConfidenceHistory.objects.filter(
        model_name=uni_model,
        coin__symbol="UNI"
    ).order_by("timestamp")
    avax_history = ConfidenceHistory.objects.filter(
        model_name=avax_model,
        coin__symbol="AVAX"
    ).order_by("timestamp")


    # eth_history = ConfidenceHistory.objects.filter(
    #     model_name=model_name,
    #     coin__symbol="ETH"
    # ).order_by("timestamp")
    # dot_history = ConfidenceHistory.objects.filter(
    #     model_name=model_name,
    #     coin__symbol="DOT"
    # ).order_by("timestamp")
    # shib_history = ConfidenceHistory.objects.filter(
    #     model_name=model_name,
    #     coin__symbol="SHIB"
    # ).order_by("timestamp")
    # ada_history = ConfidenceHistory.objects.filter(
    #     model_name=model_name,
    #     coin__symbol="ADA"
    # ).order_by("timestamp")
    # xlm_history = ConfidenceHistory.objects.filter(
    #     model_name=model_name,
    #     coin__symbol="XLM"
    # ).order_by("timestamp")
    # trx_history = ConfidenceHistory.objects.filter(
    #     model_name=model_name,
    #     coin__symbol="TRX"
    # ).order_by("timestamp")
    # atom_history = ConfidenceHistory.objects.filter(
    #     model_name=model_name,
    #     coin__symbol="ATOM"
    # ).order_by("timestamp")

    btc_list = list(btc_history.values_list("confidence", flat=True))
    xrp_list = list(xrp_history.values_list("confidence", flat=True))
    ltc_list = list(ltc_history.values_list("confidence", flat=True))
    sol_list = list(sol_history.values_list("confidence", flat=True))
    doge_list = list(doge_history.values_list("confidence", flat=True))
    link_list = list(link_history.values_list("confidence", flat=True))
    uni_list = list(uni_history.values_list("confidence", flat=True))
    avax_list = list(avax_history.values_list("confidence", flat=True))

    # eth_list = list(eth_history.values_list("confidence", flat=True))
    # dot_list = list(dot_history.values_list("confidence", flat=True))
    # shib_list = list(shib_history.values_list("confidence", flat=True))
    # ada_list = list(ada_history.values_list("confidence", flat=True))
    # xlm_list = list(xlm_history.values_list("confidence", flat=True))
    # trx_list = list(trx_history.values_list("confidence", flat=True))
    # atom_list = list(atom_history.values_list("confidence", flat=True))

    short_btc_history = ConfidenceHistory.objects.filter(
        model_name=short_model_name,
        coin__symbol="BTC"
    ).order_by("timestamp")
    short_eth_history = ConfidenceHistory.objects.filter(
        model_name=short_model_name,
        coin__symbol="ETH"
    ).order_by("timestamp")
    short_xrp_history = ConfidenceHistory.objects.filter(
        model_name=short_model_name,
        coin__symbol="XRP"
    ).order_by("timestamp")
    short_ltc_history = ConfidenceHistory.objects.filter(
        model_name=short_model_name,
        coin__symbol="LTC"
    ).order_by("timestamp")
    short_sol_history = ConfidenceHistory.objects.filter(
        model_name=short_model_name,
        coin__symbol="SOL"
    ).order_by("timestamp")
    short_doge_history = ConfidenceHistory.objects.filter(
        model_name=short_model_name,
        coin__symbol="DOGE"
    ).order_by("timestamp")
    short_link_history = ConfidenceHistory.objects.filter(
        model_name=short_model_name,
        coin__symbol="LINK"
    ).order_by("timestamp")
    short_dot_history = ConfidenceHistory.objects.filter(
        model_name=short_model_name,
        coin__symbol="DOT"
    ).order_by("timestamp")
    short_shib_history = ConfidenceHistory.objects.filter(
        model_name=short_model_name,
        coin__symbol="SHIB"
    ).order_by("timestamp")
    short_ada_history = ConfidenceHistory.objects.filter(
        model_name=short_model_name,
        coin__symbol="ADA"
    ).order_by("timestamp")
    short_uni_history = ConfidenceHistory.objects.filter(
        model_name=short_model_name,
        coin__symbol="UNI"
    ).order_by("timestamp")
    short_avax_history = ConfidenceHistory.objects.filter(
        model_name=short_model_name,
        coin__symbol="AVAX"
    ).order_by("timestamp")
    short_xlm_history = ConfidenceHistory.objects.filter(
        model_name=short_model_name,
        coin__symbol="XLM"
    ).order_by("timestamp")

    short_btc_list = list(short_btc_history.values_list("confidence", flat=True))
    short_eth_list = list(short_eth_history.values_list("confidence", flat=True))
    short_xrp_list = list(short_xrp_history.values_list("confidence", flat=True))
    short_ltc_list = list(short_ltc_history.values_list("confidence", flat=True))
    short_sol_list = list(short_sol_history.values_list("confidence", flat=True))
    short_doge_list = list(short_doge_history.values_list("confidence", flat=True))
    short_link_list = list(short_link_history.values_list("confidence", flat=True))
    short_dot_list = list(short_dot_history.values_list("confidence", flat=True))
    short_shib_list = list(short_shib_history.values_list("confidence", flat=True))
    short_ada_list = list(short_ada_history.values_list("confidence", flat=True))
    short_uni_list = list(short_uni_history.values_list("confidence", flat=True))
    short_avax_list = list(short_avax_history.values_list("confidence", flat=True))
    short_xlm_list = list(short_xlm_history.values_list("confidence", flat=True))

    # Get the latest 2016 candles
    btc_full_window = has_400_recent_candles("BTCUSDT")
    eth_full_window = has_400_recent_candles("ETHUSDT")
    xrp_full_window = has_400_recent_candles("XRPUSDT")
    ltc_full_window = has_400_recent_candles("LTCUSDT")
    sol_full_window = has_400_recent_candles("SOLUSDT")
    doge_full_window = has_400_recent_candles("DOGEUSDT")
    link_full_window = has_400_recent_candles("LINKUSDT")
    dot_full_window = has_400_recent_candles("DOTUSDT")
    shib_full_window = has_400_recent_candles("SHIBUSDT")
    ada_full_window = has_400_recent_candles("ADAUSDT")
    uni_full_window = has_400_recent_candles("UNIUSDT")
    avax_full_window = has_400_recent_candles("AVAXUSDT")
    xlm_full_window = has_400_recent_candles("XLMUSDT")
    trx_full_window = has_400_recent_candles("TRXUSDT")
    atom_full_window = has_400_recent_candles("ATOMUSDT")

    # Get recent coin prices
    btc_price = CoinAPIPrice.objects.filter(coin="BTCUSDT").order_by("-timestamp").first().close or 0
    eth_price = CoinAPIPrice.objects.filter(coin="ETHUSDT").order_by("-timestamp").first().close or 0
    xrp_price = CoinAPIPrice.objects.filter(coin="XRPUSDT").order_by("-timestamp").first().close or 0
    dot_price = CoinAPIPrice.objects.filter(coin="DOTUSDT").order_by("-timestamp").first().close or 0
    sol_price = CoinAPIPrice.objects.filter(coin="SOLUSDT").order_by("-timestamp").first().close or 0
    trx_price = CoinAPIPrice.objects.filter(coin="TRXUSDT").order_by("-timestamp").first().close or 0
    shib_price = CoinAPIPrice.objects.filter(coin="SHIBUSDT").order_by("-timestamp").first().close or 0
    link_price = CoinAPIPrice.objects.filter(coin="LINKUSDT").order_by("-timestamp").first().close or 0

    # don't have saved prices here
    grt_price = 0.10
    xtz_price = 0.87
    matic_price = 0.25
    stx_price = 0.79


    return JsonResponse({

        "btc_price": btc_price,
        "eth_price": eth_price,
        "xrp_price": xrp_price,
        "dot_price": dot_price,
        "sol_price": sol_price,
        "trx_price": trx_price,
        "shib_price": shib_price,
        "link_price": link_price,
        "grt_price": grt_price,
        "xtz_price": xtz_price,
        "matic_price": matic_price,
        "stx_price": stx_price,

        "total_long_trades": total_long_trades,
        "total_short_trades": total_short_trades,
        "total_long_wins": total_long_wins,
        "total_short_wins": total_short_wins,

        "btc_long_trades": btc_long_trades,
        "btc_long_wins": btc_long_wins,
        "uni_long_trades": uni_long_trades,
        "uni_long_wins": uni_long_wins,
        "avax_long_trades": avax_long_trades,
        "avax_long_wins": avax_long_wins,
        "xrp_long_trades": xrp_long_trades,
        "xrp_long_wins": xrp_long_wins,
        "ltc_long_trades": ltc_long_trades,
        "ltc_long_wins": ltc_long_wins,
        "sol_long_trades": sol_long_trades,
        "sol_long_wins": sol_long_wins,
        "doge_long_trades": doge_long_trades,
        "doge_long_wins": doge_long_wins,
        "link_long_trades": link_long_trades,
        "link_long_wins": link_long_wins,

        "btc_list": btc_list,
        "uni_list": uni_list,
        "avax_list": avax_list,
        "xrp_list": xrp_list,
        "ltc_list": ltc_list,
        "sol_list": sol_list,
        "doge_list": doge_list,
        "link_list": link_list,

        # "eth_list": eth_list,
        # "dot_list": dot_list,
        # "shib_list": shib_list,
        # "ada_list": ada_list,
        # "xlm_list": xlm_list,
        # "trx_list": trx_list,
        # "atom_list": atom_list,

        "short_btc_list": short_btc_list,
        "short_eth_list": short_eth_list,
        "short_xrp_list": short_xrp_list,
        "short_ltc_list": short_ltc_list,
        "short_sol_list": short_sol_list,
        "short_doge_list": short_doge_list,
        "short_link_list": short_link_list,
        "short_dot_list": short_dot_list,
        "short_shib_list": short_shib_list,
        "short_ada_list": short_ada_list,
        "short_uni_list": short_uni_list,
        "short_avax_list": short_avax_list,
        "short_xlm_list": short_xlm_list,

        "btc_full_window": btc_full_window,
        "eth_full_window": eth_full_window,
        "xrp_full_window": xrp_full_window,
        "ltc_full_window": ltc_full_window,
        "sol_full_window": sol_full_window,
        "doge_full_window": doge_full_window,
        "link_full_window": link_full_window,
        "dot_full_window": dot_full_window,
        "shib_full_window": shib_full_window,
        "ada_full_window": ada_full_window,
        "uni_full_window": uni_full_window,
        "avax_full_window": avax_full_window,
        "xlm_full_window": xlm_full_window,
        "trx_full_window": trx_full_window,
        "atom_full_window": atom_full_window,
    })







# ------------------------------------------------------------------------------




def get_patterns(request):
    coins = [
        "ADA", "AVAX", "BTC", "DOGE", "DOT", "ETH", "HBAR", "LINK",
        "LTC", "PEPE", "SHIB", "SOL", "SUI", "UNI", "XLM", "XRP"
    ]

    resolutions = [5, 15, 60]

    data = []

    for coin_symbol in sorted(coins):  # alphabetical order
        for res in resolutions:
            pattern = (
                Pattern.objects
                .filter(symbol=f"BINANCE:{coin_symbol}USDT", resolution=res)
                .exclude(status__iexact='No Pattern')
                .order_by('-timestamp')
                .first()
            )

            if pattern:
                ts = pattern.timestamp.isoformat()
                dt = datetime.fromisoformat(ts)
                formatted = dt.strftime("%Y-%m-%d %H:%M:%S")
                entry = {
                    "symbol": pattern.symbol,
                    "resolution": pattern.resolution,
                    "pattern_type": pattern.patterntype,
                    "pattern_name": pattern.patternname,
                    "status": pattern.status,
                    "entry": float(pattern.entry) if pattern.entry else None,
                    "takeprofit": float(pattern.takeprofit) if pattern.takeprofit else None,
                    "stoploss": float(pattern.stoploss) if pattern.stoploss else None,
                    "adx": float(pattern.adx) if pattern.adx else None,
                    "timestamp": formatted,
                }
            else:
                entry = {
                    "symbol": f"BINANCE:{coin_symbol}USDT",
                    "resolution": res,
                    "pattern_type": None,
                    "pattern_name": None,
                    "status": None,
                    "entry": None,
                    "takeprofit": None,
                    "stoploss": None,
                    "adx": None,
                    "timestamp": None,
                }

            data.append(entry)

    return JsonResponse(data, safe=False)



def run_update_patterns(request):
    try:
        call_command('update_patterns')
        return JsonResponse({"status": "success"})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)





def get_btc_market_regime(request):
    # Load recent BTC data (last 200 + 20 candles to compute MA and volatility)
    end_date = datetime.now(timezone.utc)
    start_date = end_date - pd.Timedelta(minutes=5 * 250)  # 250 candles * 5 min = ~20 hours

    queryset = CoinAPIPrice.objects.filter(
        coin='BTCUSDT',
        timestamp__gte=start_date,
        timestamp__lte=end_date
    ).order_by('timestamp')

    df = pd.DataFrame(list(queryset.values('timestamp','close')))
    if df.empty:
        return JsonResponse({'error': 'No data available'}, status=404)

    for col in ['close']:
        df[col] = df[col].astype(float)
    df = df.set_index('timestamp').sort_index()

    # Calculate volatility and 200 MA
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    df['ma_200'] = df['close'].rolling(200).mean()

    latest = df.iloc[-1]

    median_vol = df['volatility'].median()

    if pd.isna(latest['ma_200']) or pd.isna(latest['volatility']):
        return JsonResponse({'error': 'Insufficient data to calculate regime'}, status=400)

    if latest['close'] > latest['ma_200'] and latest['volatility'] < median_vol:
        regime = 'bull'
    elif latest['close'] < latest['ma_200'] and latest['volatility'] > median_vol:
        regime = 'bear'
    else:
        regime = 'sideways'

    response = {
        'regime': regime,
        'timestamp': df.index[-1].isoformat(),
        'close': latest['close'],
        'volatility': latest['volatility'],
        'ma_200': latest['ma_200']
    }

    return JsonResponse(response)




# new model functions ----------------------------------------------------------


import math

def safe_float(val):
    if val is None:
        return None
    try:
        val = float(val)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except (TypeError, ValueError):
        return None


def is_valid_payload(payload):
    return all(value is not None for value in payload.values())






def run_live_pipeline_view(request):
    def run():
        try:
            # Conditional import to avoid architecture issues
            from .live_pipeline import run_live_pipeline
            run_live_pipeline()
        except Exception as e:
            print(f"❌ Live pipeline error: {e}")

    Thread(target=run).start()
    return JsonResponse({"status": "pipeline started"}, status=200)




























# display RickisMetrics
def daily_metrics_health(request):

    date_str = request.GET.get("date")
    if not date_str:
        return JsonResponse({"error": "Missing ?date=YYYY-MM-DD"}, status=400)

    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        start = target_date
        end = target_date + timedelta(days=1)

    except:
        return JsonResponse({"error": "Invalid date format"}, status=400)

    fields_to_check = [
        "rsi", "macd", "macd_signal", "stochastic_k", "stochastic_d",
        "support_level", "resistance_level", "price_slope_1h", "relative_volume",
        "sma_5", "sma_20", "ema_12", "ema_26", "stddev_1h", "atr_1h",
        "change_since_high", "change_since_low", "volume_mc_ratio",
        "obv", "adx", "bollinger_upper", "bollinger_middle", "bollinger_lower",
        "fib_distance_0_236", "fib_distance_0_382", "fib_distance_0_5",
        "fib_distance_0_618", "fib_distance_0_786"
    ]

    coins = [
        "BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LINK",
        "AVAX", "XLM", "TON", "SHIB", "SUI", "HBAR", "BCH", "DOT", "LTC",
        "XMR", "UNI", "PEPE", "APT", "NEAR", "ONDO", "TAO", "ICP", "ETC",
        "RENDER", "MNT", "KAS", "CRO", "AAVE", "POL", "VET", "FIL", "ALGO",
        "ENA", "ATOM", "TIA", "ARB", "DEXE", "OP", "JUP", "MKR", "STX",
        "EOS", "WLD", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT",
        "PAXG", "CRV", "JASMY", "SAND", "GALA", "CORE", "KAIA", "LDO",
        "THETA", "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI"
    ]

    response = {}

    for coin_symbol in coins:

        try:
            coin = Coin.objects.get(symbol=coin_symbol)

        except Coin.DoesNotExist:
            continue

        entries = RickisMetrics.objects.filter(
            coin=coin,
            timestamp__gte=start,
            timestamp__lt=end
        )

        total = entries.count()
        if total == 0:
            continue

        missing = {}
        for field in fields_to_check:

            missing_count = entries.filter(Q(**{f"{field}__isnull": True})).count()
            if missing_count > 0:
                missing[field] = missing_count

        response[coin.symbol] = {
            "total": total,
            "missing": missing
        }

    return JsonResponse(response)















# round all timestamps to store rounded numbers
def round_to_five_minutes(dt):
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)



# functions for ML Model OLD -------------------------------------------------------
def calculate_volatility_5min(coin, timestamp):
    try:
        recent = (
            ShortIntervalData.objects.filter(coin=coin, timestamp__lte=timestamp)
            .order_by("-timestamp")[:6]
        )

        prices = [float(d.price) for d in recent if d.price is not None]

        if len(prices) < 3:
            print(f"⚠️ Not enough price data for volatility — {coin.symbol}")
            return None

        import numpy as np
        volatility = np.std(prices)

        return float(volatility)

    except Exception as e:
        print(f"❌ Error calculating volatility for {coin.symbol}: {e}")
        return None


def calculate_trend_slope_30min(coin, timestamp):
    try:
        if is_naive(timestamp):
            timestamp = make_aware(timestamp)

        recent = list(
            ShortIntervalData.objects
            .filter(coin=coin, timestamp__lte=timestamp)
            .order_by("-timestamp")
            .values_list("timestamp", "price")[:6]
        )

        recent = list(reversed(recent))

        prices = [float(p) for (_, p) in recent if p is not None]

        if len(prices) < 3:
            print(f"⚠️ Not enough price points for {coin.symbol} to calculate slope")
            return None

        X = np.arange(len(prices)).reshape(-1, 1)
        y = np.array(prices)
        
        # Conditional import to avoid architecture issues
        try:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression().fit(X, y)
            return float(model.coef_[0])
        except ImportError:
            # Fallback to numpy polyfit
            slope = np.polyfit(X.flatten(), y, 1)[0]
            return float(slope)

    except Exception as e:
        print(f"❌ Error calculating slope for {coin.symbol}: {e}")
        return None


def calculate_change_since_high_low(coin, timestamp):
    try:
        recent = (
            ShortIntervalData.objects.filter(coin=coin, timestamp__lte=timestamp)
            .order_by("-timestamp")[:6]  # Last 30 minutes = 6 intervals
        )

        prices = [float(d.price) for d in recent if d.price is not None]

        if len(prices) < 2:
            print(f"⚠️ Not enough data for high/low change — {coin.symbol}")
            return None, None

        current_price = prices[0]  # most recent price
        highest = max(prices)
        lowest = min(prices)

        change_since_high = ((current_price - highest) / highest) * 100 if highest != 0 else None
        change_since_low = ((current_price - lowest) / lowest) * 100 if lowest != 0 else None

        return round(change_since_high, 3), round(change_since_low, 3)

    except Exception as e:
        print(f"❌ Error calculating change since high/low for {coin.symbol}: {e}")
        return None, None











def run_live_predictions_view(request):
    call_command("run_live_predictions")
    return JsonResponse({"status": "success"})


@csrf_exempt
def run_trade_check(request):
    if request.method == "POST":
        def run_check():
            call_command("update_open_trades")

        threading.Thread(target=run_check).start()
        return JsonResponse({"status": "Check started"})

    return JsonResponse({"error": "POST only"}, status=405)



@csrf_exempt
def run_metrics_and_scan(request):
    print(f"🧪 Incoming request method: {request.method}")

    if request.method in ["GET", "POST"]:
        from django.utils.timezone import now, timedelta
        from scanner.models import Metrics
        import requests
        import json
        from threading import Thread

        print("📊 Entered method block")

        cutoff = now() - timedelta(minutes=10)
        metrics = Metrics.objects.filter(timestamp__gte=cutoff)[:100]
        print(f"📈 Found {metrics.count()} metrics")

        payload = []
        for m in metrics:
            payload.append({
                "symbol": m.coin.symbol,
                "price_change_5min": m.price_change_5min,
                "price_change_10min": m.price_change_10min,
                "price_change_1hr": m.price_change_1hr,
                "price_change_24hr": m.price_change_24hr,
                "price_change_7d": m.price_change_7d,
                "five_min_relative_volume": m.five_min_relative_volume,
                "rolling_relative_volume": m.rolling_relative_volume,
                "volume_24h": float(m.volume_24h) if m.volume_24h else 0,
                "price": float(m.last_price) if m.last_price else None
            })

        print(f"📤 Sending {len(payload)} to bot")

        def async_post(payload_copy):
            try:
                res = requests.post(
                    "https://scanner-project-bkdz5.ondigitalocean.app/post-metrics-to-bot/",
                    json=payload_copy,
                    headers={"Content-Type": "application/json"},
                    timeout=20
                )
                print(f"✅ Bot responded: {res.status_code} — {res.text}")
            except Exception as e:
                print(f"❌ Async post failed: {e}")

        Thread(target=async_post, args=(payload,)).start()

        return JsonResponse({"status": "scan triggered", "sent": len(payload)})

    return JsonResponse({"error": "Invalid method"}, status=405)



















# five min update OLD --------------------------------------------------------------


def run_five_min_update_logic_V1():

    start = datetime.now()
    print(f"⏱️ Start: {start}")

    API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": API_KEY,
    }

    coins = Coin.objects.all()
    cmc_ids = [coin.cmc_id for coin in coins]

    totalCount = 0
    actualCount = 0
    shortDatas = []

    # RickisMetric Coins
    rickisCoins = [
        "BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LINK",
        "AVAX", "XLM", "TON", "SHIB", "SUI", "HBAR", "BCH", "DOT", "LTC",
        "XMR", "UNI", "PEPE", "APT", "NEAR", "ONDO", "TAO", "ICP", "ETC",
        "RENDER", "MNT", "KAS", "CRO", "AAVE", "POL", "VET", "FIL", "ALGO",
        "ENA", "ATOM", "TIA", "ARB", "DEXE", "OP", "JUP", "MKR", "STX",
        "EOS", "WLD", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT",
        "PAXG", "CRV", "JASMY", "SAND", "GALA", "CORE", "KAIA", "LDO",
        "THETA", "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI"
    ]

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

                    timestamp = datetime.strptime(
                        crypto_data["last_updated"], "%Y-%m-%dT%H:%M:%S.%fZ"
                    )

                    if is_naive(timestamp):
                        timestamp = make_aware(timestamp)

                    timestamp = round_to_five_minutes(timestamp)

                    coin = Coin.objects.get(cmc_id=cmc_id)
                    current_price = crypto_data["quote"]["USD"]["price"]

                    #volatility = calculate_volatility_5min(coin, timestamp)
                    #trend_slope = calculate_trend_slope_30min(coin, timestamp)
                    #change_low, change_high = calculate_change_since_high_low(coin, timestamp)
                    #volume_marketcap_ratio = float(crypto_data["quote"]["USD"]["volume_24h"]) / float(crypto_data["quote"]["USD"]["market_cap"]) if crypto_data["quote"]["USD"]["market_cap"] else None

                    try:
                        coin.market_cap_rank = crypto_data["cmc_rank"]
                        coin.last_updated = timestamp
                        coin.save()

                    except Exception as e:
                        print("FAILED IN GROUP 1")
                        print(e)

                    if current_price is None:
                        print(f"⚠️ Skipping — missing price value")
                        continue  # or return, depending on context


                    try:
                        shortData = ShortIntervalData.objects.create(
                            coin=coin,
                            timestamp=timestamp,
                            price=current_price,
                            volume_5min=crypto_data["quote"]["USD"]["volume_24h"],
                            circulating_supply=crypto_data["circulating_supply"]
                        )

                        shortDatas.append(shortData)

                    except Exception as e:
                        print("FAILED IN GROUP 2")
                        print(e)

                    try:

                        totalCount += 1

                        metric = Metrics.objects.create(
                            coin=coin,
                            timestamp=timestamp,
                            rolling_relative_volume=calculate_relative_volume(coin, timestamp),
                            five_min_relative_volume=calculate_five_min_relative_volume(coin, timestamp),
                            price_change_5min=calculate_price_change_five_min(coin, timestamp),
                            price_change_10min=calculate_price_change_thirty_min(coin, timestamp),
                            price_change_1hr = crypto_data["quote"]["USD"]["percent_change_1h"],
                            price_change_24hr = crypto_data["quote"]["USD"]["percent_change_24h"],
                            price_change_7d = crypto_data["quote"]["USD"]["percent_change_7d"],
                            circulating_supply=crypto_data["circulating_supply"],
                            volume_24h = crypto_data["quote"]["USD"]["volume_24h"],
                            last_price = current_price,
                            market_cap = crypto_data["quote"]["USD"]["market_cap"],
                            #volatility_5min = volatility,
                            #trend_slope_30min = trend_slope,
                            #change_since_low = change_low,
                            #change_since_high = change_high,
                            #volume_marketcap_ratio = volume_marketcap_ratio,
                        )

                        print("Metric created.")
                        actualCount += 1
                        print(f"total count = {totalCount}")
                        print(f"actual count = {actualCount}")

                    except Exception as e:
                        print("FAILED IN GROUP 3")
                        print(e)


                    now = datetime.now()

                    if (coin.symbol in rickisCoins):

                        try:

                            macd, signal = calculate_macd(coin, timestamp)
                            stochastic_k, stochastic_d = calculate_stochastic(coin, timestamp)
                            support, resistance = calculate_support_resistance(coin, timestamp)

                            if macd is None:
                                macd = Decimal("0")
                                print(f"MACD is None for {coin.symbol}")
                            if signal is None:
                                signal = Decimal("0")
                                print(f"signal is None for {coin.symbol}")
                            if stochastic_k is None:
                                stochastic_k = Decimal("0")
                                print(f"stochastic k is None for {coin.symbol}")
                            if stochastic_d is None:
                                stochastic_d = Decimal("0")
                                print(f"stochastic d is None for {coin.symbol}")
                            if support is None:
                                support = Decimal("0")
                                print(f"support is None for {coin.symbol}")
                            if resistance is None:
                                resistance = Decimal("0")
                                print(f"resistance is None for {coin.symbol}")

                            # create RickisMetrics
                            print(f"creating RickisMetric for {coin.symbol}")
                            RickisMetrics.objects.create(
                                coin=coin,
                                timestamp=timestamp,
                                price=current_price,
                                high_24h=0.0,
                                change_5m=calculate_price_change_five_min(coin, timestamp),
                                change_1h=crypto_data["quote"]["USD"].get("percent_change_1h"),
                                change_24h=crypto_data["quote"]["USD"].get("percent_change_24h"),
                                volume=crypto_data["quote"]["USD"].get("volume_24h"),
                                avg_volume_1h=calculate_avg_volume_1h(coin, timestamp) or 0,
                                rsi=calculate_rsi(coin, timestamp) or 0,
                                macd=macd,
                                macd_signal=signal,
                                stochastic_k=stochastic_k,
                                stochastic_d=stochastic_d,
                                support_level=support,
                                resistance_level=resistance,
                            )
                            print(f"succesfully created RickisMetric for {coin.symbol}")

                        except Exception as e:
                            print("FAILED IN RickisMetrics:")
                            print(e)


        except Exception as e:
            print(f"Error updating tracked coins for batch {cmc_id_batch}: {e}")

    print("five minute update complete.")
    print(f"✅ Done in: {datetime.now() - start}")

    return











# ------------------------------------------------------------------------------





# get additional data for RickisMetrics
def run_ohlcv_update():

    print("📊 Starting OHLCV update thread -------------------------------------------------------------------")

    API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
    url = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/latest'
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": API_KEY,
    }

    rickisCoins = [
        "BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LINK",
        "AVAX", "XLM", "TON", "SHIB", "SUI", "HBAR", "BCH", "DOT", "LTC",
        "XMR", "UNI", "PEPE", "APT", "NEAR", "ONDO", "TAO", "ICP", "ETC",
        "RENDER", "MNT", "KAS", "CRO", "AAVE", "POL", "VET", "FIL", "ALGO",
        "ENA", "ATOM", "TIA", "ARB", "DEXE", "OP", "JUP", "MKR", "STX",
        "EOS", "WLD", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT",
        "PAXG", "CRV", "JASMY", "SAND", "GALA", "CORE", "KAIA", "LDO",
        "THETA", "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI"
    ]

    coins = Coin.objects.filter(symbol__in=rickisCoins)
    cmc_ids = [coin.cmc_id for coin in coins]

    batch_size = 100
    for i in range(0, len(cmc_ids), batch_size):
        batch = cmc_ids[i:i + batch_size]
        print("entered loop")

        try:
            response = requests.get(url, headers=headers, params={
                "id": ",".join(map(str, batch)),
                "convert": "USD"
            })
            response.raise_for_status()
            data = response.json().get("data", {})

            for cmc_id in batch:
                coin = Coin.objects.get(cmc_id=cmc_id)

                if str(cmc_id) not in data:
                    print(f"❌ {coin.symbol} missing in OHLCV response.")
                    continue

                quote = data[str(cmc_id)].get("quote", {}).get("USD", {})
                high_raw = quote.get("high")

                if high_raw is None:
                    print(f"⚠️ No high_24h found for {coin.symbol}, skipping update.")
                    continue

                try:
                    high_24h = Decimal(str(high_raw))
                except Exception as parse_err:
                    print(f"⚠️ Could not parse high_24h for {coin.symbol}: {high_raw}")
                    continue

                print(f"🔎 {coin.symbol} → High 24h from OHLCV: {high_24h}")

                if high_24h > 0:
                    latest = RickisMetrics.objects.filter(coin=coin).order_by("-timestamp").first()

                    if latest:
                        print(f"📌 Updating RickisMetrics row ID {latest.id} for {coin.symbol}")
                        latest.high_24h = high_24h
                        latest.save()
                        print(f"✅ Saved high_24h = {high_24h} for {coin.symbol}")
                else:
                    print(f"⚠️ Skipping update for {coin.symbol} — high_24h was 0")

        except Exception as e:
            print("❌ OHLCV error:", e)

    print("✅ OHLCV update thread complete.")


def five_min_update_old(request=None):

    start = datetime.now()
    print(f"⏱️ Start: {start}")

    API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": API_KEY,
    }

    coins = Coin.objects.all()
    cmc_ids = [coin.cmc_id for coin in coins]

    totalCount = 0
    actualCount = 0
    shortDatas = []

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

                    timestamp = datetime.strptime(
                        crypto_data["last_updated"], "%Y-%m-%dT%H:%M:%S.%fZ"
                    )

                    if is_naive(timestamp):
                        timestamp = make_aware(timestamp)

                    coin = Coin.objects.get(cmc_id=cmc_id)
                    current_price = crypto_data["quote"]["USD"]["price"]

                    #volatility = calculate_volatility_5min(coin, timestamp)
                    #trend_slope = calculate_trend_slope_30min(coin, timestamp)
                    #change_low, change_high = calculate_change_since_high_low(coin, timestamp)
                    #volume_marketcap_ratio = float(crypto_data["quote"]["USD"]["volume_24h"]) / float(crypto_data["quote"]["USD"]["market_cap"]) if crypto_data["quote"]["USD"]["market_cap"] else None



                    try:
                        coin.market_cap_rank = crypto_data["cmc_rank"]
                        coin.last_updated = timestamp
                        coin.save()
                    except Exception as e:
                        print("FAILED IN GROUP 1")
                        print(e)

                    if current_price is None:
                        print(f"⚠️ Skipping — missing price value")
                        continue  # or return, depending on context


                    try:
                        shortData = ShortIntervalData.objects.create(
                            coin=coin,
                            timestamp=timestamp,
                            price=crypto_data["quote"]["USD"]["price"],
                            volume_5min=crypto_data["quote"]["USD"]["volume_24h"],
                            circulating_supply=crypto_data["circulating_supply"]
                        )

                        shortDatas.append(shortData)

                    except Exception as e:
                        print("FAILED IN GROUP 2")
                        print(e)

                    try:

                        totalCount += 1

                        metric = Metrics.objects.create(
                            coin=coin,
                            timestamp=timestamp,
                            rolling_relative_volume=calculate_relative_volume(coin, timestamp),
                            five_min_relative_volume=calculate_five_min_relative_volume(coin, timestamp),
                            price_change_5min=calculate_price_change_five_min(coin, timestamp),
                            price_change_10min=calculate_price_change_thirty_min(coin, timestamp),
                            price_change_1hr = crypto_data["quote"]["USD"]["percent_change_1h"],
                            price_change_24hr = crypto_data["quote"]["USD"]["percent_change_24h"],
                            price_change_7d = crypto_data["quote"]["USD"]["percent_change_7d"],
                            circulating_supply=crypto_data["circulating_supply"],
                            volume_24h = crypto_data["quote"]["USD"]["volume_24h"],
                            last_price = crypto_data["quote"]["USD"]["price"],
                            market_cap = crypto_data["quote"]["USD"]["market_cap"],
                            #volatility_5min = volatility,
                            #trend_slope_30min = trend_slope,
                            #change_since_low = change_low,
                            #change_since_high = change_high,
                            #volume_marketcap_ratio = volume_marketcap_ratio,
                        )

                        print("Metric created for.")

                        actualCount +=1

                        print(f"total count = {totalCount}")
                        print(f"actual count = {actualCount}")

                    except Exception as e:
                        print("FAILED IN GROUP 3")
                        print(e)

                    now = datetime.now()


        except Exception as e:
            print(f"Error updating tracked coins for batch {cmc_id_batch}: {e}")

    print("five minute update complete.")
    print(f"✅ Done in: {datetime.now() - start}")

    if request:
        return JsonResponse({"status": "success", "message": "Update triggered successfully"})





# functions for RickisMetrics V1 --------------------------------------------------

# high of day momentum scaner
#    price
#    high_24h
#    volume
#    avg_volume_1h
#    change_5m
#    change_1h
#    rsi
#    macd

def calculate_avg_volume_1h(coin, timestamp):

    start_time = timestamp - timedelta(minutes=60)

    past_entries = RickisMetrics.objects.filter(
        coin=coin,
        timestamp__lte=timestamp,
        timestamp__gte=start_time
    ).order_by('-timestamp')[:12]

    volumes = [entry.volume for entry in past_entries if entry.volume is not None]

    if not volumes:
        return None

    print(f"successfully created avg volume 1h for {coin.symbol}")

    return sum(volumes, Decimal("0")) / Decimal(len(volumes))


def calculate_rsi(coin, timestamp, periods=14):

    # Get at least (periods + 1) rows to compute changes
    qs = (
        RickisMetrics.objects
        .filter(coin=coin, timestamp__lte=timestamp)
        .order_by("-timestamp")[:periods + 1]
    )

    if qs.count() < periods + 1:
        return None  # Not enough data

    # Convert queryset to DataFrame (in correct order)
    data = list(qs)[::-1]  # oldest to newest
    prices = [row.price for row in data]
    df = pd.Series(prices)

    # Calculate price differences
    delta = df.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=periods).mean().iloc[-1]
    avg_loss = loss.rolling(window=periods).mean().iloc[-1]

    if avg_loss == 0:
        return 100.0  # RSI maxed

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return round(float(rsi), 2)


def calculate_macd(coin, timestamp):

    # Get enough price history (at least 35 for smoothing)
    qs = (
        RickisMetrics.objects
        .filter(coin=coin, timestamp__lte=timestamp)
        .order_by("-timestamp")[:50]
    )

    if qs.count() < 35:
        return None, None  # Not enough data

    data = list(qs)[::-1]  # oldest to newest
    prices = [row.price for row in data]
    df = pd.Series(prices)

    # Calculate EMAs
    ema_12 = df.ewm(span=12, adjust=False).mean()
    ema_26 = df.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    # Return the latest values
    macd_value = macd_line.iloc[-1]
    signal_value = signal_line.iloc[-1]

    return round(float(macd_value), 6), round(float(signal_value), 6)


def calculate_stochastic(coin, timestamp, period=14):

    # Get last N rows
    qs = (
        RickisMetrics.objects
        .filter(coin=coin, timestamp__lte=timestamp)
        .order_by("-timestamp")[:period]
    )

    if qs.count() < period:
        return None, None

    data = list(qs)[::-1]  # oldest to newest
    highs = [row.high_24h for row in data if row.high_24h is not None]
    lows = [row.price for row in data if row.price is not None]  # using current price for simplicity

    if not highs or not lows:
        return None, None

    current_price = data[-1].price
    lowest_low = min(lows)
    highest_high = max(highs)

    if highest_high == lowest_low:
        return 0.0, 0.0  # prevent division by zero

    k = 100 * (current_price - lowest_low) / (highest_high - lowest_low)

    # Create dummy %K series to calculate %D (3-period SMA of %K)
    k_series = pd.Series([
        100 * (row.price - min(lows)) / (max(highs) - min(lows)) if (max(highs) - min(lows)) != 0 else 0
        for row in data[-3:]
    ])

    d = k_series.mean()

    return round(float(k), 2), round(float(d), 2)


def calculate_support_resistance(coin, timestamp, period=20):

    qs = (
        RickisMetrics.objects
        .filter(coin=coin, timestamp__lte=timestamp)
        .order_by("-timestamp")[:period]
    )

    if qs.count() < period:
        return None, None

    data = list(qs)[::-1]  # oldest to newest

    prices = [row.price for row in data if row.price is not None]
    highs = [row.high_24h for row in data if row.high_24h is not None]

    if not prices:
        return None, None

    support = min(prices)
    resistance = max(highs) if highs else max(prices)

    return round(float(support), 6), round(float(resistance), 6)





# Rickis scanner functions -----------------------------------------------------

# High of Day Scanner
from django.db.models import Q

def get_hod_movers(request):

    from django.db.models import OuterRef, Subquery

    rickisCoins = ["BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LINK"]

    latest_metrics = RickisMetrics.objects.filter(
        coin__symbol__in=rickisCoins,
        timestamp=Subquery(
            RickisMetrics.objects.filter(coin=OuterRef("coin")).order_by("-timestamp").values("timestamp")[:1]
        )
    ).order_by("coin__symbol")

    data = []

    for row in latest_metrics:
        high_24h = row.high_24h

        # If missing or zero, look for previous non-zero high_24h for this coin
        if not high_24h or high_24h == 0:
            previous = (
                RickisMetrics.objects
                .filter(
                    coin=row.coin,
                    high_24h__isnull=False,
                    high_24h__gt=0,
                    timestamp__lt=row.timestamp
                )
                .order_by('-timestamp')
                .first()
            )
            if previous:
                high_24h = previous.high_24h

        data.append({
            "symbol": row.coin.symbol,
            "name": row.coin.name,
            "price": float(row.price),
            "high_24h": float(high_24h or 0),
            "change_5m": float(row.change_5m or 0),
            "change_1h": float(row.change_1h or 0),
            "change_24h": float(row.change_24h or 0),
            "volume": float(row.volume or 0),
            "avg_volume_1h": float(row.avg_volume_1h or 0),

            "rsi": float(row.rsi) if row.rsi is not None else None,
            "macd": float(row.macd) if row.macd is not None else None,
            "macd_signal": float(row.macd_signal) if row.macd_signal is not None else None,
            "stochastic_k": float(row.stochastic_k) if row.stochastic_k is not None else None,
            "stochastic_d": float(row.stochastic_d) if row.stochastic_d is not None else None,

            "support_level": float(row.support_level) if row.support_level is not None else None,
            "resistance_level": float(row.resistance_level) if row.resistance_level is not None else None,

            "timestamp": row.timestamp.isoformat(),
        })

    return JsonResponse(data, safe=False)












# VIEW SHORT INTERVAL DATA BY COIN ---------------------------------------------

from django.core.paginator import Paginator
from django.db.models import Prefetch

def short_interval_table_view(request):

    selected_symbol = request.GET.get("symbol")
    selected_date = request.GET.get("date")
    page_number = request.GET.get("page", 1)

    # Only fetch symbols for dropdown - we don't need all coin data
    coins = Coin.objects.values('id', 'symbol').order_by("symbol")

    # Precompute date range once
    start = datetime(2025, 3, 20)
    end = datetime(2025, 4, 22)
    date_range = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end - start).days + 1)]

    intervals = []
    actual_count = 0
    expected_count = 288  # 5-minute intervals in a day

    if selected_symbol and selected_date:
        try:
            # Get coin ID efficiently
            coin = Coin.objects.filter(symbol=selected_symbol).only('id').first()

            if coin:
                # Create timezone-aware datetime objects
                day_start = make_aware(datetime.strptime(f"{selected_date}T00:00", "%Y-%m-%dT%H:%M"))
                day_end = make_aware(datetime.strptime(f"{selected_date}T23:59:59", "%Y-%m-%dT%H:%M:%S"))

                # Use the index efficiently with coin_id first (matches your composite index)
                query = ShortIntervalData.objects.filter(
                    coin_id=coin.id,
                    timestamp__range=(day_start, day_end)
                ).order_by("timestamp")

                # Get count for displaying stats
                actual_count = query.count()

                # Add pagination to handle large result sets
                paginator = Paginator(query, 300)  # Show 300 intervals per page
                page_obj = paginator.get_page(page_number)
                intervals = page_obj

        except Exception as e:
            # Handle errors gracefully
            print(f"Error retrieving data: {e}")

    return render(request, "short_intervals.html", {
        "coins": coins,
        "date_range": date_range,
        "selected_symbol": selected_symbol,
        "selected_date": selected_date,
        "data": intervals,
        "expected_count": expected_count,
        "actual_count": actual_count,
        "page_obj": intervals if isinstance(intervals, Paginator) else None,
    })


def short_interval_summary(request):
    selected_coin_id = request.GET.get("coin")
    results = []
    coins = Coin.objects.all()

    if selected_coin_id:
        try:
            coin = Coin.objects.get(id=selected_coin_id)
        except Coin.DoesNotExist:
            coin = None

        if coin:
            start_date = datetime(2025, 3, 20)
            end_date = datetime(2025, 4, 23)  # Inclusive range

            current_date = start_date

            while current_date < end_date:

                next_day = current_date + timedelta(days=1)
                count = ShortIntervalData.objects.filter(
                    coin=coin,
                    timestamp__gte=current_date,
                    timestamp__lt=next_day
                ).count()

                results.append({
                    "date": current_date.date(),
                    "count": count,
                    "expected": 288,
                })

                current_date = next_day

    return render(request, "short_intervals.html", {
        "coins": coins,
        "selected_coin_id": int(selected_coin_id) if selected_coin_id else None,
        "results": results
    })

# INDEX ------------------------------------------------------------------------

from decimal import Decimal
from django.db.models import Q


def index_view(request):
    # Only one open trade per coin — use a dict keyed by coin
    open_signals_dict = {}
    open_signals = FiredSignal.objects.filter(result="unknown")
    for signal in open_signals:
        if signal.coin_id not in open_signals_dict:
            open_signals_dict[signal.coin_id] = signal
    open_signals = list(open_signals_dict.values())

    # Most recent 10 closed trades, sorted by closed_at DESC
    closed_signals = FiredSignal.objects.filter(result__in=["win", "loss"], closed_at__isnull=False).order_by("-closed_at")[:10]

    # Reset counters starting April 7, 2025 @ 00:01 UTC
    reset_cutoff = datetime(2025, 4, 7, 0, 1)

    long_signals = FiredSignal.objects.filter(
        fired_at__gte=reset_cutoff,
        signal_type="long",
        result__in=["win", "loss"]
    )
    short_signals = FiredSignal.objects.filter(
        fired_at__gte=reset_cutoff,
        signal_type="short",
        result__in=["win", "loss"]
    )

    def count_results(qs):
        total = qs.count()
        wins = qs.filter(result="win").count()
        losses = qs.filter(result="loss").count()
        win_rate = round((wins / total) * 100, 2) if total > 0 else 0.0
        return total, wins, losses, win_rate

    long_total, long_wins, long_losses, long_win_rate = count_results(long_signals)
    short_total, short_wins, short_losses, short_win_rate = count_results(short_signals)

    return render(request, "indextwo.html", {
        "open_signals": open_signals,
        "closed_signals": closed_signals,
        "long_total": long_total,
        "long_wins": long_wins,
        "long_losses": long_losses,
        "long_win_rate": long_win_rate,
        "short_total": short_total,
        "short_wins": short_wins,
        "short_losses": short_losses,
        "short_win_rate": short_win_rate,
    })


def update_open_trades_view(request):
    open_signals = FiredSignal.objects.filter(result="unknown")
    closed = 0
    skipped = 0

    for signal in open_signals:
        recent_metric = Metrics.objects.filter(
            coin=signal.coin,
            timestamp__gte=signal.fired_at
        ).order_by("-timestamp").first()

        if not recent_metric or not recent_metric.last_price:
            skipped += 1
            continue

        current_price = Decimal(recent_metric.last_price)
        entry = Decimal(signal.price_at_fired)

        tp_price = entry * Decimal("1.03")  # +3%
        sl_price = entry * Decimal("0.98")  # -2%

        if signal.signal_type == "long":
            if current_price >= tp_price:
                signal.result = "success"
                signal.exit_price = current_price
                signal.closed_at = now()
                signal.save()
                closed += 1
            elif current_price <= sl_price:
                signal.result = "failure"
                signal.exit_price = current_price
                signal.closed_at = now()
                signal.save()
                closed += 1

        elif signal.signal_type == "short":
            if current_price <= entry * Decimal("0.97"):  # -3%
                signal.result = "success"
                signal.exit_price = current_price
                signal.closed_at = now()
                signal.save()
                closed += 1
            elif current_price >= entry * Decimal("1.02"):  # +2%
                signal.result = "failure"
                signal.exit_price = current_price
                signal.closed_at = now()
                signal.save()
                closed += 1

    return JsonResponse({
        "status": "ok",
        "closed_trades": closed,
        "skipped_trades": skipped,
        "remaining_open": open_signals.count() - closed,
    })

# ------------------------------------------------------------------------------












# function for getting data from specific timeframes ---------------------------
def fetch_short_interval_data(coins):

    API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
    BASE_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"
    HARVARD_API_KEY = 'c35740fd-4f78-45b5-9350-c4afdd929432'

    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": API_KEY,
    }

    # put all the coins in a group of 20 because of api call limits
    coins_in_group_of_twenty = []
    coin_group = []
    count = 0

    for coin in coins:

        if count < 20:
            coin_group.append(coin)
            count += 1

        else:
            count = 1
            coins_in_group_of_twenty.append(coin_group)
            coin_group = []
            coin_group.append(coin)


    for coin_group in coins_in_group_of_twenty:
        for coin in coin_group:

            # break it up into groups of three per coin because
            # the api limit is 10000 per call

            #now = datetime.now()
            now = datetime(2025, 2, 23, 0, 0, 0)
            # 58 days ago: initial end time
            #end_time = now - timedelta(days=58)

            end_time = now

            #for i in range(3):

            print(f"starting round for {coin.symbol}")

            try:

                # 87 days ago to start
                start_time = end_time - timedelta(days=10)

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
                    print('===========================================')
                    print(' short term data ')
                    print(f"Coin: {coin.symbol}")
                    print(f"Data: {data}")

                    ShortIntervalData.objects.update_or_create(
                        coin=coin,
                        timestamp=end_time,
                        defaults={
                            "price": None,
                            "volume_5min": None,
                        },
                    )


                '''

                if "data" in data and "quotes" in data["data"]:
                    for quote in data["data"]["quotes"]:

                        Metrics.objects.update_or_create(
                            coin=coin,
                            timestamp=quote["timestamp"],
                            defaults={
                                #"daily_relative_volume": calculate_daily_relative_volume(coin),
                                "rolling_relative_volume": 0,
                                "five_min_relative_volume": 0,
                                "twenty_min_relative_volume": 0,
                                "price_change_5min": 0,
                                "price_change_10min": 0,
                                "price_change_1hr": quote["quote"]["USD"]["percent_change_1h"],
                                "price_change_24hr": quote["quote"]["USD"]["percent_change_24h"],
                                "price_change_7d": quote["quote"]["USD"]["percent_change_7d"],
                                "circulating_supply": quote["quote"]["USD"]["circulating_supply"],
                                "volume_24h": quote["quote"]["USD"]["volume_24h"],
                                "last_price": quote["quote"]["USD"]["price"],
                                "market_cap": quote["quote"]["USD"]["market_cap"]
                            },
                        )

                else:
                    print('===========================================')
                    print(' metric data failure ')
                    print(f"Coin: {coin.symbol}")
                    print(f"Data: {data}")

                #end_time = end_time + timedelta(days=29)

                '''



            except Exception as e:
                print(f"Error fetching short interval data or metric for {coin.symbol}: {e}")


        print(f"finished {coin.symbol}")

        # Pause for 30 seconds
        print("pausing for 30 seconds")
        time.sleep(30)
        print("resuming")


def calculate_all_metrics():

    API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
    BASE_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"
    HARVARD_API_KEY = 'c35740fd-4f78-45b5-9350-c4afdd929432'

    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": API_KEY,
    }


    coins = Coin.objects.all()

    for coin in coins:

        now = datetime(2025, 2, 23, 0, 0, 0)
        # 58 days ago: initial end time
        #end_time = now - timedelta(days=58)

        end_time = now

        #for i in range(3):

        print(f"starting round for {coin.symbol}")

        try:

            # 87 days ago to start
            start_time = end_time - timedelta(days=10)

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

                    Metrics.objects.update_or_create(
                        coin=coin,
                        timestamp=quote["timestamp"],
                        defaults={
                            #"daily_relative_volume": calculate_daily_relative_volume(coin),
                            "rolling_relative_volume": calculate_relative_volume(coin, quote["timestamp"]),
                            "five_min_relative_volume": calculate_five_min_relative_volume(coin, quote["timestamp"]),
                            #"twenty_min_relative_volume": calculate_twenty_min_relative_volume(coin, quote["timestamp"]),
                            "price_change_5min": calculate_price_change_five_min(coin, quote["timestamp"]),
                            "price_change_10min": calculate_price_change_thirty_min(coin, quote["timestamp"]),
                            "price_change_1hr": quote["quote"]["USD"]["percent_change_1h"],
                            "price_change_24hr": quote["quote"]["USD"]["percent_change_24h"],
                            "price_change_7d": quote["quote"]["USD"]["percent_change_7d"],
                            "circulating_supply": quote["quote"]["USD"]["circulating_supply"],
                            "volume_24h": quote["quote"]["USD"]["volume_24h"],
                            "last_price": quote["quote"]["USD"]["price"],
                            "market_cap": quote["quote"]["USD"]["market_cap"]
                        },
                    )

            else:
                print('===========================================')
                print(' metric data failure ')
                print(f"Coin: {coin.symbol}")
                print(f"Data: {data}")

            #end_time = end_time + timedelta(days=29)

        except Exception as e:
            print(f"Error fetching short interval data or metric for {coin.symbol}: {e}")



    print(f"finished {coin.symbol}")

    # Pause for 30 seconds
    print("pausing for 30 seconds")
    time.sleep(30)
    print("resuming")




# CALCULATION FUNCTIONS --------------------------------------------------------


# going to need to change this when I start using it again
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


def calculate_relative_volume(coin, timestamp):
    from scanner.models import ShortIntervalData

    # Get 30d window
    cutoff = timestamp - timedelta(days=30)

    prior_30d = ShortIntervalData.objects.filter(
        coin=coin,
        timestamp__gte=cutoff,
        timestamp__lt=timestamp
    )

    # Sum in DB — don't pull all rows into Python
    total_30d_volume = prior_30d.aggregate(total=Sum("volume_5min"))["total"] or 0

    if total_30d_volume == 0:
        return 0

    avg_volume = total_30d_volume / 30 / 24 / 12  # avg per 5m candle

    recent = ShortIntervalData.objects.filter(
        coin=coin,
        timestamp__lte=timestamp
    ).order_by("-timestamp")[:6]  # last 30 minutes (6 x 5min)

    recent_volume = sum(d.volume_5min or 0 for d in recent)

    return recent_volume / avg_volume if avg_volume else 0


def calculate_price_change_five_min(coin, timestamp):
    try:
        # Convert to timezone-aware if needed
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        if is_naive(timestamp):
            timestamp = make_aware(timestamp)

        five_min_ago = timestamp - timedelta(minutes=5)

        current_data = ShortIntervalData.objects.filter(
            coin=coin, timestamp__lte=timestamp
        ).order_by('-timestamp').first()

        past_data = ShortIntervalData.objects.filter(
            coin=coin, timestamp__lte=five_min_ago
        ).order_by('-timestamp').first()

        if not current_data or not past_data:
            print(f"⚠️ Skipping {coin.symbol}: missing current or past data")
            return None

        if not current_data.price or not past_data.price:
            print(f"⚠️ Skipping {coin.symbol}: invalid price data")
            return None

        return float((current_data.price - past_data.price) / past_data.price * 100)

    except Exception as e:
        print(f"❌ Error in calculate_price_change_five_min for {coin.symbol}: {e}")
        return None


def calculate_price_change_thirty_min(coin, timestamp):
    try:
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        if is_naive(timestamp):
            timestamp = make_aware(timestamp)

        thirty_min_ago = timestamp - timedelta(minutes=30)

        current = ShortIntervalData.objects.filter(coin=coin, timestamp__lte=timestamp).order_by('-timestamp').first()
        past = ShortIntervalData.objects.filter(coin=coin, timestamp__lte=thirty_min_ago).order_by('-timestamp').first()

        if not current or not past:
            print(f"⚠️ {coin.symbol}: Not enough data for 30-min price change")
            return None
        if not current.price or not past.price:
            print(f"⚠️ {coin.symbol}: Missing price values")
            return None
        if past.price == 0:
            return None

        return float((current.price - past.price) / past.price * 100)

    except Exception as e:
        print(f"❌ Error in 30-min price change for {coin.symbol}: {e}")
        return None


def calculate_twenty_min_relative_volume(coin, timestamp):
    try:
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        if is_naive(timestamp):
            timestamp = make_aware(timestamp)

        twenty_min_ago = timestamp - timedelta(minutes=20)

        current = ShortIntervalData.objects.filter(coin=coin, timestamp__lte=timestamp).order_by('-timestamp').first()
        if not current or current.volume_5min is None:
            print(f"⚠️ {coin.symbol}: Missing current volume data")
            return None

        past_data = ShortIntervalData.objects.filter(
            coin=coin,
            timestamp__gte=twenty_min_ago,
            timestamp__lt=timestamp
        )
        volumes = [d.volume_5min for d in past_data if d.volume_5min is not None]

        if len(volumes) < 4:
            print(f"⚠️ {coin.symbol}: Insufficient volume data (20min)")
            return None

        avg_volume = sum(volumes) / len(volumes)
        if avg_volume == 0:
            return None

        return float(current.volume_5min / avg_volume)

    except Exception as e:
        print(f"❌ Error in 20-min volume for {coin.symbol}: {e}")
        return None


def calculate_five_min_relative_volume(coin, timestamp):
    try:
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        five_min_ago = timestamp - timedelta(minutes=5)

        current_data = (
            ShortIntervalData.objects
            .filter(coin=coin, timestamp__lte=timestamp)
            .only("volume_5min")  # Load only needed field
            .order_by('-timestamp')
            .first()
        )

        previous_data = (
            ShortIntervalData.objects
            .filter(coin=coin, timestamp__lte=five_min_ago)
            .only("volume_5min")
            .order_by('-timestamp')
            .first()
        )

        if not current_data or not previous_data:
            print(f"⚠️ Skipping {coin.symbol} — missing volume points")
            return None

        if current_data.volume_5min is None or previous_data.volume_5min in (None, 0):
            return None

        return float(current_data.volume_5min / previous_data.volume_5min)

    except Exception as e:
        print(f"❌ Error calculating 5-min relative volume for {coin.symbol}: {e}")
        return None


# ------------------------------------------------------------------------------











# used to get daily high and low from a 24 hour period
from django.db.models import Max, Min
def update_historical_data():

    target_date = datetime(2025, 1, 27)
    next_date = target_date + timedelta(days=1)

    coins = Coin.objects.all()

    for coin in coins:

        existing_data = HistoricalData.objects.filter(coin=coin, date=target_date).exists()

        if not existing_data:
            # Get the most recent historical data for the coin
            recent_data = HistoricalData.objects.filter(coin=coin).order_by('-date').first()

            if recent_data:
                # Use the data from the most recent record to create a new entry for the target date
                HistoricalData.objects.create(
                    coin=coin,
                    date=target_date,
                    price=recent_data.price,
                    volume_24h=recent_data.volume_24h,
                )
                print(f"Created new HistoricalData for {coin.name} on {target_date} based on recent data.")
            else:
                # If no recent data exists, create a default entry for the coin
                HistoricalData.objects.create(
                    coin=coin,
                    date=target_date,
                    price=0.0,
                    volume_24h=0.0,
                )
                print(f"Created default HistoricalData for {coin.name} on {target_date}.")
        else:
            print(f"HistoricalData for {coin.name} on {target_date} already exists. No action taken.")


# bot message notificagtions
def send_text(true_triggers_two):

    if len(true_triggers_two) > 0:

        # telegram bot information
        chat_id_danny = '1077594551'
        #chat_id_ricki = '1054741134'
        #chat_ids = [chat_id_danny, chat_id_ricki]
        chat_ids = [chat_id_danny]
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






# INITAL SETUP FUNCTIONS -------------------------------------------------------

def load_coins():

    API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'

    HARVARD_API_KEY = 'c35740fd-4f78-45b5-9350-c4afdd929432'

    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": API_KEY,
    }

    # updated coin list as of January 24, 2025
    specific_coins = ['GMT', 'VIRTUAL', 'W', 'GRASS', 'MORPHO', 'ETHFI', 'LDO',
    'PENDLE', 'UNI', 'LUNC', 'DYDX', 'GIGA', 'ID', 'PRIME', 'EDU', 'MEW', 'APE',
    'JST', 'BONK', 'TRX', 'IOTX', 'BOME', 'POPCAT', 'JTO', 'IMX', 'CTC', 'HNT',
    'BRETT', 'KAVA', 'FARTCOIN', 'OP', 'MKR', 'ASTR', 'IO', 'MOCA', 'AIOZ', 'ZRX',
    'RUNE', 'GLM', 'ENS', 'CFX', 'OM', 'GAS', 'SC', 'KAS', 'FLOKI', 'ARB', 'CORE',
    'XMR', 'ORDI', 'STRK', 'BCH', 'CRV', 'MEME', 'WOO', 'HOT', 'FLOW', 'ADA',
    'BLUR', 'AR', 'CVX', 'AXS', 'MINA', 'WLD', 'CAKE', 'MANTA', 'CELO', 'NOT',
    'EIGEN', 'SNX', 'MNT', 'POL', 'CKB', 'ETC', 'CHZ', 'IOTA', 'ZIL', 'NEO',
    'STX', 'KSM', '1INCH', 'RON', 'CRO', 'NEAR', 'EGLD', 'ANKR', 'ZK', 'PYTH',
    'TON', 'BNB', 'PAXG', 'QTUM', 'TAO', 'MANA', 'ROSE', 'TWT', 'ETH', 'WIF',
    'THETA', 'COMP', 'EOS', 'GRT', 'ATOM', 'GALA', 'ENJ', 'RSR', 'FIL', 'XRP',
    'DOGE', 'KAIA', 'DOT', 'BSV', 'ONDO', 'BTC', 'SUSHI', 'LTC', 'AXL', 'BEAM',
    'SAND', 'SEI', 'ENA', 'XTZ', 'LINK', 'INJ', 'APT', 'SOL', 'TIA', 'ICP',
    'AKT', 'RENDER', 'VET', 'AVAX', 'XLM', 'SUI', 'PNUT', 'BAT', 'SUPER', 'JUP',
    'SAFE', 'FTM', 'DASH', 'ZRO', 'ALGO', 'AAVE', 'ONE', 'HBAR', 'ATH', 'JASMY',
    'GOAT', 'AIXBT', 'MOVE', 'PENGU', 'ZEC', 'SPX', 'LPT', 'ZEN', 'DEXE', 'PEPE',
    'BTT', 'XEC', 'SHIB', 'MOG', 'TURBO']





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


def gather_daily_historical_data():

    API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
    URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"
    HARVARD_API_KEY = 'c35740fd-4f78-45b5-9350-c4afdd929432'

    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": API_KEY,
    }

    coins = Coin.objects.all()

    coins_in_group_of_twenty = []
    coin_group = []
    count = 0

    for coin in coins:

        if count < 20:
            coin_group.append(coin)
            count += 1

        else:
            count = 1
            coins_in_group_of_twenty.append(coin_group)
            coin_group = []
            coin_group.append(coin)

    for coin_group in coins_in_group_of_twenty:
        for coin in coin_group:
            try:
                end_time = datetime.now()
                start_time = end_time - timedelta(days=37)

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
                    print('==========================================')
                    print('Historical Data error with:')
                    print(f"Coin: {coin.symbol}")
                    print(f"Data: {data}")

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

        # Pause for 45 seconds
        print("pausing for 45 seconds")
        time.sleep(45)
        print("resuming")


# DELETE FUNCTIONS --------------------------------------------------------------

def delete_old_data_custom():
    # Define the cutoff date
    cutoff_date = datetime(2025, 1, 20)

    # Delete Metrics entries
    Metrics.objects.filter(timestamp__lt=cutoff_date).delete()

    HistoricalData.objects.filter(date__lt=cutoff_date).delete()

    # Delete ShortIntervalData entries
    ShortIntervalData.objects.filter(timestamp__lt=cutoff_date).delete()

    print("Data older than January 20, 2025 has been deleted from Metrics, and ShortIntervalData.")


# once a day delete unneeded data from database
def delete_old_data():

    # delete any data from ShortIntervalData that is older than 30 days
    threshold_date = now() - timedelta(days=30)
    deleted_count, _ = ShortIntervalData.objects.filter(timestamp__lt=threshold_date).delete()
    print(f"Deleted {deleted_count} old records from ShortIntervalData.")


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








# ======================================================================


def daily_update(request=None):

    API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'

    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": API_KEY,
    }

    coins = Coin.objects.all()
    cmc_ids = [coin.cmc_id for coin in coins]

    batch_size = 50
    for i in range(0, len(cmc_ids), batch_size):
        cmc_id_batch = cmc_ids[i:i + batch_size]
        params = {
            "id": ",".join(map(str, cmc_id_batch)),
            "convert": "USD",
        }

        try:
            print("=============================")
            print("daily update...")

            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            for cmc_id in cmc_id_batch:
                if str(cmc_id) in data["data"]:
                    crypto_data = data["data"][str(cmc_id)]

                    coin = Coin.objects.get(cmc_id=cmc_id)
                    current_price = crypto_data["quote"]["USD"]["price"]
                    timestamp = datetime.strptime(crypto_data["last_updated"].rstrip("Z"), "%Y-%m-%dT%H:%M:%S.%f")
                    date = timestamp.date()

                    try:
                        HistoricalData.objects.update_or_create(
                            coin=coin,
                            date=date,
                            defaults={
                                "price": current_price,
                                "volume_24h": crypto_data["quote"]["USD"]["volume_24h"],
                            },
                        )
                        print("Created new historical data")

                    except Exception as e:
                        print("Couldn't create new historical data")
                        print(e)



        except Exception as e:
            print(f"Error updating tracked coins for batch {cmc_id_batch}: {e}")

    # get yesterday's high and low data
    daily_high_low_data()

    print("daily update complete")
    myArray = ["daily update complete"]
    send_text(myArray)

    if request:
        return JsonResponse({"status": "success", "message": "Update triggered successfully"})






# ====================================================================




# ------------------------------------------------------------------------------

import os
from django.http import FileResponse, Http404

def download_parquet_file(request, filename):
    try:
        # Adjust path to the scanner folder
        filepath = os.path.join('/workspace/scanner/', filename)
        return FileResponse(open(filepath, 'rb'), as_attachment=True)
    except FileNotFoundError:
        raise Http404("File does not exist")


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


def check_high_low(metrics_queryset):

    # check for new high or low of the day

    triggers = []
    yesterday = datetime.today() - timedelta(days=1)
    current_price = metrics_queryset[0].last_price

    try:
        high_low_data = HighLowData.objects.get(coin=metrics_queryset[0].coin, timestamp__date=yesterday.date())
        yesterdays_high = high_low_data.daily_high
        yesterdays_low = high_low_data.daily_low

        if (current_price > yesterdays_high):

            # new high of the day
            trigger = str(metrics_queryset[0].coin.symbol) + " : new HIGH of the day"

            exists = check_duplicate_triggers(trigger)
            if exists == False:

                triggers.append(trigger)

                try:
                    Trigger.objects.create(trigger_name=trigger, timestamp=now())

                except Exception as e:
                    print(f"Error creating new high/low Trigger: {e}")

        elif (current_price < yesterdays_low):

            # new low of the day
            trigger = str(metrics_queryset[0].coin.symbol) + " : new LOW of the day"

            exists = check_duplicate_triggers(trigger)
            if exists == False:

                triggers.append(trigger)

                try:
                    Trigger.objects.create(trigger_name=trigger, timestamp=now())

                except Exception as e:
                    print(f"Error creating new high/low Trigger: {e}")

        if len(triggers) > 0:
            send_text(triggers)


    except HighLowData.DoesNotExist:
        print(f"❌ No data found for {metrics_queryset[0].coin} on {yesterday.strftime('%Y-%m-%d')}")


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
        #current_volume = metrics_queryset[0].volume_24h
        #previous_volume = metrics_queryset[1].volume_24h
        #volume_growth = (current_volume - previous_volume) / previous_volume * 100

        # 5 min relative volume progression
        # current and previous 2 are increasing or equivalent
        #rvol_progression = False
        #current_rvol = metrics_queryset[0].five_min_relative_volume
        #one_previous_rvol = metrics_queryset[1].five_min_relative_volume
        #two_previous_rvol = metrics_queryset[2].five_min_relative_volume
        #if (two_previous_rvol <= one_previous_rvol <= current_rvol):
            #rvol_progression = True

        # 5 min price change is greater than previous
        #five_min_price_increase = False
        #current_five_min = metrics_queryset[0].price_change_5min
        #previous_five_min = metrics_queryset[1].price_change_5min
        #previous_five_min_two = metrics_queryset[2].price_change_5min
        #if (previous_five_min < current_five_min and
            #previous_five_min < 0 and
            #current_five_min > 0):
            #five_min_price_increase = True

        # 5 min and 10 min price changes go negative, positive, positive
        #ten_min_price_increase = False
        #current_ten_min = metrics_queryset[0].price_change_10min
        #previous_ten_min = metrics_queryset[1].price_change_10min
        #previous_ten_min_two = metrics_queryset[2].price_change_10min
        #if (previous_ten_min < current_ten_min and
            #previous_ten_min < 0):
            #ten_min_price_increase = True

        if (
            metrics_queryset[0].rolling_relative_volume > 400 and
            metrics_queryset[0].price_change_5min > 0.2 and
            metrics_queryset[1].price_change_5min > 0 and
            metrics_queryset[2].price_change_5min > 0 and
            metrics_queryset[0].price_change_10min > 0 and
            metrics_queryset[0].price_change_24hr < -7 and
            metrics_queryset[0].price_change_7d < -4
        ):
            print("TRIGGER 1 passed")
            trigger_passed = True
            updated_trigger = str(metrics_queryset[0].coin.symbol) + " : LONG (1)"
            exists = check_duplicate_triggers(updated_trigger)

            if exists == False:

                true_triggers.append(updated_trigger)

                try:
                    Trigger.objects.create(trigger_name=updated_trigger, timestamp=now())

                except Exception as e:
                    print(f"Error creating new Trigger: {e}")


        if (
            metrics_queryset[0].rolling_relative_volume >= 1000000 and
            metrics_queryset[0].five_min_relative_volume >= 1.0 and
            metrics_queryset[0].price_change_5min >= 0.2 and
            metrics_queryset[0].price_change_10min <= 0.1 and
            metrics_queryset[0].price_change_1hr >= 0 and
            metrics_queryset[0].price_change_24hr <= -0.5 and
            metrics_queryset[0].price_change_7d <= 0
        ):
            print("TRIGGER 2 passed")
            trigger_passed = True
            updated_trigger_two = str(metrics_queryset[0].coin.symbol) + " : Go LONG (2)"
            exists = check_duplicate_triggers(updated_trigger_two)

            if exists == False:

                true_triggers.append(updated_trigger_two)

                try:
                    Trigger.objects.create(trigger_name=updated_trigger_two, timestamp=now())

                except Exception as e:
                    print(f"Error creating new Trigger: {e}")


        if (
            metrics_queryset[0].price_change_24hr < -5 and
            metrics_queryset[0].rolling_relative_volume >= 100000000 and
            metrics_queryset[1].price_change_5min < metrics_queryset[0].price_change_5min and
            metrics_queryset[2].price_change_5min < metrics_queryset[1].price_change_5min and
            metrics_queryset[0].price_change_10min > 0 and
            metrics_queryset[0].price_change_1hr > 0 and
            metrics_queryset[1].price_change_1hr < metrics_queryset[0].price_change_1hr and
            metrics_queryset[2].price_change_1hr < metrics_queryset[1].price_change_1hr and

            metrics_queryset[0].price_change_5min > 10000000
        ):
            print("TRIGGER 3 passed")
            trigger_passed = True
            updated_trigger_three = str(metrics_queryset[0].coin.symbol) + " : Trigger 3 Hit (LONG) Accuracy: ~60%"
            exists = check_duplicate_triggers(updated_trigger_three)

            if exists == False:

                true_triggers.append(updated_trigger_three)

                try:
                    Trigger.objects.create(trigger_name=updated_trigger_three, timestamp=now())

                except Exception as e:
                    print(f"Error creating new Trigger: {e}")



        rolling_rvol_threshold = 1000000000
        five_min_rvol_threshold = 1.1
        price_change_5min_threshold = 0.2
        price_change_10min_threshold = 0
        price_change_1hr_threshold = -0.6
        price_change_24hr_threshold = 0.1
        price_change_7d_threshold = 6.4

        if (
            metrics_queryset[0].rolling_relative_volume >= rolling_rvol_threshold and
            #metrics_queryset[0].five_min_relative_volume >= five_min_rvol_threshold and
            metrics_queryset[0].price_change_5min <= price_change_5min_threshold and
            metrics_queryset[0].price_change_10min >= price_change_10min_threshold and
            metrics_queryset[0].price_change_1hr <= price_change_1hr_threshold and
            metrics_queryset[0].price_change_24hr >= price_change_24hr_threshold and
            metrics_queryset[0].price_change_7d <= price_change_7d_threshold
        ):
            print("TRIGGER short passed")
            trigger_passed = True
            updated_trigger_four = str(metrics_queryset[0].coin.symbol) + " : Go SHORT."
            exists = check_duplicate_triggers(updated_trigger_four)

            if exists == False:

                true_triggers.append(updated_trigger_four)

                try:
                    Trigger.objects.create(trigger_name=updated_trigger_four, timestamp=now())

                except Exception as e:
                    print(f"Error creating new Trigger: {e}")


        if (
            metrics_queryset[0].price_change_24hr < -100000000 and
            #(metrics_queryset[0].rolling_relative_volume >= 2.1 or metrics_queryset[0].daily_relative_volume >= 1.3) and
            metrics_queryset[0].price_change_5min < 0 and
            metrics_queryset[0].price_change_10min < 0 and
            metrics_queryset[0].price_change_1hr > 0 and
            metrics_queryset[1].price_change_1hr < metrics_queryset[0].price_change_1hr and
            metrics_queryset[2].price_change_1hr < metrics_queryset[1].price_change_1hr and

            metrics_queryset[0].price_change_5min > 10000000
        ):
            print("TRIGGER 5 passed")
            trigger_passed = True
            updated_trigger_five = str(metrics_queryset[0].coin.symbol) + " : Trigger Five Hit (LONG) Accuracy: ~70%"
            exists = check_duplicate_triggers(updated_trigger_five)

            if exists == False:

                true_triggers.append(updated_trigger_five)

                try:
                    Trigger.objects.create(trigger_name=updated_trigger_five, timestamp=now())

                except Exception as e:
                    print(f"Error creating new Trigger: {e}")



        # TRIGGER SIX --------------------------------------------------
        if (
            #metrics_queryset[0].daily_relative_volume >= 1.5 and
            metrics_queryset[0].rolling_relative_volume >= 1.5 and
            metrics_queryset[0].price_change_5min >= 0.7 and
            metrics_queryset[0].price_change_24hr < -5 and
            metrics_queryset[0].price_change_1hr > 0 and

            metrics_queryset[0].price_change_5min > 10000000
        ):
            print("TRIGGER 6 passed")
            trigger_passed = True
            updated_trigger_six = str(metrics_queryset[0].coin.symbol) + " : Trigger Six Hit (LONG) Accuracy: ~60%"
            exists = check_duplicate_triggers(updated_trigger_six)

            if exists == False:

                true_triggers.append(updated_trigger_six)

                try:
                    Trigger.objects.create(trigger_name=updated_trigger_six, timestamp=now())

                except Exception as e:
                    print(f"Error creating new Trigger: {e}")


        # TRIGGER SEVEN --------------------------------------------------
        if (
            #metrics_queryset[0].daily_relative_volume >= 2.0 and
            metrics_queryset[0].rolling_relative_volume >= 1.5 and
            metrics_queryset[0].price_change_5min >= 0.8 and
            metrics_queryset[0].price_change_1hr > 0 and

            metrics_queryset[0].price_change_5min > 10000000
        ):
            print("TRIGGER 7 passed")
            trigger_passed = True
            updated_trigger_seven = str(metrics_queryset[0].coin.symbol) + " : Trigger Seven Hit (LONG) Accuracy: ~50%"
            exists = check_duplicate_triggers(updated_trigger_seven)

            if exists == False:

                true_triggers.append(updated_trigger_seven)

                try:
                    Trigger.objects.create(trigger_name=updated_trigger_seven, timestamp=now())

                except Exception as e:
                    print(f"Error creating new Trigger: {e}")


        # TRIGGER EIGHT --------------------------------------------------
        if (
            metrics_queryset[0].price_change_24hr < -1000000000 and
            #(metrics_queryset[0].rolling_relative_volume >= 2.1 or metrics_queryset[0].daily_relative_volume >= 1.3) and
            metrics_queryset[0].price_change_5min < 0 and
            metrics_queryset[0].price_change_10min < 0 and
            metrics_queryset[0].price_change_1hr > 0 and
            metrics_queryset[1].price_change_1hr < metrics_queryset[0].price_change_1hr and
            metrics_queryset[2].price_change_1hr < metrics_queryset[1].price_change_1hr and

            metrics_queryset[0].price_change_5min > 10000000
        ):
            print("TRIGGER 8 passed")
            trigger_passed = True
            updated_trigger_eight = str(metrics_queryset[0].coin.symbol) + " : Trigger Eight Hit (LONG) Accuracy: ~70%"
            exists = check_duplicate_triggers(updated_trigger_eight)

            if exists == False:

                true_triggers.append(updated_trigger_eight)

                try:
                    Trigger.objects.create(trigger_name=updated_trigger_eight, timestamp=now())

                except Exception as e:
                    print(f"Error creating new Trigger: {e}")


    if trigger_passed == True:
        print("at least one trigger passed ===================================")
    else:
        print("no triggers passed")

    if len(true_triggers) > 0:
        send_text(true_triggers)
        print("not sending messages at this time.")

    return


def index(request):

    top_cryptos = []
    daily_relative_volumes = []
    sorted_volumes = []

    return render(request, "index.html", {
        "top_cryptos": [],
        "sorted_volumes": [],
        "triggers": [],
        "patterns": [],
        "support_resistance_levels": [],
    })

    # snag coins with top 25 rolling rvol
    latest_metrics = Metrics.objects.filter(coin=OuterRef('pk')).order_by('-timestamp')

    # Query top coins based on their most recent rolling_relative_volume
    coins = Coin.objects.annotate(
        latest_rvol=Subquery(latest_metrics.values('rolling_relative_volume')[:1])
    ).filter(
        latest_rvol__gt=1  # Include only coins with rolling_relative_volume > 1
    ).order_by('-latest_rvol')[:25].prefetch_related(
        Prefetch(
            'metrics',  # Prefetch metrics
            queryset=Metrics.objects.order_by('-timestamp')[:80],  # Get the 6 most recent metrics
            to_attr='prefetched_metrics'
        ),
        Prefetch(
            'short_interval_data',  # Prefetch short interval data
            queryset=ShortIntervalData.objects.order_by('-timestamp')[:80],  # Get the 6 most recent short interval data
            to_attr='prefetched_short_interval_data'
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
        #check_triggers(metrics_queryset[:6])

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

    sorted_coins = sorted(top_cryptos, key=lambda x: x["rolling_relative_volume"] or 0, reverse=True)
    sorted_volumes = sorted(daily_relative_volumes, key=lambda x: x["price_change_24h_percentage"] or 0, reverse=True)

    # Handle AJAX request for partial updates
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':

        data = {
            "top_cryptos": sorted_coins,
            "sorted_volumes": sorted_volumes,
            "triggers": triggers,
            "patterns": patterns,
            "support_resistance_levels": levels,
        }

        return JsonResponse(data, safe=False)

    # Render data to the HTML template
    return render(request, "index.html", {
        "top_cryptos": [],
        "sorted_volumes": [],
        "triggers": [],
        "patterns": [],
        "support_resistance_levels": [],
    })
















#FINN HUB STUFF ----------------------------------------------------------------
def finn_test():

    FINNHUB_API_KEY = "cuf7nohr01qno7m552hgcuf7nohr01qno7m552i0"
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    finnhub_client._session.timeout = 120

    symbol = "BINANCE:DOTUSDT"

    my_response = finnhub_client.support_resistance(symbol, '5')

    print(my_response)

    return


def finn(request=None):

    FINNHUB_API_KEY = "cuf7nohr01qno7m552hgcuf7nohr01qno7m552i0"
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    finnhub_client._session.timeout = 120

    coins = Coin.objects.order_by('market_cap_rank')[:50]

    for coin in coins:

        try:

            #symbol = "BINANCE:WIFUSDT"
            symbol = coin.exchange

            if "KUCOIN" in symbol:

                symbol = symbol.replace("USDT", "-USDT")

            elif "POLONIEX" in symbol:

                symbol = symbol.replace("USDT", "_USDT")

            elif "OKX" in symbol:

                symbol = symbol.replace("USDT", "-USDT")
                symbol = symbol.replace("OKX", "OKEX")

            five_min_response = finnhub_client.aggregate_indicator(symbol, '5')
            fifteen_min_response = finnhub_client.aggregate_indicator(symbol, '15')
            thirty_min_response = finnhub_client.aggregate_indicator(symbol, '30')
            sixty_min_response = finnhub_client.aggregate_indicator(symbol, '60')

            five_min_signal = five_min_response["technicalAnalysis"]["signal"]
            fifteen_min_signal = fifteen_min_response["technicalAnalysis"]["signal"]
            thirty_min_signal = thirty_min_response["technicalAnalysis"]["signal"]
            sixty_min_signal = sixty_min_response["technicalAnalysis"]["signal"]

            if (
                five_min_signal == "buy" and
                fifteen_min_signal == "buy" and
                thirty_min_signal == "buy" and
                sixty_min_signal == "buy"
            ):
                update = [f"BUY {coin.symbol}"]
                send_text(update)

            if (
                five_min_signal == "sell" and
                fifteen_min_signal == "sell" and
                thirty_min_signal == "sell" and
                sixty_min_signal == "sell"
            ):
                update = [f"SELL {coin.symbol}"]
                send_text(update)

        except Exception as e:

            print(f"Error fetching data for {coin.symbol}: {e}")
            # Skip coin if error occurs
            continue

    if request:
        return JsonResponse({"status": "success", "message": "Update successfully"})

    return


    #symbol = "BINANCE:DOTUSDT"
    #my_response = finnhub_client.support_resistance(symbol, 'D')
    #print(my_response)
    #levels = my_response["levels"]
    #support = min(levels) if levels else None
    #resistance = max(levels) if levels else None
    #print(support)
    #print(resistance)


def check_support_resistance(request=None):

    # check and see if current coin price is near the support or resistance level

    coins = Coin.objects.all()

    for coin in coins:

        try:

            levels = SupportResistance.objects.filter(coin=coin)
            latest_metric = Metrics.objects.filter(coin=coin).order_by('-timestamp').first()

            if levels and latest_metric:

                price = latest_metric.last_price

                if len(levels) == 6:

                    level_one = levels.level_one
                    level_two = levels.level_two
                    level_three = levels.level_three
                    level_four = levels.level_four
                    level_five = levels.level_five
                    level_six = levels.level_six

                elif len(levels) == 5:

                    level_one = levels.level_one
                    level_two = levels.level_two
                    level_three = levels.level_three
                    level_four = levels.level_four
                    level_five = levels.level_five
                    level_six = 0

                elif len(levels) == 4:

                    level_one = levels.level_one
                    level_two = levels.level_two
                    level_three = levels.level_three
                    level_four = levels.level_four
                    level_five = 0
                    level_six = 0

                elif len(levels) == 3:

                    level_one = levels.level_one
                    level_two = levels.level_two
                    level_three = levels.level_three
                    level_four = 0
                    level_five = 0
                    level_six = 0

                elif len(levels) == 2:

                    level_one = levels.level_one
                    level_two = levels.level_two
                    level_three = 0
                    level_four = 0
                    level_five = 0
                    level_six = 0

                elif len(levels) == 1:

                    level_one = levels.level_one
                    level_two = 0
                    level_three = 0
                    level_four = 0
                    level_five = 0
                    level_six = 0


                upper_price = price * 1.02
                lower_price = price * 0.98

                if (
                    lower_price <= level_one <= upper_price or
                    lower_price <= level_two <= upper_price or
                    lower_price <= level_three <= upper_price or
                    lower_price <= level_four <= upper_price or
                    lower_price <= level_five <= upper_price or
                    lower_price <= level_six <= upper_price
                ):

                    print("Level is within +/- 2% of price")

                    # send message
                    update = [f"Level is within +/- 2% of {coin.symbol} price {price}"]
                    send_text(update)

        except Exception as e:

            print(f"Error fetching data for {coin.symbol}: {e}")
            # Skip coin if error occurs
            continue

    if request:
        return JsonResponse({"status": "success", "message": "Update successfully"})

    return


def support_resistance(request=None):

    FINNHUB_API_KEY = "cuf7nohr01qno7m552hgcuf7nohr01qno7m552i0"
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    finnhub_client._session.timeout = 120

    #coins = Coin.objects.order_by('market_cap_rank')[:50]

    coins = Coin.objects.order_by('market_cap_rank')

    for coin in coins:

        try:

            #symbol = "BINANCE:WIFUSDT"
            symbol = coin.exchange

            if "KUCOIN" in symbol:

                symbol = symbol.replace("USDT", "-USDT")

            elif "POLONIEX" in symbol:

                symbol = symbol.replace("USDT", "_USDT")

            elif "OKX" in symbol:

                symbol = symbol.replace("USDT", "-USDT")
                symbol = symbol.replace("OKX", "OKEX")

            my_response = finnhub_client.support_resistance(symbol, 'D')

            levels = my_response["levels"]

            if len(levels) == 6:

                level_one = levels[0]
                level_two = levels[1]
                level_three = levels[2]
                level_four = levels[3]
                level_five = levels[4]
                level_six = levels[5]

            elif len(levels) == 5:

                level_one = levels[0]
                level_two = levels[1]
                level_three = levels[2]
                level_four = levels[3]
                level_five = levels[4]
                level_six = 0

            elif len(levels) == 4:

                level_one = levels[0]
                level_two = levels[1]
                level_three = levels[2]
                level_four = levels[3]
                level_five = 0
                level_six = 0

            elif len(levels) == 3:

                level_one = levels[0]
                level_two = levels[1]
                level_three = levels[2]
                level_four = 0
                level_five = 0
                level_six = 0

            elif len(levels) == 2:

                level_one = levels[0]
                level_two = levels[1]
                level_three = 0
                level_four = 0
                level_five = 0
                level_six = 0

            elif len(levels) == 1:

                level_one = levels[0]
                level_two = 0
                level_three = 0
                level_four = 0
                level_five = 0
                level_six = 0

            else:

                level_one = 0
                level_two = 0
                level_three = 0
                level_four = 0
                level_five = 0
                level_six = 0

            SupportResistance.objects.update_or_create(
                coin = coin,
                defaults = {
                    "level_one": level_one,
                    "level_two": level_two,
                    "level_three": level_three,
                    "level_four": level_four,
                    "level_five": level_five,
                    "level_six": level_six,
                    "timestamp": datetime.utcnow()
                }
            )

        except Exception as e:

            print(f"Error fetching data for {coin.symbol}: {e}")
            # Skip coin if error occurs
            continue

    if request:
        return JsonResponse({"status": "success", "message": "Update successfully"})

    return


    # PLAN
    # every 30 min check every coin for a pattern on the one hour
    #     if there is an incomplete pattern, add to db
    #     only add the 1 hour pattern recognition data, the rest is null
    # every 5 min check on the coins that have an incomplete pattern
    #     update all the relevant info
    #     when the pattern is complete, delete from db


def thirty_min_pattern_check(request=None):

    count = 0

    patterns = Pattern.objects.all()
    if (len(patterns) < 20):

        FINNHUB_API_KEY = "cuf7nohr01qno7m552hgcuf7nohr01qno7m552i0"
        finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
        finnhub_client._session.timeout = 120

        # check all coins that don't currently have an incomplete pattern
        # get coins without pattern
        coins = Coin.objects.filter(pattern__isnull=True)

        # loop through coins
        for coin in coins:

            if (count == 20):
                break

            print(f"checking {coin.symbol}")

            try:

                has_pattern = Pattern.objects.filter(coin=coin).exists()

                if has_pattern:
                    continue

                # fix the exchange string if necessary
                #symbol = "BINANCE:BTCUSDT"
                symbol = coin.exchange.upper()

                if "KUCOIN" in symbol:

                    symbol = symbol.replace("USDT", "-USDT")

                elif "POLONIEX" in symbol:

                    symbol = symbol.replace("USDT", "_USDT")

                elif "OKX" in symbol:

                    symbol = symbol.replace("USDT", "-USDT")
                    symbol = symbol.replace("OKX", "OKEX")

                one_hour_patterns = finnhub_client.pattern_recognition(symbol, '60')

                print("got the one hour pattern")

                if (
                    not one_hour_patterns or
                    'points' not in one_hour_patterns or
                    len(one_hour_patterns["points"]) == 0
                ):

                    # Skip if no pattern detected
                    continue

                # Take the first detected pattern
                pattern_data = one_hour_patterns["points"][0]

                # one hour pattern recognition
                name = pattern_data["patternname"]
                patterntype = pattern_data["patterntype"]
                status = pattern_data["status"]

                print(f"pattern status: {status}")

                if status != "incomplete":
                    continue

                entry = pattern_data["entry"]
                takeprofit = pattern_data["profit1"]
                stoploss = pattern_data["stoploss"]

                # one hour support / resistance
                #one_hour_support_resistance = finnhub_client.support_resistance(symbol, '60')
                #support = one_hour_support_resistance["support"][0] if one_hour_support_resistance["support"] else None
                #resistance = one_hour_support_resistance["resistance"][0] if one_hour_support_resistance["resistance"] else None

                #five_min_aggregate = finnhub_client.aggregate_indicator(symbol, '5')
                #fifteen_min_aggregate = finnhub_client.aggregate_indicator(symbol, '15')
                #one_hour_aggregate = finnhub_client.aggregate_indicator(symbol, '60')

                # aggregates
                #five_min_signal = five_min_aggregate["technicalAnalysis"]["signal"]
                #five_min_adx = five_min_aggregate["trend"]["adx"]

                #fifteen_min_signal = fifteen_min_aggregate["technicalAnalysis"]["signal"]
                #fifteen_min_adx = fifteen_min_aggregate["trend"]["adx"]

                #one_hour_signal = one_hour_aggregate["technicalAnalysis"]["signal"]
                #one_hour_adx = one_hour_aggregate["trend"]["adx"]

                Pattern.objects.update_or_create(
                    coin = coin,
                    defaults = {
                        "symbol": coin.symbol,
                        # one hour pattern
                        "name": name,
                        # one hour pattern
                        "patterntype": patterntype,
                        # one hour pattern
                        "status": status,
                        # one hour pattern
                        "entry": entry,
                        # one hour pattern
                        "takeprofit": takeprofit,
                        # one hour pattern
                        "stoploss": stoploss,
                        # one hour pattern
                        #"support": support,
                        # one hour pattern
                        #"resistance": resistance,
                        # five min aggregate
                        #"five_min_signal": five_min_signal,
                        # fifteen min aggregate
                        #"fifteen_min_signal": fifteen_min_signal,
                        # one hour aggregate
                        #"one_hour_signal": one_hour_signal,
                        # five min aggregate
                        #"five_min_adx": five_min_adx,
                        # fifteen min aggregate
                        #"fifteen_min_adx": fifteen_min_adx,
                        # one hour aggregate
                        #"one_hour_adx": one_hour_adx,
                        # time now
                        #"timestamp": datetime.utcnow()
                    }
                )

                count += 1

                # send message
                update = [f"{patterntype} {name} pattern detected for {coin.symbol} on the 1hr. Status: {status}"]
                send_text(update)

            except Exception as e:

                print(f"Error fetching data for {coin.symbol}: {e}")
                # Skip coin if error occurs
                continue


    if request:
        return JsonResponse({"status": "success", "message": "Update successfully"})

    return


def five_min_pattern_check(request=None):

    FINNHUB_API_KEY = "cuf7nohr01qno7m552hgcuf7nohr01qno7m552i0"
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    finnhub_client._session.timeout = 120

    #coins_with_patterns = Coin.objects.filter(pattern__isnull=False).distinct()

    patterns = Pattern.objects.all()

    for pattern in patterns:

        try:

            coin = pattern.coin

            symbol = coin.exchange

            if "KUCOIN" in symbol:

                symbol = symbol.replace("USDT", "-USDT")

            elif "POLONIEX" in symbol:

                symbol = symbol.replace("USDT", "_USDT")

            elif "OKX" in symbol:

                symbol = symbol.replace("USDT", "-USDT")
                symbol = symbol.replace("OKX", "OKEX")

            one_hour_patterns = finnhub_client.pattern_recognition(symbol, '60')

            if (
                not one_hour_patterns or
                'points' not in one_hour_patterns or
                len(one_hour_patterns["points"]) == 0
            ):

                # no pattern data is found for a coin that should have data
                # skip it for now but might need to change this later
                continue

            # Take the first detected pattern
            pattern_data = one_hour_patterns["points"][0]

            # one hour pattern recognition
            name = pattern_data["patternname"]
            patterntype = pattern_data["patterntype"]
            status = pattern_data["status"]

            # if pattern is completed, send message and delete from db
            if status != "incomplete":

                # send message
                update = [f"{coin.symbol} pattern {name} ({patterntype}) complete: {status}"]
                send_text(update)

                # delete from db - could cause an error if a coin has mulitple patterns on the 1hr
                # not deleting right now. collecting as much data as i can to test these triggers against
                Pattern.objects.filter(coin=coin).delete()

                continue

            entry = pattern_data["entry"]
            takeprofit = pattern_data["profit1"]
            stoploss = pattern_data["stoploss"]

            # one hour support / resistance
            #one_hour_support_resistance = finnhub_client.support_resistance(symbol, '60')
            #support = one_hour_support_resistance["support"][0] if one_hour_support_resistance["support"] else None
            #resistance = one_hour_support_resistance["resistance"][0] if one_hour_support_resistance["resistance"] else None

            five_min_aggregate = finnhub_client.aggregate_indicator(symbol, '5')
            fifteen_min_aggregate = finnhub_client.aggregate_indicator(symbol, '15')
            one_hour_aggregate = finnhub_client.aggregate_indicator(symbol, '60')

            # aggregates
            five_min_signal = five_min_aggregate["technicalAnalysis"]["signal"]
            five_min_adx = five_min_aggregate["trend"]["adx"]

            fifteen_min_signal = fifteen_min_aggregate["technicalAnalysis"]["signal"]
            fifteen_min_adx = fifteen_min_aggregate["trend"]["adx"]

            one_hour_signal = one_hour_aggregate["technicalAnalysis"]["signal"]
            one_hour_adx = one_hour_aggregate["trend"]["adx"]

            pattern.patterntype = patterntype
            pattern.status = status
            pattern.entry = entry
            pattern.takeprofit = takeprofit
            pattern.stoploss = stoploss
            pattern.five_min_signal = five_min_signal
            pattern.fifteen_min_signal = fifteen_min_signal
            pattern.one_hour_signal = one_hour_signal
            pattern.five_min_adx = five_min_adx
            pattern.fifteen_min_adx = fifteen_min_adx
            pattern.one_hour_adx = one_hour_adx

            pattern.save()

            # if bullish check for a long signal, bearish check for short signal
            if (patterntype == "bullish"):
                pattern_long_signal(coin)
            elif (patterntype == "bearish"):
                pattern_short_signal(coin)

        except Exception as e:

            print(f"Error fetching data for {coin.symbol}: {e}")
            # Skip coin if error occurs
            continue

    if request:
        return JsonResponse({"status": "success", "message": "Update successfully"})

    return


def pattern_long_signal(coin):

    # check current patterns for a buy signal
    # patterntype == bullish
    # status == incomplete
    # five min signal = buy
    # fifteen min signal = buy
    # one hour signal = buy
    # all three adx values > 25
    # suggested entry price is 'close' to current price, within 2%

    # OPTIONAL: check technical indicators
    #

    pattern = Pattern.objects.filter(coin=coin).order_by('-timestamp').first()
    metrics = Metrics.objects.filter(coin=coin).order_by('-timestamp').first()
    current_price = metrics.last_price
    entry_price = pattern.entry

    # check if current price is within 2% of the entry price
    close_price = False
    if (entry_price != None and current_price != None):

        price_difference = abs(current_price - entry_price)
        percentage_difference = (price_difference / entry_price) * 100

        if (percentage_difference <= 2):

            close_price = True

    if (
        pattern.five_min_adx != None and
        pattern.fifteen_min_adx != None and
        pattern.one_hour_adx != None
        ):

        if (
            pattern.patterntype == "bullish" and
            pattern.status == "incomplete" and
            pattern.five_min_signal == "buy" and
            pattern.fifteen_min_signal == "buy" and
            pattern.one_hour_signal == "buy" and
            pattern.five_min_adx > 25 and
            pattern.fifteen_min_adx > 25 and
            pattern.one_hour_adx > 25 and
            close_price == True
        ):

            # long signal triggered
            # send message
            long_signal = [f"LONG TRIGGERED: {coin.symbol} pattern {pattern.name}. Entry: {entry_price}, Current Price: {current_price}"]
            send_text(long_signal)

    return


def pattern_short_signal(coin):

    # check current patterns for a sell signal
    # patterntype == bearish
    # status == incomplete
    # five min signal = sell
    # fifteen min signal = sell
    # one hour signal = sell
    # all three adx values > 25
    # suggested entry price is 'close' to current price, within 2%

    pattern = Pattern.objects.filter(coin=coin).order_by('-timestamp').first()
    metrics = Metrics.objects.filter(coin=coin).order_by('-timestamp').first()
    current_price = metrics.last_price
    entry_price = pattern.entry

    # check if current price is within 2% of the entry price
    close_price = False
    if (entry_price != None and current_price != None):

        price_difference = abs(current_price - entry_price)
        percentage_difference = (price_difference / entry_price) * 100

        if (percentage_difference <= 2):

            close_price = True

    if (
        pattern.five_min_adx != None and
        pattern.fifteen_min_adx != None and
        pattern.one_hour_adx != None
        ):

        if (
            pattern.patterntype == "bearish" and
            pattern.status == "incomplete" and
            pattern.five_min_signal == "sell" and
            pattern.fifteen_min_signal == "sell" and
            pattern.one_hour_signal == "sell" and
            pattern.five_min_adx > 25 and
            pattern.fifteen_min_adx > 25 and
            pattern.one_hour_adx > 25 and
            close_price == True
        ):

            # short signal triggered
            # send message
            short_signal = [f"SHORT TRIGGERED: {coin.symbol} pattern {pattern.name}. Entry: {entry_price}, Current Price: {current_price}"]
            send_text(short_signal)

    return


def pattern_recognition():

    FINNHUB_API_KEY = "cuf7nohr01qno7m552hgcuf7nohr01qno7m552i0"
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    finnhub_client._session.timeout = 120

    #coins = Coin.objects.all()
    coins = Coin.objects.all()[:80]

    '''
    DOT = Coin.objects.get(symbol="DOT")
    XRP = Coin.objects.get(symbol="XRP")
    ORDI = Coin.objects.get(symbol="ORDI")
    SAND = Coin.objects.get(symbol="SAND")
    UNI = Coin.objects.get(symbol="UNI")
    DYDX = Coin.objects.get(symbol="DYDX")
    ALGO = Coin.objects.get(symbol="ALGO")
    DOGE = Coin.objects.get(symbol="DOGE")
    GRT = Coin.objects.get(symbol="GRT")
    SHIB = Coin.objects.get(symbol="SHIB")
    coins = []
    coins.append(DOT)
    coins.append(XRP)
    coins.append(ORDI)
    coins.append(SAND)
    coins.append(UNI)
    coins.append(DYDX)
    coins.append(ALGO)
    coins.append(DOGE)
    coins.append(GRT)
    coins.append(SHIB)
    '''

    incomplete_patterns = []

    for coin in coins:

        # going through coins on Binance only to test it out
        if "BINANCE" in coin.exchange:

            try:

                #symbol = "BINANCE:BTCUSDT"
                symbol = coin.exchange

                one_hour_patterns = finnhub_client.pattern_recognition(symbol, '60')

                if not one_hour_patterns or 'points' not in one_hour_patterns:
                    continue  # Skip if no pattern detected

                # Take the first detected pattern
                pattern_data = one_hour_patterns["points"][0]

                # one hour pattern recognition
                name = pattern_data["patternname"]
                patterntype = pattern_data["patterntype"]
                status = pattern_data["status"]

                if status == "successful" or status == "complete" or status == "failed":
                    continue

                entry = pattern_data["entry"]
                takeprofit = pattern_data["profit1"]
                stoploss = pattern_data["stoploss"]

                # one hour support / resistance
                #one_hour_support_resistance = finnhub_client.support_resistance(symbol, '60')
                #support = one_hour_support_resistance["support"][0] if one_hour_support_resistance["support"] else None
                #resistance = one_hour_support_resistance["resistance"][0] if one_hour_support_resistance["resistance"] else None

                five_min_aggregate = finnhub_client.aggregate_indicator(symbol, '5')
                fifteen_min_aggregate = finnhub_client.aggregate_indicator(symbol, '15')
                one_hour_aggregate = finnhub_client.aggregate_indicator(symbol, '60')

                # aggregates
                five_min_signal = five_min_aggregate["technicalAnalysis"]["signal"]
                five_min_adx = five_min_aggregate["trend"]["adx"]

                fifteen_min_signal = fifteen_min_aggregate["technicalAnalysis"]["signal"]
                fifteen_min_adx = fifteen_min_aggregate["trend"]["adx"]

                one_hour_signal = one_hour_aggregate["technicalAnalysis"]["signal"]
                one_hour_adx = one_hour_aggregate["trend"]["adx"]

                print("---------------------------------")
                print(coin.symbol)
                print(f"name: {name}")
                print(f"patterntype: {patterntype}")
                print(f"status: {status}")
                print(f"entry: {entry}")
                print(f"takeprofit: {takeprofit}")
                print(f"stoploss: {stoploss}")
                print(f"five_min_signal: {five_min_signal}")
                print(f"fifteen_min_signal: {fifteen_min_signal}")
                print(f"one_hour_signal: {one_hour_signal}")
                print(f"five_min_adx: {five_min_adx}")
                print(f"fifteen_min_adx: {fifteen_min_adx}")
                print(f"one_hour_adx: {one_hour_adx}")

                Pattern.objects.update_or_create(
                    coin = coin,
                    defaults = {
                        # one hour pattern
                        "name": name,
                        # one hour pattern
                        "patterntype": patterntype,
                        # one hour pattern
                        "status": status,
                        # one hour pattern
                        "entry": entry,
                        # one hour pattern
                        "takeprofit": takeprofit,
                        # one hour pattern
                        "stoploss": stoploss,
                        # one hour pattern
                        #"support": support,
                        # one hour pattern
                        #"resistance": resistance,
                        # five min aggregate
                        "five_min_signal": five_min_signal,
                        # fifteen min aggregate
                        "fifteen_min_signal": fifteen_min_signal,
                        # one hour aggregate
                        "one_hour_signal": one_hour_signal,
                        # five min aggregate
                        "five_min_adx": five_min_adx,
                        # fifteen min aggregate
                        "fifteen_min_adx": fifteen_min_adx,
                        # one hour aggregate
                        "one_hour_adx": one_hour_adx,
                        # time now
                        "timestamp": datetime.utcnow()
                    }
                )



            except Exception as e:
                print(f"Error fetching data for {coin.symbol}: {e}")
                continue  # Skip coin if error occurs

    print("✅ Trade signals updated successfully!")


    # NEED A FUNCTION TO DELETE OLD PATTERN DATA FROM DB




    '''
    # do whatever with the patterns
    for pattern in incomplete_patterns:

        print("----------------------------------------------")
        print(f"Coin: {pattern['coin']}")
        print(f"Pattern Name: {pattern['patternname']}")
        print(f"Pattern Type: {pattern['patterntype']}")
        print(f"Pattern Status: {pattern['status']}")
        print(f"Anticipated Entry Price: ${pattern['entry']}")
        print(f"Recommended TP #1: ${pattern['profit1']}")
        print(f"Recommended SL: ${pattern['stoploss']}")
        print(f"five_min_signal: {pattern['five_min_signal']}")
        print(f"fifteen_min_signal: {pattern['fifteen_min_signal']}")
        print(f"one_hour_signal: {pattern['one_hour_signal']}")
        print(f"five_min_adx: {pattern['five_min_adx']}")


    if len(incomplete_patterns) == 0:
        print("no patterns found")



    return incomplete_patterns
    '''






    '''
    ----------------------------------------------------------------------------
    Potential Trigger Setup:
    ✔ A bullish pattern is detected
    ✔ Price is near support (bounce) or breaking resistance (breakout)
    ✔ Technical Rating is "BUY" or "STRONG BUY"
    ✔ Entry price is close to the trigger level
    ✔ Bullish ADX reading
    ----------------------------------------------------------------------------
    '''

    # bullish pattern detected
    #    loop through all coins, get all the patterns
    #    save the incomplete bullish patterns
    # check price against support or resistance
    #    if price is right above a support level
    #    if price is right below a resistance level
    # get the technical rating
    #    use the aggregate indicator to get the signal
    # check price against entry price
    #    get entry price from pattern detection
    # check the ADX reading


def daily_high_low_data():

    print("in daily high low data function -------------------------")

    FINNHUB_API_KEY = "cuf7nohr01qno7m552hgcuf7nohr01qno7m552i0"
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

    # collect yesterday's high and low price -----------------------------------
    # loop through all coins
    coins = Coin.objects.all()

    # 100 coins at a time (API limit is 150 calls/min)
    batch_size = 100
    total_coins = len(coins)

    # get the date for yesterday
    yesterday = datetime.today() - timedelta(days=1)
    start_timestamp = int(datetime(yesterday.year, yesterday.month, yesterday.day, 0, 0).timestamp())
    end_timestamp = int(datetime(yesterday.year, yesterday.month, yesterday.day, 23, 59, 59).timestamp())

    #for coin in coins:
    for index, coin in enumerate(coins, start=1):

        # get coin symbol
        symbol = coin.symbol.upper()

        # get coin exchange
        # options: "KRAKEN","HITBTC","COINBASE","GEMINI","POLONIEX","Binance",
        #          "ZB","BITTREX","KUCOIN","OKEX","BITFINEX","HUOBI"

        # format - "BINANCE:BTCUSDT"
        symbolString = coin.exchange.upper()

        if "KUCOIN" in symbolString:

            symbolString = symbolString.replace("USDT", "-USDT")

        elif "POLONIEX" in symbolString:

            symbolString = symbolString.replace("USDT", "_USDT")

        elif "OKX" in symbolString:

            symbolString = symbolString.replace("USDT", "-USDT")
            symbolString = symbolString.replace("OKX", "OKEX")

        resolution = "D"

        try:
            highLowData = finnhub_client.crypto_candles(symbolString, resolution, start_timestamp, end_timestamp)

            if highLowData and highLowData["s"] == "ok" and "h" in highLowData and "l" in highLowData:
                if highLowData["h"] and highLowData["l"]:
                    high_price = max(highLowData["h"])
                    low_price = min(highLowData["l"])

                    # save the high and low
                    HighLowData.objects.create(
                        coin=coin,
                        daily_high=high_price,
                        daily_low=low_price,
                        timestamp=datetime(yesterday.year, yesterday.month, yesterday.day)
                    )

                    print(f"✅ {symbol}: High={high_price}, Low={low_price} saved for {yesterday.strftime('%Y-%m-%d')}")

                else:
                    print(f"⚠️ {symbol}: No valid high/low data available for {yesterday.strftime('%Y-%m-%d')}")

            else:
                print(f"❌ Error fetching high/low data for {symbol}")

        except Exception as e:
            print(f"❌ API error for {symbol}: {str(e)}")

        '''
        if index % batch_size == 0 and index < total_coins:
            print(f"⏳ Processed {index} coins, pausing for 60 seconds to respect API limits...")
            time.sleep(60)
        '''

    return


# ------------------------------------------------------------------------------





# TRIGGER TESTING FUNCTIONS ----------------------------------------------------

# use to take a coin and timestamp and check if it went up x% before going down y%
def check_future_price(coin, timestamp):

    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)

    metrics = Metrics.objects.filter(coin=coin).order_by("timestamp")

    if not metrics.exists():
        print(f"No metrics found for {coin.symbol}")
        return None

    closest_metric = min(metrics, key=lambda m: abs(m.timestamp - timestamp))
    end_time = closest_metric.timestamp + timedelta(hours=10)
    future_metrics = metrics.filter(timestamp__gte=closest_metric.timestamp, timestamp__lte=end_time)

    if not future_metrics.exists():
        print(f"No future metrics found for {coin.symbol}")
        return None

    current_price = closest_metric.last_price

    try:

        take_profit_price = current_price + (current_price * decimal.Decimal(0.04))
        stop_loss_price = current_price - (current_price * decimal.Decimal(0.02))

        for metric in future_metrics:

            take_profit_hit = False
            stop_loss_hit = False

            if metric.last_price >= take_profit_price:
                take_profit_hit = True
                break

            if metric.last_price <= stop_loss_price:
                stop_loss_hit = True
                break

        if (take_profit_hit == True):
            return True

        elif (stop_loss_hit == True):
            return False

        else:
            return None

    except Exception as e:
        print(f"Error: {e}")

    return None


def hourly_candles():

    API_KEY = '7dd5dd98-35d0-475d-9338-407631033cd9'
    #BASE_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"
    BASE_URL = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/historical"
    #HARVARD_API_KEY = 'c35740fd-4f78-45b5-9350-c4afdd929432'

    HEADERS = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": API_KEY,
    }

    now = datetime.utcnow()
    start_time = now - timedelta(days=29)

    coins = Coin.objects.all()

    #coin = Coin.objects.filter(symbol="DOT")
    #coin_id = 9481

    count = 0

    amount_of_trades = 0
    successful_trades = 0

    for coin in coins:

        coin_id = coin.cmc_id
        if not coin_id:
            continue

        params = {
            "id": coin_id,
            "time_period": "hourly",
            "time_start": start_time.isoformat(),
            "time_end": now.isoformat(),
            "interval": "1h",
            "convert": "USD",
        }

        response = requests.get(BASE_URL, headers=HEADERS, params=params)
        data = response.json()

        count += 1

        if "data" in data and "quotes" in data["data"]:

            candles = data["data"]["quotes"]

            for i in range(1, len(candles)-1):

                prev_candle = candles[i-1]
                curr_candle = candles[i]

                prev_open = prev_candle["quote"]["USD"]["open"]
                prev_close = prev_candle["quote"]["USD"]["close"]
                curr_open = curr_candle["quote"]["USD"]["open"]
                curr_close = curr_candle["quote"]["USD"]["close"]

                if (
                    prev_close < prev_open and  # Previous candle is bearish
                    curr_close > curr_open and  # Current candle is bullish
                    curr_open < prev_close and  # Current open is lower than previous close
                    curr_close > prev_open      # Current close is higher than previous open
                ):
                    next_candle = candles[i+1]
                    price_now = curr_candle["quote"]["USD"]["close"]
                    price = next_candle["quote"]["USD"]["high"]
                    percentage_change = 0

                    if (price_now != 0 and price_now != None):
                        percentage_change = ((price-price_now)/price_now)*100

                    timestamp = curr_candle['quote']['USD']['timestamp']

                    print(f"Bullish engulfing candle for {coin.symbol}:")
                    print(f"Timestamp: {curr_candle['quote']['USD']['timestamp']}")
                    print(f"Price entering next candle: {price_now}")
                    print(f"High price within next hour: {price}")
                    print(f"Percentage change: {round(percentage_change, 2)}%")

                    amount_of_trades += 1

                    result = check_future_price(coin, timestamp)

                    if result == None:
                        print(f"Result: None")
                        amount_of_trades -= 1

                    elif result == True:
                        print(f"Result: Successful trade")
                        successful_trades += 1

                    else:
                        print(f"Result: Failed trade")

                    print("--------------------------------")



        else:
            print("failed to get data")

        success_rate = (successful_trades/amount_of_trades) * 100
        print(f"Amount of trades thus far: {amount_of_trades}")
        print(f"Success rate: {success_rate}")

        if count >= 20:
            count = 0
            print("Pausing for 60 seconds...")
            time.sleep(60)


def find_tp_sl():

    coins = Coin.objects.all()
    results = {}

    for coin in coins:
        max_increase = Metrics.objects.filter(coin=coin).aggregate(Max('price_change_1hr'))['price_change_1hr__max']
        max_decrease = Metrics.objects.filter(coin=coin).aggregate(Min('price_change_1hr'))['price_change_1hr__min']

        results[coin.symbol] = {
            "max_increase": max_increase,
            "max_decrease": max_decrease
        }

        print(f"{coin.symbol}: Max Increase: {max_increase:.2f}%, Max Decrease: {max_decrease:.2f}%")

    return results


# used to test out a trigger combination against the data we have in the db
def check_trigger():

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
    trigger_six_trades = 0
    trigger_six_success = 0
    trigger_seven_trades = 0
    trigger_seven_success = 0


    jan_13_count = 0
    jan_14_count = 0
    jan_15_count = 0
    jan_16_count = 0
    jan_17_count = 0
    jan_18_count = 0
    jan_19_count = 0
    jan_20_count = 0
    jan_21_count = 0
    jan_22_count = 0
    jan_23_count = 0
    jan_24_count = 0
    jan_25_count = 0
    jan_26_count = 0
    jan_27_count = 0
    jan_28_count = 0
    jan_29_count = 0
    jan_30_count = 0
    jan_31_count = 0
    feb_1_count = 0
    feb_2_count = 0
    feb_3_count = 0
    feb_4_count = 0
    feb_5_count = 0
    feb_6_count = 0
    feb_7_count = 0
    feb_8_count = 0
    feb_9_count = 0
    feb_10_count = 0
    feb_11_count = 0
    feb_12_count = 0
    feb_13_count = 0
    feb_14_count = 0
    feb_15_count = 0
    feb_16_count = 0
    feb_17_count = 0
    feb_18_count = 0
    feb_19_count = 0
    feb_20_count = 0
    feb_21_count = 0
    feb_22_count = 0
    feb_23_count = 0

    for coin in coins:

        metrics = Metrics.objects.filter(coin=coin).order_by('timestamp')

        if not metrics:
            continue

        if not len(metrics) > 10:
            continue

        trigger_one_hit_counter = 0
        trigger_one_hit = False
        trigger_two_hit_counter = 0
        trigger_two_hit = False
        trigger_three_hit_counter = 0
        trigger_three_hit = False
        trigger_short_hit_counter = 0
        trigger_short_hit = False
        trigger_five_hit_counter = 0
        trigger_five_hit = False
        trigger_six_hit_counter = 0
        trigger_six_hit = False
        trigger_seven_hit_counter = 0
        trigger_seven_hit = False

        for x in range(6, len(metrics)):

            day = metrics[x].timestamp.day

            if (metrics[x].rolling_relative_volume != None and
                metrics[x].price_change_5min != None and
                metrics[x].price_change_10min != None and
                metrics[x].price_change_1hr != None and
                metrics[x].price_change_24hr != None and
                metrics[x].five_min_relative_volume != None and
                metrics[x].twenty_min_relative_volume != None and
                metrics[x-1].price_change_5min != None and
                metrics[x-2].price_change_5min != None and
                metrics[x-3].price_change_5min != None and
                metrics[x-1].price_change_10min != None and
                metrics[x-2].price_change_10min != None and
                metrics[x-3].price_change_10min != None):


                # TRIGGER 1 ----------------------------------------------------
                if (trigger_one_hit == True):
                    trigger_one_hit_counter += 1

                if (trigger_one_hit_counter > 13):
                    trigger_one_hit = False
                    trigger_one_hit_counter = 0

                if (
                    trigger_one_hit == False and
                    metrics[x].rolling_relative_volume > 350 and
                    metrics[x].price_change_5min > 0.35 and
                    metrics[x].price_change_24hr < -7 and
                    metrics[x].price_change_7d < -5
                ):

                    trigger_one_hit = True
                    amount_of_trades += 1
                    trigger_one_trades += 1
                    trigger_price = metrics[x].last_price
                    stop_loss_price = trigger_price - (trigger_price * decimal.Decimal(0.02))
                    take_profit_price = trigger_price + (trigger_price * decimal.Decimal(0.02))
                    take_profit_hit = False
                    stop_loss_hit = False
                    take_profit_timestamp = None
                    stop_loss_timestamp = None
                    try:
                        for y in range(x, len(metrics)):
                            if (metrics[y].last_price >= take_profit_price):
                                take_profit_hit = True
                                take_profit_timestamp = metrics[y].timestamp
                                break

                            if (metrics[y].last_price <= stop_loss_price):
                                stop_loss_hit = True
                                stop_loss_timestamp = metrics[y].timestamp
                                break

                        if (take_profit_hit == True):
                            successful_trades += 1
                            trigger_one_success += 1
                        elif (stop_loss_hit == True):
                            failed_trades += 1
                        else:
                            amount_of_trades -= 1
                            trigger_one_trades -= 1

                    except:
                        print("failed in trigger 1")


                # TRIGGER 2 ----------------------------------------------------
                if (trigger_two_hit == True):
                    trigger_two_hit_counter += 1

                if (trigger_two_hit_counter > 13):
                    trigger_two_hit = False
                    trigger_two_hit_counter = 0

                if (
                    trigger_two_hit == False and
                    metrics[x].rolling_relative_volume > 330 and
                    metrics[x].price_change_5min > 0.2 and
                    metrics[x-1].price_change_5min > 0 and
                    metrics[x].price_change_10min > 0.1 and
                    metrics[x].price_change_24hr < -4
                ):

                    trigger_two_hit = True
                    amount_of_trades += 1
                    trigger_two_trades += 1
                    trigger_price = metrics[x].last_price
                    stop_loss_price = trigger_price - (trigger_price * decimal.Decimal(0.03))
                    take_profit_price = trigger_price + (trigger_price * decimal.Decimal(0.03))
                    take_profit_hit = False
                    stop_loss_hit = False
                    take_profit_timestamp = None
                    stop_loss_timestamp = None

                    try:
                        for y in range(x, len(metrics)):
                            if (metrics[y].last_price >= take_profit_price):
                                take_profit_hit = True
                                take_profit_timestamp = metrics[y].timestamp
                                break

                            if (metrics[y].last_price <= stop_loss_price):
                                stop_loss_hit = True
                                stop_loss_timestamp = metrics[y].timestamp
                                break

                        if (take_profit_hit == True):
                            successful_trades += 1
                            trigger_two_success += 1
                        elif (stop_loss_hit == True):
                            failed_trades += 1
                        else:
                            amount_of_trades -= 1
                            trigger_two_trades -= 1

                    except:
                        print("failed in trigger 2")


                # TRIGGER 3 ----------------------------------------------------

                if (trigger_three_hit == True):
                    trigger_three_hit_counter += 1

                if (trigger_three_hit_counter > 13):
                    trigger_three_hit = False
                    trigger_three_hit_counter = 0

                if (
                    trigger_three_hit == False and
                    metrics[x].rolling_relative_volume > 300 and
                    metrics[x].price_change_5min > 0.3 and
                    metrics[x-1].price_change_5min > 0 and
                    #metrics[x-2].price_change_5min > 0 and
                    metrics[x].price_change_10min > 0.1 and
                    metrics[x].price_change_24hr < -7 and
                    metrics[x].price_change_7d < -5 and
                    metrics[x].volume_24h > 1000000

                ):

                    trigger_three_hit = True
                    amount_of_trades += 1
                    trigger_three_trades += 1
                    trigger_price = metrics[x].last_price
                    stop_loss_price = trigger_price - (trigger_price * decimal.Decimal(0.03))
                    take_profit_price = trigger_price + (trigger_price * decimal.Decimal(0.03))
                    take_profit_hit = False
                    stop_loss_hit = False
                    take_profit_timestamp = None
                    stop_loss_timestamp = None

                    try:
                        for y in range(x, len(metrics)):
                            if (metrics[y].last_price >= take_profit_price):
                                take_profit_hit = True
                                take_profit_timestamp = metrics[y].timestamp
                                break

                            if (metrics[y].last_price <= stop_loss_price):
                                stop_loss_hit = True
                                stop_loss_timestamp = metrics[y].timestamp
                                break

                        if (take_profit_hit == True):
                            successful_trades += 1
                            trigger_three_success += 1
                        elif (stop_loss_hit == True):
                            failed_trades += 1
                        else:
                            amount_of_trades -= 1
                            trigger_three_trades -= 1

                    except:
                        print("failed in trigger 3")


                # SHORT Trigger ------------------------------------------------
                if (trigger_short_hit == True):
                    trigger_short_hit_counter += 1

                if (trigger_short_hit_counter > 13):
                    trigger_short_hit = False
                    trigger_short_hit_counter = 0

                # 0% success rate
                if (
                    trigger_short_hit == False and
                    metrics[x].rolling_relative_volume > 300 and
                    metrics[x].price_change_5min < -0.2 and
                    metrics[x-1].price_change_5min < -0.1 and
                    metrics[x].price_change_10min < -0.3 and
                    metrics[x].price_change_1hr > 0
                    #metrics[x].price_change_24hr > 2.1 and
                    #metrics[x].price_change_7d > 0
                ):

                    trigger_short_hit = True
                    amount_of_trades += 1
                    trigger_short_trades += 1
                    trigger_price = metrics[x].last_price
                    stop_loss_price = trigger_price + (trigger_price * decimal.Decimal(0.02))
                    take_profit_price = trigger_price - (trigger_price * decimal.Decimal(0.02))
                    take_profit_hit = False
                    stop_loss_hit = False
                    take_profit_timestamp = None
                    stop_loss_timestamp = None

                    try:
                        for y in range(x, len(metrics)):
                            if (metrics[y].last_price <= take_profit_price):
                                take_profit_hit = True
                                take_profit_timestamp = metrics[y].timestamp
                                break

                            if (metrics[y].last_price >= stop_loss_price):
                                stop_loss_hit = True
                                stop_loss_timestamp = metrics[y].timestamp
                                break

                        if (take_profit_hit == True):
                            successful_trades += 1
                            trigger_short_success += 1
                        elif (stop_loss_hit == True):
                            failed_trades += 1
                        else:
                            amount_of_trades -= 1
                            trigger_short_trades -= 1

                    except:
                        print("failed in short trigger")


                # TRIGGER 5 ----------------------------------------------------
                if (trigger_five_hit == True):
                    trigger_five_hit_counter += 1

                if (trigger_five_hit_counter > 13):
                    trigger_five_hit = False
                    trigger_five_hit_counter = 0

                if (
                    trigger_five_hit == False and
                    metrics[x].rolling_relative_volume > 350 and
                    metrics[x].price_change_10min > 0.3 and
                    metrics[x].price_change_24hr < -7 and
                    metrics[x].price_change_7d < -5
                ):

                    trigger_five_hit = True
                    amount_of_trades += 1
                    trigger_five_trades += 1
                    trigger_price = metrics[x].last_price
                    stop_loss_price = trigger_price - (trigger_price * decimal.Decimal(0.03))
                    take_profit_price = trigger_price + (trigger_price * decimal.Decimal(0.03))
                    take_profit_hit = False
                    stop_loss_hit = False
                    take_profit_timestamp = None
                    stop_loss_timestamp = None
                    success = False




                    timestamp = metrics[x].timestamp
                    if (timestamp.date() == datetime(2025, 2, 23).date()):
                        feb_23_count += 1
                    elif (timestamp.date() == datetime(2025, 2, 22).date()):
                        feb_22_count += 1
                    elif (timestamp.date() == datetime(2025, 2, 21).date()):
                        feb_21_count += 1
                    elif (timestamp.date() == datetime(2025, 2, 20).date()):
                        feb_20_count += 1
                    elif (timestamp.date() == datetime(2025, 2, 19).date()):
                        feb_19_count += 1
                    elif (timestamp.date() == datetime(2025, 2, 18).date()):
                        feb_18_count += 1
                    elif (timestamp.date() == datetime(2025, 2, 17).date()):
                        feb_17_count += 1
                    elif (timestamp.date() == datetime(2025, 2, 16).date()):
                        feb_16_count += 1
                    elif (timestamp.date() == datetime(2025, 2, 15).date()):
                        feb_15_count += 1
                    elif (timestamp.date() == datetime(2025, 2, 14).date()):
                        feb_14_count += 1
                    elif (timestamp.date() == datetime(2025, 2, 13).date()):
                        feb_13_count += 1
                    elif (timestamp.date() == datetime(2025, 2, 12).date()):
                        feb_12_count += 1
                    elif (timestamp.date() == datetime(2025, 2, 11).date()):
                        feb_11_count += 1
                    elif (timestamp.date() == datetime(2025, 2, 10).date()):
                        feb_10_count += 1
                    elif (timestamp.date() == datetime(2025, 2, 9).date()):
                        feb_9_count += 1
                    elif (timestamp.date() == datetime(2025, 2, 8).date()):
                        feb_8_count += 1
                    elif (timestamp.date() == datetime(2025, 2, 7).date()):
                        feb_7_count += 1
                    elif (timestamp.date() == datetime(2025, 2, 6).date()):
                        feb_6_count += 1
                    elif (timestamp.date() == datetime(2025, 2, 5).date()):
                        feb_5_count += 1
                    elif (timestamp.date() == datetime(2025, 2, 4).date()):
                        feb_4_count += 1
                    elif (timestamp.date() == datetime(2025, 2, 3).date()):
                        feb_3_count += 1
                    elif (timestamp.date() == datetime(2025, 2, 2).date()):
                        feb_2_count += 1
                    elif (timestamp.date() == datetime(2025, 2, 1).date()):
                        feb_1_count += 1
                    elif (timestamp.date() == datetime(2025, 1, 31).date()):
                        jan_31_count += 1
                    elif (timestamp.date() == datetime(2025, 1, 30).date()):
                        jan_30_count += 1
                    elif (timestamp.date() == datetime(2025, 1, 29).date()):
                        jan_29_count += 1
                    elif (timestamp.date() == datetime(2025, 1, 28).date()):
                        jan_28_count += 1
                    elif (timestamp.date() == datetime(2025, 1, 27).date()):
                        jan_27_count += 1
                    elif (timestamp.date() == datetime(2025, 1, 26).date()):
                        jan_26_count += 1
                    elif (timestamp.date() == datetime(2025, 1, 25).date()):
                        jan_25_count += 1
                    elif (timestamp.date() == datetime(2025, 1, 24).date()):
                        jan_24_count += 1
                    elif (timestamp.date() == datetime(2025, 1, 23).date()):
                        jan_23_count += 1
                    elif (timestamp.date() == datetime(2025, 1, 22).date()):
                        jan_22_count += 1
                    elif (timestamp.date() == datetime(2025, 1, 21).date()):
                        jan_21_count += 1
                    elif (timestamp.date() == datetime(2025, 1, 20).date()):
                        jan_20_count += 1
                    elif (timestamp.date() == datetime(2025, 1, 19).date()):
                        jan_19_count += 1
                    elif (timestamp.date() == datetime(2025, 1, 18).date()):
                        jan_18_count += 1
                    elif (timestamp.date() == datetime(2025, 1, 17).date()):
                        jan_17_count += 1
                    elif (timestamp.date() == datetime(2025, 1, 16).date()):
                        jan_16_count += 1
                    elif (timestamp.date() == datetime(2025, 1, 15).date()):
                        jan_15_count += 1
                    elif (timestamp.date() == datetime(2025, 1, 14).date()):
                        jan_14_count += 1
                    elif (timestamp.date() == datetime(2025, 1, 13).date()):
                        jan_13_count += 1




                    try:
                        for y in range(x, len(metrics)):
                            if (metrics[y].last_price >= take_profit_price):
                                take_profit_hit = True
                                take_profit_timestamp = metrics[y].timestamp
                                break

                            if (metrics[y].last_price <= stop_loss_price):
                                stop_loss_hit = True
                                stop_loss_timestamp = metrics[y].timestamp
                                break

                        if (take_profit_hit == True):
                            successful_trades += 1
                            trigger_five_success += 1
                        elif (stop_loss_hit == True):
                            failed_trades += 1
                        else:
                            amount_of_trades -= 1
                            trigger_five_trades -= 1

                    except:
                        print("failed in trigger 5")


                # TRIGGER SIX --------------------------------------------------
                if (trigger_six_hit == True):
                    trigger_six_hit_counter += 1

                if (trigger_six_hit_counter > 13):
                    trigger_six_hit = False
                    trigger_six_hit_counter = 0

                if (
                    trigger_six_hit == False and
                    metrics[x].rolling_relative_volume > 320 and
                    metrics[x].price_change_5min > 0 and
                    metrics[x-1].price_change_5min > 0 and
                    metrics[x].price_change_24hr < -7
                ):

                    trigger_six_hit = True
                    amount_of_trades += 1
                    trigger_six_trades += 1
                    trigger_price = metrics[x].last_price
                    stop_loss_price = trigger_price - (trigger_price * decimal.Decimal(0.03))
                    take_profit_price = trigger_price + (trigger_price * decimal.Decimal(0.03))
                    take_profit_hit = False
                    stop_loss_hit = False
                    take_profit_timestamp = None
                    stop_loss_timestamp = None

                    try:
                        for y in range(x, len(metrics)):
                            if (metrics[y].last_price >= take_profit_price):
                                take_profit_hit = True
                                take_profit_timestamp = metrics[y].timestamp
                                break

                            if (metrics[y].last_price <= stop_loss_price):
                                stop_loss_hit = True
                                stop_loss_timestamp = metrics[y].timestamp
                                break

                        if (take_profit_hit == True):
                            successful_trades += 1
                            trigger_six_success += 1
                        elif (stop_loss_hit == True):
                            failed_trades += 1
                        else:
                            amount_of_trades -= 1
                            trigger_six_trades -= 1

                    except:
                        print("failed in trigger 6")


                # TRIGGER SEVEN ------------------------------------------------
                if (trigger_seven_hit == True):
                    trigger_seven_hit_counter += 1

                if (trigger_seven_hit_counter > 13):
                    trigger_seven_hit = False
                    trigger_seven_hit_counter = 0

                if (
                    trigger_seven_hit == False and
                    metrics[x].rolling_relative_volume > 300 and
                    metrics[x].price_change_5min > 0.1 and
                    metrics[x-1].price_change_5min > 0.1 and
                    metrics[x].price_change_10min > 0 and
                    metrics[x].price_change_24hr < 0 and
                    metrics[x].price_change_7d < 0
                ):

                    trigger_seven_hit = True
                    amount_of_trades += 1
                    trigger_seven_trades += 1
                    trigger_price = metrics[x].last_price
                    stop_loss_price = trigger_price - (trigger_price * decimal.Decimal(0.02))
                    take_profit_price = trigger_price + (trigger_price * decimal.Decimal(0.02))
                    take_profit_hit = False
                    stop_loss_hit = False
                    take_profit_timestamp = None
                    stop_loss_timestamp = None
                    try:
                        for y in range(x, len(metrics)):
                            if (metrics[y].last_price >= take_profit_price):
                                take_profit_hit = True
                                take_profit_timestamp = metrics[y].timestamp
                                break

                            if (metrics[y].last_price <= stop_loss_price):
                                stop_loss_hit = True
                                stop_loss_timestamp = metrics[y].timestamp
                                break

                        if (take_profit_hit == True):
                            successful_trades += 1
                            trigger_seven_success += 1
                        elif (stop_loss_hit == True):
                            failed_trades += 1
                        else:
                            amount_of_trades -= 1
                            trigger_seven_trades -= 1

                    except:
                        print("failed in trigger 7")


    print("Results: ")
    print(f"Amount of trades: {amount_of_trades}")
    print(f"Successful trades: {successful_trades}")
    print(f"Failed trades: {failed_trades}")
    print(f"Trigger One: {trigger_one_trades}")
    print(f"Trigger Two: {trigger_two_trades}")
    print(f"Trigger Three: {trigger_three_trades}")
    print(f"Trigger Short: {trigger_short_trades}")
    print(f"Trigger Five: {trigger_five_trades}")
    print(f"Trigger Six: {trigger_six_trades}")
    print(f"Trigger Seven: {trigger_seven_trades}")

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

    trigger_six_success_percentage = 0
    if (trigger_six_trades != 0):
        trigger_six_success_percentage = (trigger_six_success / trigger_six_trades) * 100
    print(f"Trigger Six Success: {trigger_six_success_percentage}%")

    trigger_seven_success_percentage = 0
    if (trigger_seven_trades != 0):
        trigger_seven_success_percentage = (trigger_seven_success / trigger_seven_trades) * 100
    print(f"Trigger Seven Success: {trigger_seven_success_percentage}%")

    print(f"jan 13: {jan_13_count} trades")
    print(f"jan 14: {jan_14_count} trades")
    print(f"jan 15: {jan_15_count} trades")
    print(f"jan 16: {jan_16_count} trades")
    print(f"jan 17: {jan_17_count} trades")
    print(f"jan 18: {jan_18_count} trades")
    print(f"jan 19: {jan_19_count} trades")
    print(f"jan 20: {jan_20_count} trades")
    print(f"jan 21: {jan_21_count} trades")
    print(f"jan 22: {jan_22_count} trades")
    print(f"jan 23: {jan_23_count} trades")
    print(f"jan 24: {jan_24_count} trades")
    print(f"jan 25: {jan_25_count} trades")
    print(f"jan 26: {jan_26_count} trades")
    print(f"jan 27: {jan_27_count} trades")
    print(f"jan 28: {jan_28_count} trades")
    print(f"jan 29: {jan_29_count} trades")
    print(f"jan 30: {jan_30_count} trades")
    print(f"jan 31: {jan_31_count} trades")
    print(f"Feb 1: {feb_1_count} trades")
    print(f"Feb 2: {feb_2_count} trades")
    print(f"Feb 3: {feb_3_count} trades")
    print(f"Feb 4: {feb_4_count} trades")
    print(f"Feb 5: {feb_5_count} trades")
    print(f"Feb 6: {feb_6_count} trades")
    print(f"Feb 7: {feb_7_count} trades")
    print(f"Feb 8: {feb_8_count} trades")
    print(f"Feb 9: {feb_9_count} trades")
    print(f"Feb 10: {feb_10_count} trades")
    print(f"Feb 11: {feb_11_count} trades")
    print(f"Feb 12: {feb_12_count} trades")
    print(f"Feb 13: {feb_13_count} trades")
    print(f"Feb 14: {feb_14_count} trades")
    print(f"Feb 15: {feb_15_count} trades")
    print(f"Feb 16: {feb_16_count} trades")
    print(f"Feb 17: {feb_17_count} trades")
    print(f"Feb 18: {feb_18_count} trades")
    print(f"Feb 19: {feb_19_count} trades")
    print(f"Feb 20: {feb_20_count} trades")
    print(f"Feb 21: {feb_21_count} trades")
    print(f"Feb 22: {feb_22_count} trades")
    print(f"Feb 23: {feb_23_count} trades")


def brute_force_short():

    coins = Coin.objects.all()
    amount_of_trades = 0
    successful_trades = 0
    failed_trades = 0

    rolling_rvol_threshold = 1.7
    five_min_rvol_threshold = 1.1
    price_change_5min_threshold = 0.2
    #price_change_10min_threshold = 0.8
    price_change_10min_threshold = 0
    price_change_1hr_threshold = -0.6
    #price_change_24hr_threshold = 1.5
    price_change_24hr_threshold = 0.1
    price_change_7d_threshold = 6.4

    top_percentage = 0
    top_rolling_rvol = 0
    top_five_min_rvol = 0
    top_price_change_5min = 0
    top_price_change_10min = 0
    top_price_change_1hr = 0
    top_price_change_24hr = 0
    top_price_change_7d = 0

    #rolling_rvol_threshold = 1.5
    #five_min_rvol_threshold = 1.1
    #price_change_5min_threshold = 0.8
    #price_change_10min_threshold = 1.8
    #price_change_1hr_threshold = -0.3
    #price_change_24hr_threshold = -10.3
    #price_change_7d_threshold = 0


    for a in range(100, -50, -2):
        value_a = a / 10

        price_change_7d_threshold = value_a

        amount_of_trades = 0
        successful_trades = 0
        failed_trades = 0

        for coin in coins:

            metrics = Metrics.objects.filter(coin=coin).order_by('timestamp')
            trigger_one_hit_counter = 0
            trigger_one_hit = False

            for x in range(6, len(metrics)):

                if (metrics[x].rolling_relative_volume != None and
                    metrics[x].price_change_5min != None and
                    metrics[x].price_change_10min != None and
                    metrics[x].price_change_1hr != None and
                    metrics[x].price_change_24hr != None and
                    metrics[x].five_min_relative_volume != None and
                    metrics[x].twenty_min_relative_volume != None):

                    # TRIGGER 1 ----------------------------------------------------
                    if (trigger_one_hit == True):
                        trigger_one_hit_counter += 1

                    if (trigger_one_hit_counter > 13):
                        trigger_one_hit = False
                        trigger_one_hit_counter = 0

                    if (
                        trigger_one_hit == False and
                        metrics[x].rolling_relative_volume >= rolling_rvol_threshold and
                        metrics[x].five_min_relative_volume >= five_min_rvol_threshold and
                        metrics[x].price_change_5min <= price_change_5min_threshold and
                        metrics[x].price_change_10min >= price_change_10min_threshold and
                        metrics[x].price_change_1hr <= price_change_1hr_threshold and
                        metrics[x].price_change_24hr >= price_change_24hr_threshold and
                        metrics[x].price_change_7d <= price_change_7d_threshold
                    ):


                        trigger_one_hit = True

                        amount_of_trades += 1

                        trigger_price = metrics[x].last_price
                        stop_loss_price = trigger_price + (trigger_price * decimal.Decimal(0.02))
                        take_profit_price = trigger_price - (trigger_price * decimal.Decimal(0.06))

                        # try to go through remaining metrics
                        take_profit_hit = False
                        stop_loss_hit = False
                        take_profit_timestamp = None
                        stop_loss_timestamp = None
                        try:
                            for y in range(x, len(metrics)):
                                if (metrics[y].last_price <= take_profit_price):
                                    take_profit_hit = True
                                    take_profit_timestamp = metrics[y].timestamp
                                    break

                                if (metrics[y].last_price >= stop_loss_price):
                                    stop_loss_hit = True
                                    stop_loss_timestamp = metrics[y].timestamp
                                    break

                            if (take_profit_hit == True):
                                successful_trades += 1
                            elif (stop_loss_hit == True):
                                failed_trades += 1
                            else:
                                amount_of_trades -= 1

                        except:
                            print("failed in trigger 1")


        # check success rate
        success_percentage = 0
        if (amount_of_trades != 0):
            success_percentage = (successful_trades / amount_of_trades) * 100

        if (amount_of_trades > 50 and success_percentage > top_percentage):
            top_percentage = success_percentage
            top_rolling_rvol = rolling_rvol_threshold
            top_five_min_rvol = five_min_rvol_threshold
            top_price_change_5min = price_change_5min_threshold
            top_price_change_10min = price_change_10min_threshold
            top_price_change_1hr = price_change_1hr_threshold
            top_price_change_24hr = price_change_24hr_threshold
            top_price_change_7d = price_change_7d_threshold

            print("Current Results:")
            print(f"top_percentage: {top_percentage}")
            print(f"amount of trades: {amount_of_trades}")
            print(f"top_rolling_rvol: {top_rolling_rvol}")
            print(f"top_five_min_rvol: {top_five_min_rvol}")
            print(f"top_price_change_5min: {top_price_change_5min}")
            print(f"top_price_change_10min: {top_price_change_10min}")
            print(f"top_price_change_1hr: {top_price_change_1hr}")
            print(f"top_price_change_24hr: {top_price_change_24hr}")
            print(f"top_price_change_7d: {top_price_change_7d}")

        else:
            print("not better yet")
            print(f"amount of trades: {amount_of_trades}")
            print(f"success rate: {success_percentage}%")



    print("Final Results:")
    print(f"top_percentage: {top_percentage}")
    print(f"amount of trades: {amount_of_trades}")
    print(f"top_rolling_rvol: {top_rolling_rvol}")
    print(f"top_five_min_rvol: {top_five_min_rvol}")
    print(f"top_price_change_5min: {top_price_change_5min}")
    print(f"top_price_change_10min: {top_price_change_10min}")
    print(f"top_price_change_1hr: {top_price_change_1hr}")
    print(f"top_price_change_24hr: {top_price_change_24hr}")
    print(f"top_price_change_7d: {top_price_change_7d}")


# used to find all possible entry points for a successful trade to view that data
def find_best_trigger():

    #coins = Coin.objects.all()

    coin = Coin.objects.get(symbol = "DOT")

    results = []

    amount_of_trades = 0
    successful_trades = 0
    failed_trades = 0
    trigger_one_trades = 0
    trigger_one_success = 0

    metrics = Metrics.objects.filter(coin=coin).order_by('timestamp')

    trigger_one_hit_counter = 0
    trigger_one_hit = False

    for x in range(6, len(metrics)):

        trigger_price = metrics[x].last_price
        stop_loss_price = trigger_price - (trigger_price * decimal.Decimal(0.02))
        take_profit_price = trigger_price + (trigger_price * decimal.Decimal(0.05))

        take_profit_hit = False
        stop_loss_hit = False

        try:
            for y in range(x, len(metrics)):
                if (metrics[y].last_price >= take_profit_price):
                    take_profit_hit = True
                    break

                if (metrics[y].last_price <= stop_loss_price):
                    stop_loss_hit = True
                    break

            if (take_profit_hit == True):

                # successful entry point
                dict = {}
                dict[0] = coin.symbol
                dict[1] = metrics[x].timestamp
                dict[3] = metrics[x].rolling_relative_volume
                dict[4] = metrics[x].five_min_relative_volume
                dict[5] = metrics[x].twenty_min_relative_volume
                dict[6] = metrics[x].price_change_5min
                dict[7] = metrics[x].price_change_10min
                dict[8] = metrics[x].price_change_1hr
                dict[9] = metrics[x].price_change_24hr
                dict[10] = metrics[x].price_change_7d

                results.append(dict)


        except Exception as e:
            print(f"Error: {e}")


    # results has all the metrics
    for success in results:
        print(f"timestamp: {success[1]}")
        print(f"rolling_relative_volume: {success[3]}")
        print(f"five_min_relative_volume: {success[4]}")
        print(f"twenty_min_relative_volume: {success[5]}")
        print(f"price_change_5min: {success[6]}")
        print(f"price_change_10min: {success[7]}")
        print(f"price_change_1hr: {success[8]}")
        print(f"price_change_24hr: {success[9]}")
        print(f"price_change_7d: {success[10]}")


# brute force try - one metric at a time
def brute_force_one():

    # 1. create a trigger combination
    # 2. run it against db and get success rate
    # 3. if the trigger didn't go off more than 50 times, disregard
    # 4. if success rate is higher than previous, save it

    coins = Coin.objects.all()
    amount_of_trades = 0
    successful_trades = 0
    failed_trades = 0

    rolling_rvol_threshold = 0.9
    five_min_rvol_threshold = -0.9
    price_change_5min_threshold = 1.7 # if we do price is <= this value
    #price_change_10min_threshold = 3.3 # if we do price <= this value
    price_change_10min_threshold = 3.0 # if we do price >= this value
    price_change_1hr_threshold = 3.0
    price_change_24hr_threshold = 0
    price_change_7d_threshold = 0.5 # if price is >= this value
    price_change_7d_threshold = 0 # if price is <= this value

    top_percentage = 0
    top_rolling_rvol = 0
    top_five_min_rvol = 0
    top_price_change_5min = 0
    top_price_change_10min = 0
    top_price_change_1hr = 0
    top_price_change_24hr = 0
    top_price_change_7d = 0

    for a in range(-100, 101, 5):
        value_a = a / 10

        #rolling_rvol_threshold = value_a
        #five_min_rvol_threshold = value_a
        #price_change_5min_threshold = value_a
        #price_change_10min_threshold = value_a
        #price_change_1hr_threshold = value_a
        #price_change_24hr_threshold = value_a
        price_change_7d_threshold = value_a

        amount_of_trades = 0
        successful_trades = 0
        failed_trades = 0

        for coin in coins:

            has_metrics = Metrics.objects.filter(coin=coin).exists()

            if has_metrics == False:
                continue

            metrics = Metrics.objects.filter(coin=coin).order_by('timestamp')

            trigger_one_hit_counter = 0
            trigger_one_hit = False

            for x in range(6, len(metrics)):

                if (metrics[x].rolling_relative_volume != None and
                    metrics[x].price_change_5min != None and
                    metrics[x].price_change_10min != None and
                    metrics[x].price_change_1hr != None and
                    metrics[x].price_change_24hr != None and
                    metrics[x].price_change_7d != None and
                    metrics[x].five_min_relative_volume != None and
                    metrics[x].twenty_min_relative_volume != None):

                    # TRIGGER 1 ----------------------------------------------------
                    if (trigger_one_hit == True):
                        trigger_one_hit_counter += 1

                    if (trigger_one_hit_counter > 13):
                        trigger_one_hit = False
                        trigger_one_hit_counter = 0

                    if (
                        trigger_one_hit == False and
                        #metrics[x].rolling_relative_volume >= rolling_rvol_threshold and
                        #metrics[x].five_min_relative_volume >= five_min_rvol_threshold and
                        #metrics[x].price_change_5min <= price_change_5min_threshold and
                        #metrics[x].price_change_10min >= price_change_10min_threshold and
                        #metrics[x].price_change_1hr >= price_change_1hr_threshold and
                        #metrics[x].price_change_24hr >= price_change_24hr_threshold and
                        metrics[x].price_change_7d >= price_change_7d_threshold
                    ):


                        trigger_one_hit = True

                        amount_of_trades += 1

                        trigger_price = metrics[x].last_price
                        stop_loss_price = trigger_price - (trigger_price * decimal.Decimal(0.02))
                        take_profit_price = trigger_price + (trigger_price * decimal.Decimal(0.04))

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
                                    break

                                if (metrics[y].last_price <= stop_loss_price):
                                    stop_loss_hit = True
                                    stop_loss_timestamp = metrics[y].timestamp
                                    break

                            if (take_profit_hit == True):
                                successful_trades += 1
                            elif (stop_loss_hit == True):
                                failed_trades += 1
                            else:
                                amount_of_trades -= 1

                        except:
                            print("failed in trigger 1")


        # check success rate
        success_percentage = 0
        if (amount_of_trades != 0):
            success_percentage = (successful_trades / amount_of_trades) * 100

        if (amount_of_trades > 100 and success_percentage > top_percentage):
            top_percentage = success_percentage
            top_rolling_rvol = rolling_rvol_threshold
            top_five_min_rvol = five_min_rvol_threshold
            top_price_change_5min = price_change_5min_threshold
            top_price_change_10min = price_change_10min_threshold
            top_price_change_1hr = price_change_1hr_threshold
            top_price_change_24hr = price_change_24hr_threshold
            top_price_change_7d = price_change_7d_threshold

            print("Current Results:")
            print(f"amount_of_trades: {amount_of_trades}")
            print(f"top_percentage: {top_percentage}")
            print(f"top_rolling_rvol: {top_rolling_rvol}")
            print(f"top_five_min_rvol: {top_five_min_rvol}")
            print(f"top_price_change_5min: {top_price_change_5min}")
            print(f"top_price_change_10min: {top_price_change_10min}")
            print(f"top_price_change_1hr: {top_price_change_1hr}")
            print(f"top_price_change_24hr: {top_price_change_24hr}")
            print(f"top_price_change_7d: {top_price_change_7d}")

        else:
            print("not better yet")
            print(f"amount of trades: {amount_of_trades}")
            print(f"success rate: {success_percentage}%")



    print("Final Results:")
    print(f"top_percentage: {top_percentage}")
    print(f"top_rolling_rvol: {top_rolling_rvol}")
    print(f"top_five_min_rvol: {top_five_min_rvol}")
    print(f"top_price_change_5min: {top_price_change_5min}")
    print(f"top_price_change_10min: {top_price_change_10min}")
    print(f"top_price_change_1hr: {top_price_change_1hr}")
    print(f"top_price_change_24hr: {top_price_change_24hr}")
    print(f"top_price_change_7d: {top_price_change_7d}")


# brute force try - looping multiple metrics
def brute_force():

    # brute force try on every single trigger combination and save best results

    # 1. create a trigger combination
    # 2. run it against db and get success rate
    # 3. if the trigger didn't go off more than 30 times, disregard
    # 4. if success rate is higher than previous, save it

    #coins = Coin.objects.all()

    coin = Coin.objects.get(symbol="XRP")

    coins = [coin]

    amount_of_trades = 0
    successful_trades = 0
    failed_trades = 0

    rolling_rvol_threshold = 1.4
    five_min_rvol_threshold = 1.0
    price_change_5min_threshold = 0.2
    price_change_10min_threshold = 0.1
    price_change_1hr_threshold = 0
    price_change_24hr_threshold = -0.5
    price_change_7d_threshold = 0.5

    top_percentage = 0
    top_rolling_rvol = 0
    top_five_min_rvol = 0
    top_price_change_5min = 0
    top_price_change_10min = 0
    top_price_change_1hr = 0
    top_price_change_24hr = 0
    top_price_change_7d = 0

    rolling_rvol_threshold = 0
    five_min_rvol_threshold = 0
    price_change_5min_threshold = 0
    price_change_10min_threshold = 0
    price_change_1hr_threshold = 0
    price_change_24hr_threshold = 0
    price_change_7d_threshold = 0

    start = 5
    finish = 25
    step = 1

    for a in range(start, finish, step): # 20 steps

        value_a = a / 10
        rolling_rvol_threshold = value_a

        for c in range(-10, 15, 1): # 25 steps

            value_c = c / 10
            price_change_5min_threshold = value_c

            for e in range(10, 25, 1):

                value_e = e / 10
                price_change_1hr_threshold = value_e
                #price_change_1hr_threshold = 1.9

                amount_of_trades = 0
                successful_trades = 0
                failed_trades = 0

                for coin in coins:

                    metrics = Metrics.objects.filter(coin=coin).order_by('timestamp')
                    trigger_one_hit_counter = 0
                    trigger_one_hit = False

                    for x in range(6, len(metrics)):

                        day = metrics[x].timestamp.day

                        if (metrics[x].rolling_relative_volume != None and
                            metrics[x].price_change_5min != None and
                            metrics[x].price_change_10min != None and
                            metrics[x].price_change_1hr != None and
                            metrics[x].price_change_24hr != None and
                            metrics[x].five_min_relative_volume != None and
                            metrics[x].twenty_min_relative_volume != None):

                            # TRIGGER 1 ----------------------------------------------------
                            if (trigger_one_hit == True):
                                trigger_one_hit_counter += 1

                            if (trigger_one_hit_counter > 13):
                                trigger_one_hit = False
                                trigger_one_hit_counter = 0

                            if (
                                trigger_one_hit == False and
                                metrics[x].rolling_relative_volume > rolling_rvol_threshold and
                                #metrics[x].five_min_relative_volume > five_min_rvol_threshold and
                                metrics[x].price_change_5min > price_change_5min_threshold and
                                #metrics[x].price_change_10min < price_change_10min_threshold and
                                metrics[x].price_change_1hr > price_change_1hr_threshold
                                #metrics[x].price_change_24hr < price_change_24hr_threshold and
                                #metrics[x].price_change_7d < price_change_7d_threshold
                            ):

                                trigger_one_hit = True
                                amount_of_trades += 1

                                trigger_price = metrics[x].last_price
                                stop_loss_price = trigger_price - (trigger_price * decimal.Decimal(0.02))
                                take_profit_price = trigger_price + (trigger_price * decimal.Decimal(0.04))

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
                                            break

                                        if (metrics[y].last_price <= stop_loss_price):
                                            stop_loss_hit = True
                                            stop_loss_timestamp = metrics[y].timestamp
                                            break

                                    if (take_profit_hit == True):
                                        successful_trades += 1
                                    elif (stop_loss_hit == True):
                                        failed_trades += 1
                                    else:
                                        amount_of_trades -= 1

                                except:
                                    print("failed in trigger 1")


                # check success rate
                success_percentage = 0
                if (amount_of_trades != 0):
                    success_percentage = (successful_trades / amount_of_trades) * 100

                if (amount_of_trades >= 25 and success_percentage > top_percentage):
                    top_percentage = success_percentage
                    top_rolling_rvol = rolling_rvol_threshold
                    #top_five_min_rvol = five_min_rvol_threshold
                    top_price_change_5min = price_change_5min_threshold
                    #top_price_change_10min = price_change_10min_threshold
                    top_price_change_1hr = price_change_1hr_threshold
                    #top_price_change_24hr = price_change_24hr_threshold
                    #top_price_change_7d = price_change_7d_threshold

                    print("Current Results:")
                    print(f"top_percentage: {top_percentage}")
                    print(f"amount of trades: {amount_of_trades}")
                    print(f"top_rolling_rvol: {top_rolling_rvol}")
                    #print(f"top_five_min_rvol: {top_five_min_rvol}")
                    print(f"top_price_change_5min: {top_price_change_5min}")
                    #print(f"top_price_change_10min: {top_price_change_10min}")
                    print(f"top_price_change_1hr: {top_price_change_1hr}")
                    #print(f"top_price_change_24hr: {top_price_change_24hr}")
                    #print(f"top_price_change_7d: {top_price_change_7d}")

                else:
                    print("not better yet") # 15 steps
                    #print(f"amount of trades: {amount_of_trades}")
                    #print(f"success rate: {success_percentage}%")

            print("inner loop finished")

        print("middle loop finished")

    print("outer loop finished")


    message = []
    one = "Results:"
    two = f"top_percentage: {top_percentage}"
    three = f"amount of trades: {amount_of_trades}"
    four = f"top_rolling_rvol: {top_rolling_rvol}"
    five = f"top_price_change_5min: {top_price_change_5min}"
    six = f"top_price_change_1hr: {top_price_change_1hr}"
    message.append(one)
    message.append(two)
    message.append(three)
    message.append(four)
    message.append(five)
    message.append(six)
    send_text(message)

    print("Final Results:")
    print(f"top_percentage: {top_percentage}")
    print(f"amount of trades: {amount_of_trades}")
    print(f"top_rolling_rvol: {top_rolling_rvol}")
    #print(f"top_five_min_rvol: {top_five_min_rvol}")
    print(f"top_price_change_5min: {top_price_change_5min}")
    #print(f"top_price_change_10min: {top_price_change_10min}")
    print(f"top_price_change_1hr: {top_price_change_1hr}")
    #print(f"top_price_change_24hr: {top_price_change_24hr}")
    #print(f"top_price_change_7d: {top_price_change_7d}")


# go through coins and find best trigger combination for each and store it
def trigger_combination():

    coins = Coin.objects.all()

    for coin in coins:

        if not coin.symbol == "XRP":
            continue

        results = []

        metrics = Metrics.objects.filter(coin=coin).order_by('timestamp')

        print(coin.symbol)
        print(len(metrics))

        if not metrics:
            continue

        amount_of_trades = 0
        successful_trades = 0
        failed_trades = 0
        trigger_one_trades = 0
        trigger_one_success = 0
        trigger_one_hit_counter = 0
        trigger_one_hit = False

        for x in range(6, len(metrics)):

            trigger_price = metrics[x].last_price
            stop_loss_price = trigger_price - (trigger_price * decimal.Decimal(0.02))
            take_profit_price = trigger_price + (trigger_price * decimal.Decimal(0.05))

            take_profit_hit = False
            stop_loss_hit = False

            try:
                for y in range(x, len(metrics)):
                    if (metrics[y].last_price >= take_profit_price):
                        take_profit_hit = True
                        break

                    if (metrics[y].last_price <= stop_loss_price):
                        stop_loss_hit = True
                        break

                if (
                    take_profit_hit == True and
                    metrics[x].rolling_relative_volume != None and
                    metrics[x].five_min_relative_volume != None and
                    metrics[x].twenty_min_relative_volume != None and
                    metrics[x].price_change_5min != None and
                    metrics[x].price_change_10min != None and
                    metrics[x].price_change_1hr != None and
                    metrics[x].price_change_24hr != None and
                    metrics[x].price_change_7d != None
                ):

                    # successful entry point
                    dict = {}
                    dict[0] = coin.symbol
                    dict[1] = metrics[x].timestamp
                    dict[3] = metrics[x].rolling_relative_volume
                    dict[4] = metrics[x].five_min_relative_volume
                    dict[5] = metrics[x].twenty_min_relative_volume
                    dict[6] = metrics[x].price_change_5min
                    dict[7] = metrics[x].price_change_10min
                    dict[8] = metrics[x].price_change_1hr
                    dict[9] = metrics[x].price_change_24hr
                    dict[10] = metrics[x].price_change_7d

                    results.append(dict)


            except Exception as e:
                print(f"Error: {e}")

        average_rolling_relative_volume = 0
        average_five_min_relative_volume = 0
        average_twenty_min_relative_volume = 0
        average_price_change_5min = 0
        average_price_change_10min = 0
        average_price_change_1hr = 0
        average_price_change_24hr = 0
        average_price_change_7d = 0

        for combo in results:

            average_rolling_relative_volume += abs(combo[3])
            average_five_min_relative_volume += abs(combo[4])
            average_twenty_min_relative_volume += abs(combo[5])
            average_price_change_5min += combo[6]
            average_price_change_10min += combo[7]
            average_price_change_1hr += combo[8]
            average_price_change_24hr += combo[9]
            average_price_change_7d += combo[10]


        print(f"Average successful metrics for {coin.symbol}:")

        for combo in results:
            print(f"rolling_relative_volume: {combo[3]}")
        for combo in results:
            print(f"five_min_relative_volume: {combo[4]}")
        for combo in results:
            print(f"twenty_min_relative_volume: {combo[5]}")
        for combo in results:
            print(f"price_change_5min: {combo[6]}")
        for combo in results:
            print(f"price_change_10min: {combo[7]}")
        for combo in results:
            print(f"price_change_1hr: {combo[8]}")
        for combo in results:
            print(f"price_change_24hr: {combo[9]}")
        for combo in results:
            print(f"price_change_7d: {combo[10]}")



        average_rolling_relative_volume /= len(results)
        average_five_min_relative_volume /= len(results)
        average_twenty_min_relative_volume /= len(results)
        average_price_change_5min /= len(results)
        average_price_change_10min /= len(results)
        average_price_change_1hr /= len(results)
        average_price_change_24hr /= len(results)
        average_price_change_7d /= len(results)

        '''
        Metrics.objects.create(
            coin = coin,
            timestamp = datetime.utcnow(),
            #daily_relative_volume = daily_relative_volume,
            rolling_relative_volume = average_rolling_relative_volume,
            five_min_relative_volume = average_five_min_relative_volume,
            twenty_min_relative_volume = average_twenty_min_relative_volume,
            price_change_5min = average_price_change_5min,
            price_change_10min = average_price_change_10min,
            price_change_1hr = average_price_change_1hr,
            price_change_24hr = average_price_change_24hr,
            price_change_7d = average_price_change_7d,
            circulating_supply = 1234,
        )
        '''

        print(f"Average successful metrics for {coin.symbol}:")
        print(f"rolling_relative_volume: {average_rolling_relative_volume}")
        print(f"five_min_relative_volume: {average_five_min_relative_volume}")
        print(f"twenty_min_relative_volume: {average_twenty_min_relative_volume}")
        print(f"price_change_5min: {average_price_change_5min}")
        print(f"price_change_10min: {average_price_change_10min}")
        print(f"price_change_1hr: {average_price_change_1hr}")
        print(f"price_change_24hr: {average_price_change_24hr}")
        print(f"price_change_7d: {average_price_change_7d}")




# ------------------------------------------------------------------------------



# I DONT KNOW ------------------------------------------------------------------

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
                "Timestamp", "Rolling RVOL", "5-Min RVOL",
                "20-Min RVOL", "5-Min Price Change", "10-Min Price Change",
                "1-Hour Price Change", "24-Hour Price Change", "7-Day Price Change",
                "Circulating Supply", "24-Hour Volume", "Last Price", "Market Cap"
            ])

        #with open('output.txt', 'w') as f:
            for metric in metrics:
                csvwriter.writerow([
                    metric.timestamp,
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






#
