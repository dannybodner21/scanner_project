import os
import json
import requests
import numpy as np
import pandas as pd
import ta
import xgboost as xgb
import time
import base64
import hmac
import hashlib
from datetime import datetime
from scanner.models import Coin, ModelTrade, RealTrade
from scipy.stats import linregress
from django.utils.timezone import now


KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET")
KRAKEN_DERIVATIVES_URL = "https://futures.kraken.com/derivatives/api/v3/sendorder"

KRAKEN_SYMBOL_MAP = {
    "BTC": "PF_XBTUSD",   # Bitcoin
    "ETH": "PF_ETHUSD",   # Ethereum
    "XRP": "PF_XRPUSD",   # Ripple
    "LTC": "PF_LTCUSD",   # Litecoin
    "SOL": "PF_SOLUSD",   # Solana
    "DOGE": "PF_DOGEUSD", # Dogecoin
    "LINK": "PF_LINKUSD", # Chainlink
    "DOT": "PF_DOTUSD",   # Polkadot
    "SHIB": "PF_SHIBUSD", # Shiba Inu
    "ADA": "PF_ADAUSD",   # Cardano
}

COINAPI_SYMBOL_MAP = {
    "BTCUSDT": "BINANCE_SPOT_BTC_USDT",
    "ETHUSDT": "BINANCE_SPOT_ETH_USDT",
    "XRPUSDT": "BINANCE_SPOT_XRP_USDT",
    "LTCUSDT": "BINANCE_SPOT_LTC_USDT",
    "SOLUSDT": "BINANCE_SPOT_SOL_USDT",
    "DOGEUSDT": "BINANCE_SPOT_DOGE_USDT",
    "LINKUSDT": "BINANCE_SPOT_LINK_USDT",
    "DOTUSDT": "BINANCE_SPOT_DOT_USDT",
    "SHIBUSDT": "BINANCE_SPOT_SHIB_USDT",
    "ADAUSDT": "BINANCE_SPOT_ADA_USDT",
}

COIN_SYMBOL_MAP_DB = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "XRPUSDT": "XRP",
    "LTCUSDT": "LTC",
    "SOLUSDT": "SOL",
    "DOGEUSDT": "DOGE",
    "LINKUSDT": "LINK",
    "DOTUSDT": "DOT",
    "SHIBUSDT": "SHIB",
    "ADAUSDT": "ADA",
}

COINS = list(COIN_SYMBOL_MAP_DB.keys())

COINAPI_KEY = "01293e2a-dcf1-4e81-8310-c6aa9d0cb743"
BASE_URL = "https://rest.coinapi.io/v1/ohlcv"

FEATURES = [
    'returns_5m', 'returns_15m', 'returns_1h', 'returns_4h', 'momentum',
    'volume_ma_20', 'vol_spike', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_lower', 'atr_14', 'adx_14', 'obv', 'obv_slope',
    'ema_9', 'ema_21', 'ema_diff', 'volatility', 'ma_200',
    'bull_regime', 'bear_regime', 'sideways_regime',
    'slope_1h', 'dist_from_high_24h', 'dist_from_low_24h',
    'stoch_k', 'stoch_d', 'price_change_5', 'volume_change_5',
    'high_1h', 'low_1h', 'pos_in_range_1h', 'vwap_1h', 'pos_vs_vwap'
]

INPUT_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'returns_5m', 'returns_15m', 'returns_1h', 'returns_4h', 'momentum',
    'volume_ma_20', 'vol_spike', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_lower', 'atr_14', 'adx_14', 'obv', 'obv_slope',
    'ema_9', 'ema_21', 'ema_diff', 'volatility', 'ma_200',
    'bull_regime', 'bear_regime', 'sideways_regime',
    'slope_1h', 'dist_from_high_24h', 'dist_from_low_24h',
    'stoch_k', 'stoch_d', 'price_change_5', 'volume_change_5',
    'high_1h', 'low_1h', 'pos_in_range_1h', 'vwap_1h', 'pos_vs_vwap'
]

CONFIDENCE_THRESHOLD = 0.5

def send_text(messages):

    if len(messages) > 0:

        chat_id_danny = '1077594551'
        chat_ids = [chat_id_danny]
        bot_token = '7672687080:AAFWvkwzp-LQE92XdO9vcVa5yWJDUxO17yE'
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

        message = ""
        for chat_id in chat_ids:
            for trigger in messages:

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

def calculate_trend_slope(prices):
    if len(prices) < 12:
        return np.nan
    x = np.arange(len(prices))
    slope, _, _, _, _ = linregress(x, prices)
    return slope

def fetch_ohlcv(coin, limit=300):
    coinapi_symbol = COINAPI_SYMBOL_MAP.get(coin)
    if not coinapi_symbol:
        raise ValueError(f"CoinAPI symbol mapping not found for {coin}")

    url = f"{BASE_URL}/{coinapi_symbol}/latest?period_id=5MIN&limit={limit}"
    headers = {"X-CoinAPI-Key": COINAPI_KEY}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame([{
        'timestamp': pd.to_datetime(candle['time_period_start']),
        'open': candle['price_open'],
        'high': candle['price_high'],
        'low': candle['price_low'],
        'close': candle['price_close'],
        'volume': candle['volume_traded']
    } for candle in data])

    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def add_features(df):
    df['returns_5m'] = df['close'].pct_change(1)
    df['returns_15m'] = df['close'].pct_change(3)
    df['returns_1h'] = df['close'].pct_change(12)
    df['returns_4h'] = df['close'].pct_change(48)
    df['momentum'] = df['close'] - df['close'].shift(5)

    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['vol_spike'] = df['volume'] / df['volume_ma_20']

    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    bollinger = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_lower'] = bollinger.bollinger_lband()

    df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['adx_14'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)

    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['obv_slope'] = df['obv'].diff()

    df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
    df['ema_diff'] = df['ema_9'] - df['ema_21']

    df['volatility'] = df['close'].rolling(20).std()
    df['ma_200'] = ta.trend.sma_indicator(df['close'], window=200)

    df['bull_regime'] = ((df['adx_14'] > 25) & (df['close'] > df['ma_200'])).astype(int)
    df['bear_regime'] = ((df['adx_14'] > 25) & (df['close'] < df['ma_200'])).astype(int)
    df['sideways_regime'] = (df['adx_14'] <= 25).astype(int)

    df['slope_1h'] = df['close'].rolling(12).apply(calculate_trend_slope, raw=False)
    df['dist_from_high_24h'] = (df['close'] - df['high'].rolling(288).max()) / df['high'].rolling(288).max()
    df['dist_from_low_24h'] = (df['close'] - df['low'].rolling(288).min()) / df['low'].rolling(288).min()

    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    df['price_change_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['volume_change_5'] = (df['volume'] - df['volume'].shift(5)) / df['volume'].shift(5)

    df['high_1h'] = df['high'].rolling(12).max()
    df['low_1h'] = df['low'].rolling(12).min()
    df['pos_in_range_1h'] = (df['close'] - df['low_1h']) / (df['high_1h'] - df['low_1h'])
    df['vwap_1h'] = (df['close'] * df['volume']).rolling(12).sum() / df['volume'].rolling(12).sum()
    df['pos_vs_vwap'] = df['close'] - df['vwap_1h']

    df = df.dropna()
    return df

def prepare_instance(df):
    row = df.iloc[-1]
    instance = {col: row[col] for col in INPUT_COLUMNS}
    return instance, row

def print_feature_stats(df, coin):
    print(f"--- Feature stats for {coin} ---")
    for feature in FEATURES:
        if feature in df.columns:
            series = df[feature]
            print(f"{feature} stats:")
            print(f"  count: {series.count()}")
            print(f"  mean: {series.mean()}")
            print(f"  std: {series.std()}")
            print(f"  min: {series.min()}")
            print(f"  25%: {series.quantile(0.25)}")
            print(f"  50%: {series.median()}")
            print(f"  75%: {series.quantile(0.75)}")
            print(f"  max: {series.max()}")
        else:
            print(f"⚠ Feature {feature} missing in dataframe for {coin}")

def get_kraken_signature(uri_path, data, nonce):
    post_data = f"nonce={nonce}"
    encoded = (str(nonce) + post_data).encode()
    message = uri_path.encode() + hashlib.sha256(encoded).digest()
    mac = hmac.new(base64.b64decode(KRAKEN_API_SECRET), message, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()

def place_kraken_order(uri_path, payload):
    nonce = str(int(time.time() * 1000))
    payload["nonce"] = nonce
    headers = {
        "API-Key": KRAKEN_API_KEY,
        "API-Sign": get_kraken_signature(uri_path, payload, nonce),
    }
    response = requests.post(f"https://api.kraken.com{uri_path}", headers=headers, data=payload, timeout=10)
    response.raise_for_status()
    return response.json()

def get_kraken_balance():
    nonce = str(int(time.time() * 1000))
    data = {"nonce": nonce}
    post_data = "&".join([f"{key}={value}" for key, value in data.items()])
    encoded = (nonce + post_data).encode()
    message = b'/0/private/Balance' + hashlib.sha256(encoded).digest()

    signature = hmac.new(base64.b64decode(KRAKEN_API_SECRET), message, hashlib.sha512)
    signature_b64 = base64.b64encode(signature.digest()).decode()

    headers = {
        "API-Key": KRAKEN_API_KEY,
        "API-Sign": signature_b64,
    }

    response = requests.post("https://api.kraken.com/0/private/Balance", headers=headers, data=data, timeout=10)
    response.raise_for_status()
    data = response.json()
    return float(data['result'].get('ZUSD', 0.0))


def get_kraken_futures_signature(private_key, data, nonce, path):
    message = hashlib.sha256((data + nonce + path.removeprefix("/derivatives")).encode()).digest()
    return base64.b64encode(hmac.new(base64.b64decode(private_key), msg=message, digestmod=hashlib.sha512).digest()).decode()


def place_kraken_futures_order(body):
    nonce = str(int(time.time() * 1000))
    body_encoded = requests.models.RequestEncodingMixin._encode_params(body)
    headers = {
        "APIKey": KRAKEN_API_KEY,
        "Authent": get_kraken_futures_signature(KRAKEN_API_SECRET, body_encoded, nonce, "/derivatives/api/v3/sendorder"),
        "Nonce": nonce
    }
    response = requests.post(KRAKEN_DERIVATIVES_URL, headers=headers, data=body_encoded, timeout=10)
    response.raise_for_status()
    return response.json()


def simulate_kraken_orders(coin_symbol, usd_amount, leverage, entry_price, tp_price, sl_price):
    print(f"\n🟢 REAL LONG TRADE (simulated) for {coin_symbol}")

    quantity = round(usd_amount / entry_price, 8)

    print("\n🔹 Limit BUY Order:")
    print(json.dumps({
        "symbol": KRAKEN_SYMBOL_MAP[coin_symbol],
        "side": "buy",
        "orderType": "limit",
        "price": round(entry_price, 8),
        "size": quantity,
        "marginMode": "isolated",
        "leverage": leverage
    }, indent=2))

    print("\n🔹 Take Profit Limit SELL:")
    print(json.dumps({
        "symbol": KRAKEN_SYMBOL_MAP[coin_symbol],
        "side": "sell",
        "orderType": "limit",
        "price": round(tp_price, 8),
        "size": quantity,
        "reduceOnly": True
    }, indent=2))

    stop_trigger_price = round(entry_price * 0.98, 8)     # triggers at -2.00%
    stop_limit_price   = round(entry_price * 0.9805, 8)   # attempts limit sell at -1.95%

    print("\n🔹 Stop Loss - Stop Limit:")
    print(json.dumps({
        "symbol": KRAKEN_SYMBOL_MAP[coin_symbol],
        "side": "sell",
        "orderType": "stopLimit",
        "stopPrice": stop_limit_price,
        "price": stop_limit_price,
        "size": quantity,
        "reduceOnly": True
    }, indent=2))

    print("\n🔹 Fallback Market Stop:")
    print(json.dumps({
        "symbol": KRAKEN_SYMBOL_MAP[coin_symbol],
        "side": "sell",
        "orderType": "market",
        "size": quantity,
        "reduceOnly": True,
        "note": "Failsafe in case stop-limit fails"
    }, indent=2))

def send_real_trade_updates():

    open_trades = RealTrade.objects.filter(exit_timestamp__isnull=True)

    if not open_trades.exists():
        return

    updates = []
    for trade in open_trades:
        coin_symbol = trade.coin.symbol
        kraken_symbol = KRAKEN_SYMBOL_MAP[coin_symbol]

        # Fetch current price via CoinAPI
        coinapi_symbol = COINAPI_SYMBOL_MAP[f"{coin_symbol}USDT"]
        url = f"https://rest.coinapi.io/v1/ohlcv/{coinapi_symbol}/latest?period_id=5MIN&limit=1"
        headers = {"X-CoinAPI-Key": COINAPI_KEY}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            continue
        current_price = data[0]["price_close"]

        direction = trade.trade_type.upper()
        entry_price = float(trade.entry_price)
        percent_change = ((current_price - entry_price) / entry_price) * 100
        percent_display = f"{percent_change:+.2f}%"

        updates.append(
            f"*{coin_symbol}* {direction}\n"
            f"📅 {trade.entry_timestamp.strftime('%Y-%m-%d %H:%M')}\n"
            f"💵 Entry: `{entry_price}`\n"
            f"📈 Now: `{current_price:.4f}`\n"
            f"📊 Change: *{percent_display}*\n"
        )

    send_text(updates)

def run_live_pipeline(request=None):
    import django
    django.setup()

    long_model = xgb.Booster()
    long_model.load_model("two_long_xgb_model.bin")

    short_model = xgb.Booster()
    short_model.load_model("two_short_xgb_model.bin")

    for coin in COINS:
        try:
            print(f"Processing {coin}...")
            df = fetch_ohlcv(coin, limit=300)

            if df.empty or len(df) < 220:
                print(f"⚠ Skipping {coin} due to insufficient raw data length")
                continue

            df = add_features(df)

            # Verify all features exist in df columns and are not NaN in last row
            missing_features = [f for f in FEATURES if f not in df.columns]
            if missing_features:
                print(f"⚠ Missing features {missing_features} for {coin}, skipping.")
                continue

            last_row = df.iloc[-1]
            nan_features = [f for f in FEATURES if pd.isna(last_row[f])]
            if nan_features:
                print(f"⚠ Features with NaN in last row: {nan_features} for {coin}, skipping.")
                continue

            instance, row = prepare_instance(df)
            feature_df = pd.DataFrame([instance])
            feature_df = feature_df[INPUT_COLUMNS]
            dmatrix = xgb.DMatrix(feature_df)

            #proba = model.predict(dmatrix)[0]
            long_proba = long_model.predict(dmatrix)[0]
            short_proba = short_model.predict(dmatrix)[0]

            print(f"{coin}: Long = {long_proba:.4f}, Short = {short_proba:.4f}")

            db_symbol = COIN_SYMBOL_MAP_DB.get(coin)
            coin_obj = Coin.objects.get(symbol=db_symbol)

            existing_trade = ModelTrade.objects.filter(
                coin=coin_obj,
                exit_timestamp__isnull=True
            ).first()

            if not existing_trade:
                if long_proba >= CONFIDENCE_THRESHOLD:
                    ModelTrade.objects.create(
                        coin=coin_obj,
                        trade_type="long",
                        entry_timestamp=row["timestamp"],
                        duration_minutes=0,
                        entry_price=row["close"],
                        model_confidence=long_proba,
                        take_profit_percent=4,
                        stop_loss_percent=2,
                        confidence_trade=CONFIDENCE_THRESHOLD
                    )
                    print(f"💰 LONG trade placed for {coin} @ {CONFIDENCE_THRESHOLD}")
                elif short_proba >= CONFIDENCE_THRESHOLD:
                    ModelTrade.objects.create(
                        coin=coin_obj,
                        trade_type="short",
                        entry_timestamp=row["timestamp"],
                        duration_minutes=0,
                        entry_price=row["close"],
                        model_confidence=short_proba,
                        take_profit_percent=4,
                        stop_loss_percent=2,
                        confidence_trade=CONFIDENCE_THRESHOLD
                    )
                    print(f"🔻 SHORT trade placed for {coin} @ {CONFIDENCE_THRESHOLD}")
            else:
                print(f"⚠ Trade already open for {coin}: {existing_trade.trade_type} @ {existing_trade.entry_timestamp}")


            live_balance = get_kraken_balance()
            if live_balance:
                print("✅ Kraken balance:")
                print(live_balance)

            # Real Trades
            if long_proba >= 0.8 and not RealTrade.objects.filter(exit_timestamp__isnull=True).exists():

                try:
                    usd_amount = 200
                    leverage = 10
                    tp_pct = 4.0
                    sl_pct = 2.0

                    raw_price = row["close"]
                    entry_price = round(raw_price * 0.999, 8)  # slightly under market
                    tp_price = round(entry_price * (1 + tp_pct / 100), 8)
                    sl_trigger = round(entry_price * (1 - sl_pct / 100), 8)  # trigger
                    sl_limit = round(entry_price * (1 - 0.0195), 8)  # -1.95%
                    quantity = round(usd_amount / entry_price, 8)
                    coin_symbol = COIN_SYMBOL_MAP_DB[coin]
                    kraken_symbol = KRAKEN_SYMBOL_MAP[coin_symbol]

                    coin_symbol = COIN_SYMBOL_MAP_DB[coin]
                    kraken_symbol = KRAKEN_SYMBOL_MAP[coin_symbol]

                    # 1. ENTRY LIMIT BUY
                    entry_order = {
                        "orderType": "post",
                        "symbol": kraken_symbol,
                        "side": "buy",
                        "size": str(quantity),
                        "limitPrice": str(entry_price),
                        "marginType": "isolated",
                        "leverage": str(leverage),
                    }
                    print("\n📤 Placing ENTRY order...")
                    res1 = place_kraken_futures_order(entry_order)
                    print("ENTRY ORDER RESPONSE:", res1)

                    # 2. TAKE PROFIT LIMIT SELL
                    tp_order = {
                        "orderType": "lmt",
                        "symbol": kraken_symbol,
                        "side": "sell",
                        "size": str(quantity),
                        "limitPrice": str(tp_price),
                        "reduceOnly": True,
                    }
                    print("\n📤 Placing TAKE PROFIT order...")
                    res2 = place_kraken_futures_order(tp_order)
                    print("TP ORDER RESPONSE:", res2)

                    # 3. STOP LIMIT
                    sl_order = {
                        "orderType": "stp_lmt",
                        "symbol": kraken_symbol,
                        "side": "sell",
                        "size": str(quantity),
                        "limitPrice": str(sl_price),
                        "stopPrice": str(sl_price),
                        "reduceOnly": True,
                    }
                    print("\n📤 Placing STOP LIMIT order...")
                    res3 = place_kraken_futures_order(sl_order)
                    print("STOP LIMIT ORDER RESPONSE:", res3)

                    # 4. FALLBACK MARKET STOP
                    fallback_sl = {
                        "orderType": "mkt",
                        "symbol": kraken_symbol,
                        "side": "sell",
                        "size": str(quantity),
                        "reduceOnly": True,
                    }
                    print("\n📤 Placing fallback MARKET SL...")
                    res4 = place_kraken_futures_order(fallback_sl)
                    print("FALLBACK SL ORDER RESPONSE:", res4)

                    # 5. Save RealTrade
                    RealTrade.objects.create(
                        coin=coin_obj,
                        trade_type="long",
                        entry_timestamp=row["timestamp"],
                        entry_price=entry_price,
                        model_confidence=long_proba,
                        take_profit_percent=tp_pct,
                        stop_loss_percent=sl_pct,
                        entry_usd_amount=usd_amount,
                        account_balance_before=live_balance
                    )

                    print(f"\n✅ REAL TRADE executed and logged for {coin_symbol}")

                except Exception as e:
                    print(f"❌ REAL trade error for {coin}: {e}")

        except Exception as e:
            print(f"❌ Error processing {coin}: {e}")

        send_real_trade_updates()

if __name__ == "__main__":
    run_live_pipeline()
