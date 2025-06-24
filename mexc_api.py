import time, hmac, hashlib, requests

API_KEY = 'YOUR_KEY'
API_SECRET = 'YOUR_SECRET'
BASE_URL = 'https://contract.mexc.com'

def sign(params):
    qs = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
    return hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()

def post(endpoint, params):
    params["api_key"] = API_KEY
    params["req_time"] = int(time.time() * 1000)
    params["sign"] = sign(params)
    return requests.post(BASE_URL + endpoint, data=params).json()

def place_limit_long(symbol, price, volume, leverage=10):
    return post("/api/v1/private/order/submit", {
        "symbol": symbol,
        "price": price,
        "vol": volume,
        "side": 1,              # Open Long
        "open_type": 1,         # 1=Isolated
        "leverage": leverage,
        "order_type": 1         # 1=Limit
    })

def place_limit_tp(symbol, price, volume):
    return post("/api/v1/private/order/submit", {
        "symbol": symbol,
        "price": price,
        "vol": volume,
        "side": 3,              # Close Long
        "open_type": 1,
        "order_type": 1
    })

def place_market_sl(symbol, trigger_price, volume):
    return post("/api/v1/private/planorder/place", {
        "symbol": symbol,
        "trigger_price": trigger_price,
        "price": 0,             # market execution
        "vol": volume,
        "side": 3,              # Close Long
        "open_type": 1,
        "order_type": 2,        # Market
        "trigger_type": 1       # Last price
    })

def execute_trade(symbol, entry_price, volume):
    tp_price = round(entry_price * 1.04, 2)
    sl_price = round(entry_price * 0.98, 2)

    print(f"Placing long entry: {symbol} @ {entry_price}")
    entry = place_limit_long(symbol, entry_price, volume)
    print("Entry response:", entry)

    print(f"Placing TP @ {tp_price}")
    tp = place_limit_tp(symbol, tp_price, volume)
    print("TP response:", tp)

    print(f"Placing SL trigger @ {sl_price} (market fallback)")
    sl = place_market_sl(symbol, sl_price, volume)
    print("SL response:", sl)

# Example usage
execute_trade("BTC_USDT", entry_price=64000, volume=0.001)
