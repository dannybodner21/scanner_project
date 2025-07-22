import requests
import time
import hmac
import hashlib
import base64
import json
from datetime import datetime, timedelta
import pandas as pd

# 🔒 Your credentials
API_KEY = "2a0144b4-2da4-41f3-9a17-c7ce9fcb185c"
API_SECRET = "tXnfodWgvRtb3yvQhj6id3FOTFphpFC2fNnDOyvtsY3jPubEptwsAaxzNebbhLidCUTIjeqaLGdOGklLqHaXTg=="   

def generate_signature(secret, timestamp, method, request_path, body=""):
    prehash = f"{timestamp}{method}{request_path}{body}"
    signature = hmac.new(
        secret.encode('utf-8'),
        prehash.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

def fetch_coinbase_ohlcv(product_id="BTC-USDT", limit=300):
    granularity = 300
    now = datetime.utcnow().replace(second=0, microsecond=0)
    to_time = now - timedelta(minutes=now.minute % 5)
    from_time = to_time - timedelta(seconds=granularity * limit)

    start_iso = from_time.isoformat() + "Z"
    end_iso = to_time.isoformat() + "Z"

    method = "GET"
    request_path = f"/api/v3/brokerage/products/{product_id}/candles"
    query_string = f"?start={start_iso}&end={end_iso}&granularity={granularity}"
    full_path = request_path + query_string

    timestamp = str(int(time.time()))
    signature = generate_signature(API_SECRET, timestamp, method, request_path, "")

    headers = {
        "CB-ACCESS-KEY": API_KEY,
        "CB-ACCESS-SIGN": signature,
        "CB-ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }

    url = f"https://api.coinbase.com{full_path}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    candles = response.json().get("candles", [])

    df = pd.DataFrame([{
        "timestamp": pd.to_datetime(row[0], unit="s"),
        "low": row[1],
        "high": row[2],
        "open": row[3],
        "close": row[4],
        "volume": row[5]
    } for row in candles])

    return df.sort_values("timestamp").reset_index(drop=True)

# 🔧 Test
if __name__ == "__main__":
    df = fetch_coinbase_ohlcv("BTC-USDT", limit=300)
    print(df.tail())