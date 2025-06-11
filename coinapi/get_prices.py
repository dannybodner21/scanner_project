import requests

API_KEY = "01293e2a-dcf1-4e81-8310-c6aa9d0cb743"
symbol_id = "BINANCE_SPOT_BTC_USDT"

url = f"https://rest.coinapi.io/v1/ohlcv/{symbol_id}/history"

params = {
    "period_id": "5MIN",
    "time_start": "2025-01-01T00:00:00",
    "limit": 100
}

headers = {"X-CoinAPI-Key": API_KEY}

response = requests.get(url, headers=headers, params=params)
response.raise_for_status()
data = response.json()

for candle in data:
    print(candle)
