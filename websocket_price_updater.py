# websocket_price_updater.py

import asyncio
import websockets
import json
import django
import os
from decimal import Decimal
from price_cache import latest_prices, price_lock

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "scanner_project.settings")  # ← CHANGE THIS
django.setup()

from scanner.models import RealTrade

POLYGON_API_KEY = 'qq9Sptr4VfkonQimqFJEgc3oyXoaJ54L'

COIN_SYMBOLS = [
    'BTC', 'ETH', 'XRP', 'LTC', 'SOL',
    'DOGE', 'LINK', 'DOT', 'SHIB', 'ADA'
]

symbol_map = {
    'BTC': 'XT.X:BTC-USD',
    'ETH': 'XT.X:ETH-USD',
    'XRP': 'XT.X:XRP-USD',
    'LTC': 'XT.X:LTC-USD',
    'SOL': 'XT.X:SOL-USD',
    'DOGE': 'XT.X:DOGE-USD',
    'LINK': 'XT.X:LINK-USD',
    'DOT': 'XT.X:DOT-USD',
    'SHIB': 'XT.X:SHIB-USD',
    'ADA': 'XT.X:ADA-USD',
}

async def handle_polygon_ws():
    uri = "wss://socket.polygon.io/crypto"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"action": "auth", "params": POLYGON_API_KEY}))

        for symbol in COIN_SYMBOLS:
            await ws.send(json.dumps({
                "action": "subscribe",
                "params": symbol_map[symbol]
            }))

        while True:
            try:
                msg = await ws.recv()
                data = json.loads(msg)

                for update in data:
                    if update.get("ev") == "XT":
                        symbol = update["pair"].split(":")[1].split("-")[0]
                        price = Decimal(str(update["p"]))
                        with price_lock:
                            latest_prices[symbol] = price
            except Exception as e:
                print(f"WebSocket error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(handle_polygon_ws())
