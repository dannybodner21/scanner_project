from decimal import Decimal
from threading import Lock

latest_prices = {}
price_lock = Lock()
