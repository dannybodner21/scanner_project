from django.core.management.base import BaseCommand
from scanner.models import Coin, ShortIntervalData, RickisMetrics, ModelTrade
from scanner.helpers import (
    round_to_five_minutes, get_recent_prices, get_recent_volumes,
    calculate_rsi, calculate_ema_from_prices, calculate_macd,
    calculate_stochastic, calculate_support_resistance, calculate_avg_volume_1h,
    calculate_relative_volume, calculate_sma, calculate_ema, calculate_price_slope_1h,
    calculate_stddev_1h, calculate_atr_1h, calculate_price_change_five_min,
    calculate_change_since_high, calculate_change_since_low, calculate_obv,
    calculate_fib_distances, calculate_bollinger_bands, calculate_adx
)
from django.utils.timezone import make_aware, is_naive, now
from datetime import datetime
import requests
import time



def get_chart_patterns_for_coin(symbol, finnhub_api_key):
    url = "https://finnhub.io/api/v1/scan/pattern"
    patterns = {}
    resolutions = [5, 15, 60]

    for res in resolutions:
        params = {
            "symbol": symbol,
            "resolution": res,
            "token": finnhub_api_key
        }
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            if data.get('points'):
                latest = data['points'][0]

                status = latest.status or ""
                patterntype = latest.patterntype or "no pattern"
                patternname = lastest.patternname or "no name"
                target = latest.profit1 or "no target"

                chart_string = f'{status} {patterntype} {patternname} | target: {target}'

                patterns[f'pattern_{res}m'] = chart_string
            else:
                patterns[f'pattern_{res}m'] = 'No Pattern'
        except Exception as e:
            print(f"❌ Error fetching pattern for {symbol} at {res}m: {e}")
            patterns[f'pattern_{res}m'] = 'No Pattern'

    return patterns



def get_fear_greed_index():
    url = "https://api.alternative.me/fng/?limit=1&format=json"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        value = int(data['data'][0]['value'])
        return value
    except Exception as e:
        print(f"❌ Error fetching Fear & Greed Index: {e}")
        return None



def run_five_min_update_logic():
    start = datetime.now()
    print(f"\n⏱️ Start: {start}")

    API_KEY_QUOTES = 'c35740fd-4f78-45b5-9350-c4afdd929432'
    API_KEY_OHLCV = '7dd5dd98-35d0-475d-9338-407631033cd9'
    FINNHUB_API_KEY = 'cuf7nohr01qno7m552hgcuf7nohr01qno7m552i0'

    url_quotes = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
    url_ohlcv = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/latest'

    headers_quotes = {"Accepts": "application/json", "X-CMC_PRO_API_KEY": API_KEY_QUOTES}
    headers_ohlcv = {"Accepts": "application/json", "X-CMC_PRO_API_KEY": API_KEY_OHLCV}

    coins = Coin.objects.all()
    cmc_ids = [coin.cmc_id for coin in coins]

    rickisSymbols = [
        "BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LINK",
        "AVAX", "XLM", "TON", "SHIB", "SUI", "HBAR", "BCH", "DOT", "LTC",
        "XMR", "UNI", "PEPE", "APT", "NEAR", "ONDO", "TAO", "ICP", "ETC",
        "RENDER", "MNT", "KAS", "CRO", "AAVE", "POL", "VET", "FIL", "ALGO",
        "ENA", "ATOM", "TIA", "ARB", "DEXE", "OP", "JUP", "MKR", "STX",
        "EOS", "WLD", "BONK", "FARTCOIN", "SEI", "INJ", "IMX", "GRT",
        "PAXG", "CRV", "JASMY", "SAND", "GALA", "CORE", "KAIA", "LDO",
        "THETA", "IOTA", "HNT", "MANA", "FLOW", "CAKE", "MOVE", "FLOKI"
    ]

    batch_size = 100
    for i in range(0, len(cmc_ids), batch_size):
        cmc_id_batch = cmc_ids[i:i+batch_size]
        params = {"id": ",".join(map(str, cmc_id_batch)), "convert": "USD"}

        try:
            response_quotes = requests.get(url_quotes, headers=headers_quotes, params=params)
            response_ohlcv = requests.get(url_ohlcv, headers=headers_ohlcv, params=params)
            response_quotes.raise_for_status()
            response_ohlcv.raise_for_status()
            data_quotes = response_quotes.json()
            data_ohlcv = response_ohlcv.json()

            for cmc_id in cmc_id_batch:
                if str(cmc_id) in data_quotes["data"] and str(cmc_id) in data_ohlcv["data"]:
                    crypto_data = data_quotes["data"][str(cmc_id)]
                    ohlcv_data = data_ohlcv["data"][str(cmc_id)]

                    timestamp = datetime.strptime(
                        crypto_data["last_updated"], "%Y-%m-%dT%H:%M:%S.%fZ"
                    )
                    if is_naive(timestamp):
                        timestamp = make_aware(timestamp)
                    timestamp = round_to_five_minutes(timestamp)

                    coin = Coin.objects.get(cmc_id=cmc_id)
                    current_price = crypto_data["quote"]["USD"]["price"]

                    if current_price is None:
                        continue

                    ShortIntervalData.objects.get_or_create(
                        coin=coin,
                        timestamp=timestamp,
                        defaults={
                            'price': current_price,
                            'volume_5min': crypto_data["quote"]["USD"].get("volume_24h", 0),
                            'circulating_supply': crypto_data.get("circulating_supply", 0)
                        }
                    )

                    if coin.symbol in rickisSymbols:
                        try:
                            macd, macd_signal = calculate_macd(coin, timestamp)
                            stochastic_k, stochastic_d = calculate_stochastic(coin, timestamp)

                            if stochastic_k is None or stochastic_d is None:
                                raise ValueError("Missing stochastic values")

                            support, resistance = calculate_support_resistance(coin, timestamp)
                            fib_distances = calculate_fib_distances(
                                high=ohlcv_data["quote"]["USD"].get("high"),
                                low=ohlcv_data["quote"]["USD"].get("low"),
                                current_price=current_price,
                            )

                            adx = calculate_adx(coin, timestamp)
                            bollinger_upper, bollinger_middle, bollinger_lower = calculate_bollinger_bands(coin, timestamp)



                            # Finnhub stuff ------------------------------------
                            finnhub_symbol = f"BINANCE:{coin.symbol}USDT"
                            patterns = get_chart_patterns_for_coin(finnhub_symbol, FINNHUB_API_KEY)

                            print(patterns)

                            chart_pattern_5m = patterns.get('pattern_5m', 'No Pattern')
                            chart_pattern_15m = patterns.get('pattern_15m', 'No Pattern')
                            chart_pattern_60m = patterns.get('pattern_60m', 'No Pattern')

                            print("--------------------")
                            print(chart_pattern_5m)
                            print("--------------------")
                            print(chart_pattern_15m)
                            print("--------------------")
                            print(chart_pattern_60m)
                            print("--------------------")

                            metrics, _ = RickisMetrics.objects.update_or_create(
                                coin=coin,
                                timestamp=timestamp,
                                defaults={
                                    'price': current_price,
                                    'high_24h': ohlcv_data["quote"]["USD"].get("high"),
                                    'low_24h': ohlcv_data["quote"]["USD"].get("low"),
                                    'open': ohlcv_data["quote"]["USD"].get("open"),
                                    'close': ohlcv_data["quote"]["USD"].get("close"),
                                    'change_1h': crypto_data["quote"]["USD"].get("percent_change_1h"),
                                    'change_24h': crypto_data["quote"]["USD"].get("percent_change_24h"),
                                    'volume': crypto_data["quote"]["USD"].get("volume_24h"),
                                    'avg_volume_1h': calculate_avg_volume_1h(coin, timestamp),
                                    'rsi': calculate_rsi(coin, timestamp),
                                    'macd': macd,
                                    'macd_signal': macd_signal,
                                    'stochastic_k': stochastic_k,
                                    'stochastic_d': stochastic_d,
                                    'support_level': support,
                                    'resistance_level': resistance,
                                    'sma_5': calculate_sma(coin, timestamp, window=5),
                                    'sma_20': calculate_sma(coin, timestamp, window=20),
                                    'stddev_1h': calculate_stddev_1h(coin, timestamp),
                                    'atr_1h': calculate_atr_1h(coin, timestamp),
                                    'change_since_high': calculate_change_since_high(current_price, ohlcv_data["quote"]["USD"].get("high")),
                                    'change_since_low': calculate_change_since_low(current_price, ohlcv_data["quote"]["USD"].get("low")),
                                    'fib_distance_0_236': fib_distances.get("fib_distance_0_236"),
                                    'fib_distance_0_382': fib_distances.get("fib_distance_0_382"),
                                    'fib_distance_0_5': fib_distances.get("fib_distance_0_5"),
                                    'fib_distance_0_618': fib_distances.get("fib_distance_0_618"),
                                    'fib_distance_0_786': fib_distances.get("fib_distance_0_786"),
                                    'adx': adx,
                                    'bollinger_upper': bollinger_upper,
                                    'bollinger_middle': bollinger_middle,
                                    'bollinger_lower': bollinger_lower,
                                    'chart_pattern_5m': patterns.get('pattern_5m', 'No Pattern'),
                                    'chart_pattern_15m': patterns.get('pattern_15m', 'No Pattern'),
                                    'chart_pattern_60m': patterns.get('pattern_60m', 'No Pattern'),

                                }
                            )

                            five_min_price_change = calculate_price_change_five_min(coin, timestamp)
                            if five_min_price_change is None:
                                raise ValueError("Missing price change five min")

                            rel_vol = calculate_relative_volume(coin, timestamp)
                            if rel_vol is None:
                                raise ValueError("Missing relative_volume")

                            obv = calculate_obv(coin, timestamp)
                            if obv is None:
                                raise ValueError("Missing obv")

                            metrics.relative_volume = rel_vol
                            metrics.change_5m = five_min_price_change
                            metrics.obv = obv
                            metrics.save()

                        except Exception as e:
                            print(f"❌ Failed RickisMetrics for {coin.symbol}: {e}")

        except Exception as e:
            print(f"❌ API batch error {cmc_id_batch}: {e}")

        time.sleep(1.5)

    print("\n✅ Five minute update complete.")

    # Update model trades
    open_trades = ModelTrade.objects.filter(exit_timestamp__isnull=True)

    for trade in open_trades:
        try:
            latest_metric = RickisMetrics.objects.filter(coin=trade.coin).order_by("-timestamp").first()
            if not latest_metric:
                continue

            current_price = float(latest_metric.price)
            entry = float(trade.entry_price)
            tp = 3.0
            sl = 2.0
            result = True

            if trade.trade_type == "long":
                if current_price >= entry * 1.03:
                    status = "💰 TAKE PROFIT"
                elif current_price <= entry * 0.98:
                    status = "🛑 STOP LOSS"
                    result = False
                else:
                    continue
            else:  # short
                if current_price <= entry * 0.97:
                    status = "💰 TAKE PROFIT"
                elif current_price >= entry * 1.02:
                    status = "🛑 STOP LOSS"
                    result = False
                else:
                    continue

            trade.exit_price = current_price
            trade.exit_timestamp = now()
            trade.result = result
            trade.save()

            print(f"{status} | {trade.trade_type.upper()} {trade.coin.symbol} @ {current_price:.6f}")

        except Exception as e:
            print(f"❌ Error evaluating trade for {trade.coin.symbol}: {e}")

class Command(BaseCommand):
    help = 'Run the five minute update logic'

    def handle(self, *args, **kwargs):
        run_five_min_update_logic()
