#!/usr/bin/env python3
"""
Add Coinbase Advanced API for more coins without CoinAPI costs
Coinbase is US-compliant and free for market data
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz

# Additional coins available on Coinbase (US-compliant)
COINBASE_COINS = {
    'MATIC-USD': 'MATICUSDT',
    'ATOM-USD': 'ATOMUSDT', 
    'NEAR-USD': 'NEARUSDT',
    'FIL-USD': 'FILUSDT',
    'AAVE-USD': 'AAVEUSDT',
    'COMP-USD': 'COMPUSDT',
    'MKR-USD': 'MKRUSDT',
    'SNX-USD': 'SNXUSDT',
    'CRV-USD': 'CRVUSDT',
    'YFI-USD': 'YFIUSDT',
    'SUSHI-USD': 'SUSHIUSDT',
    'RUNE-USD': 'RUNEUSDT',
}

def get_coinbase_candles(coin_symbol, limit=100):
    """
    Get OHLCV data from Coinbase Advanced API (FREE)
    coin_symbol: 'BTC-USD', 'ETH-USD', etc.
    """
    try:
        # Coinbase uses granularity in seconds (300 = 5 minutes)
        url = f'https://api.exchange.coinbase.com/products/{coin_symbol}/candles'
        params = {
            'granularity': 300,  # 5 minutes in seconds
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            print(f"⚠️ No data returned for {coin_symbol}")
            return None
        
        # Coinbase returns: [timestamp, low, high, open, close, volume]
        df_data = []
        for candle in data[:limit]:
            df_data.append({
                'timestamp': pd.to_datetime(candle[0], unit='s', utc=True),
                'open': float(candle[3]),
                'high': float(candle[2]), 
                'low': float(candle[1]),
                'close': float(candle[4]),
                'volume': float(candle[5])
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ {coin_symbol}: Retrieved {len(df)} candles")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching {coin_symbol}: {e}")
        return None

def test_coinbase_integration():
    """Test Coinbase API integration"""
    
    print("🔸 TESTING COINBASE API INTEGRATION")
    print("=" * 50)
    
    test_coins = ['BTC-USD', 'ETH-USD', 'MATIC-USD', 'ATOM-USD']
    
    for coin in test_coins:
        print(f"\n  Testing {coin}...")
        df = get_coinbase_candles(coin, limit=10)
        
        if df is not None:
            latest = df.iloc[-1]
            print(f"    Latest: {latest['timestamp']}")
            print(f"    Price: ${latest['close']:.4f}")
            print(f"    Volume: {latest['volume']:.2f}")
        else:
            print(f"    ❌ Failed to get data")
    
    print(f"\n📊 POTENTIAL COIN EXPANSION:")
    print(f"   Current CoinAPI coins: 13")
    print(f"   Additional Coinbase coins: {len(COINBASE_COINS)}")
    print(f"   Total possible coins: {13 + len(COINBASE_COINS)}")
    print(f"   Additional cost: $0 (Coinbase is free)")

def calculate_cost_savings():
    """Calculate potential cost savings"""
    
    print(f"\n💰 COST ANALYSIS:")
    print("=" * 50)
    
    # Current costs
    current_calls_per_day = 13 * 288  # 13 coins, every 5 min
    coinapi_cost_per_call = 0.0001  # Estimate
    daily_cost = current_calls_per_day * coinapi_cost_per_call
    monthly_cost = daily_cost * 30
    
    print(f"   Current CoinAPI usage:")
    print(f"   - Calls per day: {current_calls_per_day}")
    print(f"   - Estimated monthly cost: ${monthly_cost:.2f}")
    
    # With Coinbase addition
    additional_coins = len(COINBASE_COINS)
    additional_signals = additional_coins * 288 * 30  # Per month
    
    print(f"\n   With Coinbase addition:")
    print(f"   - Additional coins: {additional_coins}")
    print(f"   - Additional signals/month: {additional_signals}")
    print(f"   - Additional cost: $0.00 (FREE)")
    print(f"   - Total coins: {13 + additional_coins}")

if __name__ == "__main__":
    test_coinbase_integration()
    calculate_cost_savings()
    
    print(f"\n🎯 RECOMMENDATION:")
    print("   1. Keep existing 13 CoinAPI coins (proven data quality)")
    print("   2. Add 12 Coinbase coins for more signals (free)")
    print("   3. Total: 25 coins without increasing costs")
    print("   4. Monitor signal quality and adjust as needed")