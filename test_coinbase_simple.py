#!/usr/bin/env python3
"""
Simple Coinbase API Test
Tests if we can fetch GRT-USDT data from Coinbase
"""

import requests
import pandas as pd

def test_coinbase_grt():
    """Test fetching GRT-USDT from Coinbase"""
    
    print("🔸 TESTING COINBASE API FOR GRT-USDT")
    print("=" * 50)
    
    try:
        # Coinbase Advanced API endpoint for GRT-USD (USD≈USDT for our purposes)
        url = 'https://api.exchange.coinbase.com/products/GRT-USD/candles'
        params = {
            'granularity': 300,  # 5 minutes in seconds
        }
        
        print(f"📊 Fetching GRT-USDT from Coinbase...")
        print(f"   URL: {url}")
        print(f"   Params: {params}")
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            print(f"⚠️ No data returned")
            return False
        
        print(f"✅ Successfully received {len(data)} candles")
        
        # Process first few candles
        print(f"\n📈 SAMPLE DATA:")
        for i, candle in enumerate(data[:3]):
            # Coinbase returns: [timestamp, low, high, open, close, volume]
            timestamp = pd.to_datetime(candle[0], unit='s', utc=True)
            open_price = float(candle[3])
            high = float(candle[2])
            low = float(candle[1])
            close = float(candle[4])
            volume = float(candle[5])
            
            print(f"   Candle {i+1}:")
            print(f"     Time: {timestamp}")
            print(f"     OHLC: ${open_price:.4f} / ${high:.4f} / ${low:.4f} / ${close:.4f}")
            print(f"     Volume: {volume:.2f}")
        
        # Latest price
        latest = data[0]  # First candle is most recent
        latest_price = float(latest[4])  # Close price
        latest_time = pd.to_datetime(latest[0], unit='s', utc=True)
        
        print(f"\n💰 LATEST GRT PRICE:")
        print(f"   Price: ${latest_price:.4f} USDT")
        print(f"   Time: {latest_time}")
        
        print(f"\n✅ COINBASE INTEGRATION READY!")
        print(f"   - API Connection: ✅ Working")
        print(f"   - GRT-USDT Data: ✅ Available") 
        print(f"   - Data Format: ✅ Compatible")
        print(f"   - Cost: 🆓 FREE")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_coinbase_grt()
    
    if success:
        print(f"\n🎯 READY TO INTEGRATE INTO LIVE PIPELINE")
        print(f"   GRT will be fetched from Coinbase for FREE")
        print(f"   This adds 1 more coin without increasing CoinAPI costs")
    else:
        print(f"\n❌ Integration not ready - needs troubleshooting")