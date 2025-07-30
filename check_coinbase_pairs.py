#!/usr/bin/env python3
"""
Check Available GRT Trading Pairs on Coinbase
"""

import requests

def check_coinbase_products():
    """Check what GRT pairs are available on Coinbase"""
    
    print("🔍 CHECKING COINBASE TRADING PAIRS")
    print("=" * 50)
    
    try:
        # Get all available products
        url = 'https://api.exchange.coinbase.com/products'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        products = response.json()
        
        # Filter for GRT pairs
        grt_pairs = [p for p in products if 'GRT' in p['id']]
        
        print(f"📊 Found {len(grt_pairs)} GRT trading pairs:")
        for pair in grt_pairs:
            print(f"   {pair['id']}: {pair['display_name']}")
            print(f"     Status: {pair['status']}")
            print(f"     Quote: {pair['quote_currency']}")
            print()
        
        # Check if we have USD pair
        grt_usd = [p for p in grt_pairs if p['quote_currency'] == 'USD']
        if grt_usd:
            print(f"✅ GRT-USD available: {grt_usd[0]['id']}")
            return 'GRT-USD'
        else:
            print(f"❌ No GRT-USD pair found")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_grt_usd():
    """Test fetching GRT-USD data"""
    
    print(f"\n📊 TESTING GRT-USD DATA FETCH")
    print("=" * 50)
    
    try:
        url = 'https://api.exchange.coinbase.com/products/GRT-USD/candles'
        params = {'granularity': 300}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data:
            print(f"✅ Successfully fetched {len(data)} GRT-USD candles")
            
            # Latest price
            latest = data[0]
            latest_price = float(latest[4])
            
            print(f"💰 Latest GRT price: ${latest_price:.4f} USD")
            return True
        else:
            print(f"❌ No data returned")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    # Check available pairs
    grt_pair = check_coinbase_products()
    
    if grt_pair:
        # Test the available pair
        success = test_grt_usd()
        
        if success:
            print(f"\n🎯 SOLUTION:")
            print(f"   Use GRT-USD instead of GRT-USDT")
            print(f"   Convert USD to USDT equivalent in the pipeline")
            print(f"   This still gives us GRT data for FREE")
        else:
            print(f"\n❌ GRT-USD data fetch failed")
    else:
        print(f"\n❌ No suitable GRT pair found on Coinbase")