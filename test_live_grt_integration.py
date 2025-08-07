#!/usr/bin/env python3
"""
Test GRT integration in live pipeline
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'scanner_project.settings')
sys.path.append('/Users/danielbodner/Desktop/scanner_project')
django.setup()

# Import live pipeline functions
from scanner.live_pipeline import fetch_latest_candle_coinbase, get_recent_candles

def test_live_grt_integration():
    """Test GRT integration in live pipeline"""
    
    print("🔸 TESTING LIVE GRT INTEGRATION")
    print("=" * 45)
    
    # Test 1: Coinbase function directly
    print("Test 1: fetch_latest_candle_coinbase('GRTUSDT')")
    try:
        result = fetch_latest_candle_coinbase('GRTUSDT')
        if result:
            print("✅ Success!")
            print(f"   Coin: {result['coin']}")
            print(f"   Timestamp: {result['timestamp']}")
            print(f"   Price: ${result['close']:.4f}")
            print(f"   Volume: {result['volume']:.1f}")
        else:
            print("❌ Failed - returned None")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: get_recent_candles function
    print(f"\nTest 2: get_recent_candles('GRTUSDT', 10)")
    try:
        df = get_recent_candles('GRTUSDT', 10)
        if df is not None and len(df) > 0:
            print(f"✅ Success! Got {len(df)} candles")
            print(f"   Latest timestamp: {df['timestamp'].max()}")
            print(f"   Latest price: ${df['close'].iloc[-1]:.4f}")
            print(f"   Price range: ${df['close'].min():.4f} - ${df['close'].max():.4f}")
        else:
            print("❌ Failed - no data returned")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Check if GRT is properly routed to Coinbase
    print(f"\nTest 3: Verify GRT routing")
    from scanner.live_pipeline import COINBASE_SYMBOL_MAP, COIN_SYMBOL_MAP_DB
    
    grt_in_coinbase = 'GRTUSDT' in COINBASE_SYMBOL_MAP
    grt_in_db_map = 'GRTUSDT' in COIN_SYMBOL_MAP_DB
    
    print(f"   GRT in COINBASE_SYMBOL_MAP: {grt_in_coinbase}")
    if grt_in_coinbase:
        print(f"   GRT maps to: {COINBASE_SYMBOL_MAP['GRTUSDT']}")
    
    print(f"   GRT in COIN_SYMBOL_MAP_DB: {grt_in_db_map}")
    if grt_in_db_map:
        print(f"   GRT DB symbol: {COIN_SYMBOL_MAP_DB['GRTUSDT']}")
    
    if grt_in_coinbase and grt_in_db_map:
        print("   ✅ GRT properly configured for Coinbase → Database routing")
    else:
        print("   ❌ GRT routing configuration incomplete")
    
    return True

if __name__ == "__main__":
    test_live_grt_integration()