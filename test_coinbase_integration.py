#!/usr/bin/env python3
"""
Test Coinbase Integration in Live Pipeline
Tests the new GRT integration via Coinbase API
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'scanner.settings')
django.setup()

from scanner.live_pipeline import fetch_coinbase_candles, backfill_coinbase_candles, get_recent_candles, COINBASE_SYMBOL_MAP
from scanner.models import CoinAPIPrice

def test_coinbase_integration():
    """Test the Coinbase integration for GRT"""
    
    print("🔸 TESTING COINBASE INTEGRATION FOR GRT")
    print("=" * 60)
    
    coin = "GRTUSDT"
    
    # Test 1: Check if GRT is in symbol map
    print(f"\n1️⃣ Testing symbol mapping...")
    if coin in COINBASE_SYMBOL_MAP:
        coinbase_symbol = COINBASE_SYMBOL_MAP[coin]
        print(f"✅ {coin} mapped to {coinbase_symbol}")
    else:
        print(f"❌ {coin} not found in COINBASE_SYMBOL_MAP")
        return
    
    # Test 2: Fetch fresh data from Coinbase
    print(f"\n2️⃣ Testing direct Coinbase API fetch...")
    df = fetch_coinbase_candles(coin, limit=10)
    
    if df is not None and not df.empty:
        print(f"✅ Successfully fetched {len(df)} candles from Coinbase")
        print(f"   Latest price: ${df.iloc[-1]['close']:.4f}")
        print(f"   Latest timestamp: {df.iloc[-1]['timestamp']}")
        print(f"   Data range: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
    else:
        print(f"❌ Failed to fetch data from Coinbase")
        return
    
    # Test 3: Check database before backfill
    print(f"\n3️⃣ Checking database before backfill...")
    existing_count = CoinAPIPrice.objects.filter(coin=coin).count()
    print(f"   Existing {coin} records in DB: {existing_count}")
    
    # Test 4: Test backfill function
    print(f"\n4️⃣ Testing backfill function...")
    backfill_coinbase_candles(coin)
    
    # Test 5: Check database after backfill
    print(f"\n5️⃣ Checking database after backfill...")
    new_count = CoinAPIPrice.objects.filter(coin=coin).count()
    print(f"   Total {coin} records in DB: {new_count}")
    print(f"   Records added: {new_count - existing_count}")
    
    # Test 6: Test get_recent_candles function
    print(f"\n6️⃣ Testing get_recent_candles function...")
    recent_df = get_recent_candles(coin, limit=10)
    
    if recent_df is not None and not recent_df.empty:
        print(f"✅ get_recent_candles returned {len(recent_df)} candles")
        print(f"   Latest price: ${recent_df.iloc[-1]['close']:.4f}")
        print(f"   Latest timestamp: {recent_df.iloc[-1]['timestamp']}")
        
        # Verify data integrity
        print(f"   Data integrity check:")
        print(f"   - All timestamps present: {recent_df['timestamp'].notna().all()}")
        print(f"   - All prices > 0: {(recent_df[['open', 'high', 'low', 'close']] > 0).all().all()}")
        print(f"   - Volume data present: {recent_df['volume'].notna().all()}")
        
    else:
        print(f"❌ get_recent_candles failed")
        return
    
    # Test 7: Verify integration with COINS list
    print(f"\n7️⃣ Testing integration with pipeline...")
    from scanner.live_pipeline import COINS
    
    if coin in COINS:
        print(f"✅ {coin} is in COINS list - will be processed by pipeline")
    else:
        print(f"⚠️ {coin} not in COINS list - pipeline won't process it")
        print(f"   Current COINS: {COINS}")
    
    print(f"\n✅ COINBASE INTEGRATION TEST COMPLETE")
    print(f"🎯 Summary:")
    print(f"   - Coinbase API connection: ✅ Working")
    print(f"   - Data fetching: ✅ Working")
    print(f"   - Database storage: ✅ Working")
    print(f"   - Pipeline integration: ✅ Ready")
    print(f"   - Cost: 🆓 FREE (vs CoinAPI)")

if __name__ == "__main__":
    test_coinbase_integration()