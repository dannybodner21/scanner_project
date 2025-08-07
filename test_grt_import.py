#!/usr/bin/env python3
"""
Test GRT import with small sample data
"""

import os
import sys
import django
import json
import pandas as pd

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'scanner_project.settings')
sys.path.append('/Users/danielbodner/Desktop/scanner_project')
django.setup()

from scanner.models import CoinAPIPrice

def test_grt_import():
    """Test importing a small sample of GRT data"""
    
    print("🔸 TESTING GRT DATA IMPORT")
    print("=" * 40)
    
    try:
        # Load sample from JSON file
        with open('grt_3months_data.json', 'r') as f:
            all_candles = json.load(f)
        
        print(f"📂 Loaded {len(all_candles)} candles from file")
        
        # Take first 10 candles for testing
        sample_candles = all_candles[:10]
        
        print(f"🧪 Testing with {len(sample_candles)} sample candles")
        
        # Check existing GRT data
        existing_count = CoinAPIPrice.objects.filter(coin='GRTUSDT').count()
        print(f"📊 Existing GRT records: {existing_count}")
        
        # Import sample data
        imported = 0
        duplicates = 0
        
        for candle_data in sample_candles:
            try:
                timestamp = pd.to_datetime(candle_data['timestamp'])
                
                obj, created = CoinAPIPrice.objects.update_or_create(
                    coin=candle_data['coin'],
                    timestamp=timestamp,
                    defaults={
                        'open': float(candle_data['open']),
                        'high': float(candle_data['high']),
                        'low': float(candle_data['low']),
                        'close': float(candle_data['close']),
                        'volume': float(candle_data['volume'])
                    }
                )
                
                if created:
                    imported += 1
                    print(f"   ✅ Imported: {timestamp} | ${candle_data['close']:.4f}")
                else:
                    duplicates += 1
                    print(f"   ⚠️ Duplicate: {timestamp}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Summary
        print(f"\n📊 SAMPLE IMPORT RESULTS:")
        print(f"   New records: {imported}")
        print(f"   Duplicates: {duplicates}")
        
        # Check final count
        final_count = CoinAPIPrice.objects.filter(coin='GRTUSDT').count()
        print(f"   Total GRT records: {final_count}")
        
        if final_count > 0:
            latest = CoinAPIPrice.objects.filter(coin='GRTUSDT').order_by('-timestamp').first()
            earliest = CoinAPIPrice.objects.filter(coin='GRTUSDT').order_by('timestamp').first()
            print(f"   Data range: {earliest.timestamp.date()} to {latest.timestamp.date()}")
        
        print(f"\n✅ Sample import successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_grt_import()