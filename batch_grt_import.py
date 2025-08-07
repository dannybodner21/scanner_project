#!/usr/bin/env python3
"""
Import GRT data in batches to avoid timeouts
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

def batch_import_grt():
    """Import GRT data in batches"""
    
    print("🚀 BATCH GRT DATA IMPORT")
    print("=" * 50)
    
    try:
        # Load data
        with open('grt_3months_data.json', 'r') as f:
            all_candles = json.load(f)
        
        print(f"📂 Loaded {len(all_candles)} candles from file")
        
        # Check existing data
        existing_count = CoinAPIPrice.objects.filter(coin='GRTUSDT').count()
        print(f"📊 Existing GRT records: {existing_count}")
        
        # Import in batches of 1000
        batch_size = 1000
        total_imported = 0
        total_duplicates = 0
        batch_count = 0
        
        for i in range(0, len(all_candles), batch_size):
            batch_count += 1
            batch = all_candles[i:i + batch_size]
            
            print(f"\nBatch {batch_count}: Processing {len(batch)} candles (records {i+1}-{min(i+batch_size, len(all_candles))})")
            
            batch_imported = 0
            batch_duplicates = 0
            
            for candle_data in batch:
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
                        batch_imported += 1
                        total_imported += 1
                    else:
                        batch_duplicates += 1
                        total_duplicates += 1
                
                except Exception as e:
                    print(f"   ❌ Error importing candle: {e}")
            
            print(f"   ✅ Batch {batch_count}: {batch_imported} imported, {batch_duplicates} duplicates")
            
            # Show progress
            progress = ((i + len(batch)) / len(all_candles)) * 100
            print(f"   📈 Progress: {progress:.1f}%")
        
        # Final summary
        print(f"\n📊 FINAL IMPORT RESULTS:")
        print(f"   Total processed: {len(all_candles)}")
        print(f"   New records: {total_imported}")
        print(f"   Duplicates: {total_duplicates}")
        
        final_count = CoinAPIPrice.objects.filter(coin='GRTUSDT').count()
        new_records = final_count - existing_count
        
        print(f"   Final GRT records: {final_count}")
        print(f"   Net new records: {new_records}")
        
        if final_count > 0:
            latest = CoinAPIPrice.objects.filter(coin='GRTUSDT').order_by('-timestamp').first()
            earliest = CoinAPIPrice.objects.filter(coin='GRTUSDT').order_by('timestamp').first()
            
            print(f"\n📅 FINAL DATA RANGE:")
            print(f"   Earliest: {earliest.timestamp}")
            print(f"   Latest: {latest.timestamp}")
            print(f"   Total period: {(latest.timestamp - earliest.timestamp).days} days")
        
        print(f"\n🎯 GRT import complete! Ready for trading.")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = batch_import_grt()
    if success:
        print("\n🎉 SUCCESS!")
    else:
        print("\n❌ FAILED!")