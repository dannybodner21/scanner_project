#!/usr/bin/env python3
"""
Check current GRT data status in database
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'scanner_project.settings')
sys.path.append('/Users/danielbodner/Desktop/scanner_project')
django.setup()

from scanner.models import CoinAPIPrice

def check_grt_status():
    """Check current GRT data in database"""
    
    print("🔸 GRT DATABASE STATUS CHECK")
    print("=" * 40)
    
    try:
        # Count GRT records
        grt_count = CoinAPIPrice.objects.filter(coin='GRTUSDT').count()
        print(f"📊 Total GRT records: {grt_count}")
        
        if grt_count > 0:
            # Get date range
            latest = CoinAPIPrice.objects.filter(coin='GRTUSDT').order_by('-timestamp').first()
            earliest = CoinAPIPrice.objects.filter(coin='GRTUSDT').order_by('timestamp').first()
            
            print(f"📅 Date range:")
            print(f"   Earliest: {earliest.timestamp}")
            print(f"   Latest: {latest.timestamp}")
            print(f"   Total days: {(latest.timestamp - earliest.timestamp).days}")
            
            # Sample recent data
            recent_records = CoinAPIPrice.objects.filter(coin='GRTUSDT').order_by('-timestamp')[:5]
            print(f"\n📈 Recent records (last 5):")
            for i, record in enumerate(recent_records):
                print(f"   {i+1}. {record.timestamp} | ${record.close:.4f}")
            
            # Check data density
            total_minutes = (latest.timestamp - earliest.timestamp).total_seconds() / 60
            expected_candles = int(total_minutes / 5)  # Every 5 minutes
            coverage = (grt_count / expected_candles) * 100 if expected_candles > 0 else 0
            print(f"\n📊 Data coverage: {coverage:.1f}% of expected 5-min candles")
            print(f"   Expected: {expected_candles} candles")
            print(f"   Actual: {grt_count} candles")
            print(f"   Missing: {max(0, expected_candles - grt_count)} candles")
        
        else:
            print("❌ No GRT data found in database")
        
        # Check other coins for comparison
        print(f"\n🔍 Other coin data for comparison:")
        other_coins = CoinAPIPrice.objects.values('coin').distinct()
        for coin_record in other_coins:
            coin = coin_record['coin']
            count = CoinAPIPrice.objects.filter(coin=coin).count()
            print(f"   {coin}: {count} records")
        
        return grt_count > 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    check_grt_status()