#!/usr/bin/env python3
"""
Backfill GRT Historical Data from Coinbase
Fetches 5-minute candles from January 1, 2022 to present
Stores in CoinAPIPrice model with coin as GRTUSDT
"""

import os
import sys
import django
import requests
import pandas as pd
import pytz
from datetime import datetime, timedelta
import time

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'scanner.settings')
django.setup()

from scanner.models import CoinAPIPrice

def fetch_coinbase_historical_batch(start_time, end_time):
    """
    Fetch historical candles from Coinbase for a specific time range
    Coinbase returns max 300 candles per request
    """
    try:
        # Convert to Unix timestamps
        start_unix = int(start_time.timestamp())
        end_unix = int(end_time.timestamp())
        
        url = 'https://api.exchange.coinbase.com/products/GRT-USD/candles'
        params = {
            'start': start_time.isoformat(),
            'end': end_time.isoformat(),
            'granularity': 300  # 5 minutes
        }
        
        print(f"📊 Fetching {start_time.date()} to {end_time.date()}...")
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            print(f"   ⚠️ No data returned for {start_time.date()}")
            return []
        
        # Convert Coinbase format to our format
        # Coinbase: [timestamp, low, high, open, close, volume]
        candles = []
        for candle_data in data:
            candles.append({
                'coin': 'GRTUSDT',
                'timestamp': pd.to_datetime(candle_data[0], unit='s', utc=True),
                'open': float(candle_data[3]),
                'high': float(candle_data[2]),
                'low': float(candle_data[1]),
                'close': float(candle_data[4]),
                'volume': float(candle_data[5])
            })
        
        print(f"   ✅ Fetched {len(candles)} candles")
        return candles
        
    except Exception as e:
        print(f"   ❌ Error fetching {start_time.date()}: {e}")
        return []

def save_candles_to_db(candles):
    """Save candles to database, avoiding duplicates"""
    
    saved_count = 0
    duplicate_count = 0
    
    for candle in candles:
        try:
            obj, created = CoinAPIPrice.objects.update_or_create(
                coin=candle['coin'],
                timestamp=candle['timestamp'],
                defaults={
                    'open': candle['open'],
                    'high': candle['high'],
                    'low': candle['low'],
                    'close': candle['close'],
                    'volume': candle['volume']
                }
            )
            
            if created:
                saved_count += 1
            else:
                duplicate_count += 1
                
        except Exception as e:
            print(f"   ❌ Error saving candle {candle['timestamp']}: {e}")
    
    return saved_count, duplicate_count

def backfill_grt_historical():
    """Main backfill function"""
    
    print("🚀 GRT HISTORICAL BACKFILL FROM COINBASE")
    print("=" * 60)
    
    # Define date range
    start_date = datetime(2022, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime.now(pytz.UTC)
    
    print(f"📅 Backfill period: {start_date.date()} to {end_date.date()}")
    print(f"🕐 Total period: {(end_date - start_date).days} days")
    
    # Check existing data
    existing_count = CoinAPIPrice.objects.filter(coin='GRTUSDT').count()
    print(f"📊 Existing GRT records in database: {existing_count}")
    
    if existing_count > 0:
        latest_record = CoinAPIPrice.objects.filter(coin='GRTUSDT').order_by('-timestamp').first()
        earliest_record = CoinAPIPrice.objects.filter(coin='GRTUSDT').order_by('timestamp').first()
        print(f"📊 Existing data range: {earliest_record.timestamp.date()} to {latest_record.timestamp.date()}")
    
    # Coinbase returns max 300 candles per request (25 hours for 5-min candles)
    # We'll fetch in 24-hour chunks to be safe
    chunk_hours = 24
    current_date = start_date
    
    total_fetched = 0
    total_saved = 0
    total_duplicates = 0
    batch_count = 0
    
    print(f"\n🔄 Starting backfill in {chunk_hours}-hour chunks...")
    print("=" * 60)
    
    while current_date < end_date:
        batch_count += 1
        chunk_end = min(current_date + timedelta(hours=chunk_hours), end_date)
        
        print(f"\nBatch {batch_count}: {current_date.strftime('%Y-%m-%d %H:%M')} to {chunk_end.strftime('%Y-%m-%d %H:%M')}")
        
        # Fetch data for this chunk
        candles = fetch_coinbase_historical_batch(current_date, chunk_end)
        
        if candles:
            total_fetched += len(candles)
            
            # Save to database
            saved, duplicates = save_candles_to_db(candles)
            total_saved += saved
            total_duplicates += duplicates
            
            print(f"   💾 Saved: {saved} new records, {duplicates} duplicates")
        
        # Move to next chunk
        current_date = chunk_end
        
        # Rate limiting - be nice to Coinbase
        time.sleep(1)
        
        # Progress update every 10 batches
        if batch_count % 10 == 0:
            progress = ((current_date - start_date).total_seconds() / (end_date - start_date).total_seconds()) * 100
            print(f"\n📈 Progress: {progress:.1f}% complete")
            print(f"📊 Summary so far: {total_fetched} fetched, {total_saved} saved, {total_duplicates} duplicates")
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ GRT HISTORICAL BACKFILL COMPLETE")
    print("=" * 60)
    
    final_count = CoinAPIPrice.objects.filter(coin='GRTUSDT').count()
    new_records = final_count - existing_count
    
    print(f"📊 FINAL STATISTICS:")
    print(f"   Total candles fetched: {total_fetched}")
    print(f"   New records saved: {total_saved}")
    print(f"   Duplicate records: {total_duplicates}")
    print(f"   Total GRT records in DB: {final_count}")
    print(f"   Net new records: {new_records}")
    
    if final_count > 0:
        latest = CoinAPIPrice.objects.filter(coin='GRTUSDT').order_by('-timestamp').first()
        earliest = CoinAPIPrice.objects.filter(coin='GRTUSDT').order_by('timestamp').first()
        
        print(f"\n📅 FINAL DATA RANGE:")
        print(f"   Earliest: {earliest.timestamp}")
        print(f"   Latest: {latest.timestamp}")
        print(f"   Total period: {(latest.timestamp - earliest.timestamp).days} days")
        
        # Calculate expected vs actual records
        expected_candles = ((latest.timestamp - earliest.timestamp).total_seconds() / 300) + 1
        coverage = (final_count / expected_candles) * 100
        print(f"   Data coverage: {coverage:.1f}% of expected candles")
    
    print(f"\n💰 COST: $0.00 (Coinbase is FREE!)")
    print(f"🎯 GRT ready for trading with full historical data")

if __name__ == "__main__":
    try:
        backfill_grt_historical()
    except KeyboardInterrupt:
        print("\n⚠️ Backfill interrupted by user")
        print("📊 Partial data may have been saved")
    except Exception as e:
        print(f"\n❌ Backfill failed: {e}")
        print("📞 Check your internet connection and try again")