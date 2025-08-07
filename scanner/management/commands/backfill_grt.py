#!/usr/bin/env python3
"""
Django Management Command to backfill GRT historical data from Coinbase
Usage: python manage.py backfill_grt
"""

import requests
import pandas as pd
import pytz
from datetime import datetime, timedelta
import time

from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice

class Command(BaseCommand):
    help = 'Backfill GRT historical data from Coinbase (Jan 1, 2022 to present)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--start-date',
            type=str,
            default='2022-01-01',
            help='Start date (YYYY-MM-DD) - default: 2022-01-01'
        )
        parser.add_argument(
            '--chunk-hours',
            type=int,
            default=24,
            help='Hours per batch - default: 24'
        )
        parser.add_argument(
            '--delay',
            type=float,
            default=1.0,
            help='Delay between requests in seconds - default: 1.0'
        )

    def fetch_coinbase_historical_batch(self, start_time, end_time):
        """Fetch historical candles from Coinbase for a specific time range"""
        try:
            url = 'https://api.exchange.coinbase.com/products/GRT-USD/candles'
            params = {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'granularity': 300  # 5 minutes
            }
            
            self.stdout.write(f"📊 Fetching {start_time.date()} to {end_time.date()}...")
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                self.stdout.write(f"   ⚠️ No data returned for {start_time.date()}")
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
            
            self.stdout.write(f"   ✅ Fetched {len(candles)} candles")
            return candles
            
        except Exception as e:
            self.stdout.write(f"   ❌ Error fetching {start_time.date()}: {e}")
            return []

    def save_candles_to_db(self, candles):
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
                self.stdout.write(f"   ❌ Error saving candle {candle['timestamp']}: {e}")
        
        return saved_count, duplicate_count

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("🚀 GRT HISTORICAL BACKFILL FROM COINBASE"))
        self.stdout.write("=" * 60)
        
        # Parse start date
        start_date = datetime.strptime(options['start_date'], '%Y-%m-%d').replace(tzinfo=pytz.UTC)
        end_date = datetime.now(pytz.UTC)
        chunk_hours = options['chunk_hours']
        delay = options['delay']
        
        self.stdout.write(f"📅 Backfill period: {start_date.date()} to {end_date.date()}")
        self.stdout.write(f"🕐 Total period: {(end_date - start_date).days} days")
        self.stdout.write(f"⏱️ Chunk size: {chunk_hours} hours")
        self.stdout.write(f"🕰️ Delay between requests: {delay} seconds")
        
        # Check existing data
        existing_count = CoinAPIPrice.objects.filter(coin='GRTUSDT').count()
        self.stdout.write(f"📊 Existing GRT records in database: {existing_count}")
        
        if existing_count > 0:
            latest_record = CoinAPIPrice.objects.filter(coin='GRTUSDT').order_by('-timestamp').first()
            earliest_record = CoinAPIPrice.objects.filter(coin='GRTUSDT').order_by('timestamp').first()
            self.stdout.write(f"📊 Existing data range: {earliest_record.timestamp.date()} to {latest_record.timestamp.date()}")
        
        current_date = start_date
        total_fetched = 0
        total_saved = 0
        total_duplicates = 0
        batch_count = 0
        
        self.stdout.write(f"\n🔄 Starting backfill in {chunk_hours}-hour chunks...")
        self.stdout.write("=" * 60)
        
        try:
            while current_date < end_date:
                batch_count += 1
                chunk_end = min(current_date + timedelta(hours=chunk_hours), end_date)
                
                self.stdout.write(f"\nBatch {batch_count}: {current_date.strftime('%Y-%m-%d %H:%M')} to {chunk_end.strftime('%Y-%m-%d %H:%M')}")
                
                # Fetch data for this chunk
                candles = self.fetch_coinbase_historical_batch(current_date, chunk_end)
                
                if candles:
                    total_fetched += len(candles)
                    
                    # Save to database
                    saved, duplicates = self.save_candles_to_db(candles)
                    total_saved += saved
                    total_duplicates += duplicates
                    
                    self.stdout.write(f"   💾 Saved: {saved} new records, {duplicates} duplicates")
                
                # Move to next chunk
                current_date = chunk_end
                
                # Rate limiting
                if current_date < end_date:  # Don't sleep after last batch
                    time.sleep(delay)
                
                # Progress update every 10 batches
                if batch_count % 10 == 0:
                    progress = ((current_date - start_date).total_seconds() / (end_date - start_date).total_seconds()) * 100
                    self.stdout.write(f"\n📈 Progress: {progress:.1f}% complete")
                    self.stdout.write(f"📊 Summary so far: {total_fetched} fetched, {total_saved} saved, {total_duplicates} duplicates")
        
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("\n⚠️ Backfill interrupted by user"))
            self.stdout.write("📊 Partial data may have been saved")
            return
        
        # Final summary
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(self.style.SUCCESS("✅ GRT HISTORICAL BACKFILL COMPLETE"))
        self.stdout.write("=" * 60)
        
        final_count = CoinAPIPrice.objects.filter(coin='GRTUSDT').count()
        new_records = final_count - existing_count
        
        self.stdout.write(f"📊 FINAL STATISTICS:")
        self.stdout.write(f"   Total batches processed: {batch_count}")
        self.stdout.write(f"   Total candles fetched: {total_fetched}")
        self.stdout.write(f"   New records saved: {total_saved}")
        self.stdout.write(f"   Duplicate records: {total_duplicates}")
        self.stdout.write(f"   Total GRT records in DB: {final_count}")
        self.stdout.write(f"   Net new records: {new_records}")
        
        if final_count > 0:
            latest = CoinAPIPrice.objects.filter(coin='GRTUSDT').order_by('-timestamp').first()
            earliest = CoinAPIPrice.objects.filter(coin='GRTUSDT').order_by('timestamp').first()
            
            self.stdout.write(f"\n📅 FINAL DATA RANGE:")
            self.stdout.write(f"   Earliest: {earliest.timestamp}")
            self.stdout.write(f"   Latest: {latest.timestamp}")
            self.stdout.write(f"   Total period: {(latest.timestamp - earliest.timestamp).days} days")
            
            # Calculate expected vs actual records
            total_minutes = (latest.timestamp - earliest.timestamp).total_seconds() / 60
            expected_candles = int(total_minutes / 5) + 1  # Every 5 minutes
            coverage = (final_count / expected_candles) * 100 if expected_candles > 0 else 0
            self.stdout.write(f"   Data coverage: {coverage:.1f}% of expected 5-min candles")
        
        self.stdout.write(f"\n💰 COST: $0.00 (Coinbase is FREE!)")
        self.stdout.write(self.style.SUCCESS("🎯 GRT ready for trading with full historical data"))