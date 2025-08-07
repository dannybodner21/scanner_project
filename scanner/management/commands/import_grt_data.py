#!/usr/bin/env python3
"""
Django Management Command to import GRT data from JSON file
Usage: python manage.py import_grt_data --file grt_3months_data.json
"""

import json
import pandas as pd
from datetime import datetime

from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice

class Command(BaseCommand):
    help = 'Import GRT historical data from JSON file'

    def add_arguments(self, parser):
        parser.add_argument(
            '--file',
            type=str,
            required=True,
            help='JSON file containing GRT candle data'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be imported without actually saving'
        )

    def handle(self, *args, **options):
        json_file = options['file']
        dry_run = options['dry_run']
        
        self.stdout.write(self.style.SUCCESS("🚀 GRT DATA IMPORT FROM JSON"))
        self.stdout.write("=" * 60)
        
        try:
            # Load JSON data
            self.stdout.write(f"📂 Loading data from {json_file}...")
            with open(json_file, 'r') as f:
                candles_data = json.load(f)
            
            self.stdout.write(f"✅ Loaded {len(candles_data)} candles from file")
            
            # Check existing data
            existing_count = CoinAPIPrice.objects.filter(coin='GRTUSDT').count()
            self.stdout.write(f"📊 Existing GRT records in database: {existing_count}")
            
            if existing_count > 0:
                latest_record = CoinAPIPrice.objects.filter(coin='GRTUSDT').order_by('-timestamp').first()
                earliest_record = CoinAPIPrice.objects.filter(coin='GRTUSDT').order_by('timestamp').first()
                self.stdout.write(f"📊 Existing data range: {earliest_record.timestamp.date()} to {latest_record.timestamp.date()}")
            
            # Process and import data
            imported_count = 0
            duplicate_count = 0
            error_count = 0
            
            self.stdout.write(f"\n🔄 Processing {len(candles_data)} candles...")
            if dry_run:
                self.stdout.write(self.style.WARNING("🔍 DRY RUN MODE - No data will be saved"))
            
            for i, candle_data in enumerate(candles_data):
                try:
                    # Parse timestamp
                    timestamp = pd.to_datetime(candle_data['timestamp'])
                    
                    # Validate data
                    if not all(key in candle_data for key in ['coin', 'timestamp', 'open', 'high', 'low', 'close', 'volume']):
                        self.stdout.write(f"   ❌ Missing required fields in candle {i+1}")
                        error_count += 1
                        continue
                    
                    if not dry_run:
                        # Save to database
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
                            imported_count += 1
                        else:
                            duplicate_count += 1
                    else:
                        # Dry run - just check if record exists
                        exists = CoinAPIPrice.objects.filter(
                            coin=candle_data['coin'],
                            timestamp=timestamp
                        ).exists()
                        
                        if exists:
                            duplicate_count += 1
                        else:
                            imported_count += 1
                
                except Exception as e:
                    self.stdout.write(f"   ❌ Error processing candle {i+1}: {e}")
                    error_count += 1
                
                # Progress update every 1000 records
                if (i + 1) % 1000 == 0:
                    progress = ((i + 1) / len(candles_data)) * 100
                    self.stdout.write(f"   📈 Progress: {progress:.1f}% ({i+1}/{len(candles_data)})")
            
            # Final summary
            self.stdout.write("\n" + "=" * 60)
            if dry_run:
                self.stdout.write(self.style.SUCCESS("✅ DRY RUN COMPLETE"))
            else:
                self.stdout.write(self.style.SUCCESS("✅ GRT DATA IMPORT COMPLETE"))
            self.stdout.write("=" * 60)
            
            self.stdout.write(f"📊 IMPORT STATISTICS:")
            self.stdout.write(f"   Records processed: {len(candles_data)}")
            if dry_run:
                self.stdout.write(f"   Would import: {imported_count} new records")
                self.stdout.write(f"   Would skip: {duplicate_count} duplicates")
            else:
                self.stdout.write(f"   New records imported: {imported_count}")
                self.stdout.write(f"   Duplicate records: {duplicate_count}")
            self.stdout.write(f"   Errors: {error_count}")
            
            if not dry_run:
                # Final database state
                final_count = CoinAPIPrice.objects.filter(coin='GRTUSDT').count()
                self.stdout.write(f"   Total GRT records in DB: {final_count}")
                
                if final_count > 0:
                    latest = CoinAPIPrice.objects.filter(coin='GRTUSDT').order_by('-timestamp').first()
                    earliest = CoinAPIPrice.objects.filter(coin='GRTUSDT').order_by('timestamp').first()
                    
                    self.stdout.write(f"\n📅 FINAL DATA RANGE:")
                    self.stdout.write(f"   Earliest: {earliest.timestamp}")
                    self.stdout.write(f"   Latest: {latest.timestamp}")
                    self.stdout.write(f"   Total period: {(latest.timestamp - earliest.timestamp).days} days")
                    
                    # Data coverage calculation
                    total_minutes = (latest.timestamp - earliest.timestamp).total_seconds() / 60
                    expected_candles = int(total_minutes / 5) + 1  # Every 5 minutes
                    coverage = (final_count / expected_candles) * 100 if expected_candles > 0 else 0
                    self.stdout.write(f"   Data coverage: {coverage:.1f}% of expected 5-min candles")
            
            self.stdout.write(f"\n💰 COST: $0.00 (Coinbase is FREE!)")
            if not dry_run:
                self.stdout.write(self.style.SUCCESS("🎯 GRT ready for trading with historical data"))
            else:
                self.stdout.write(self.style.WARNING("🔍 Run without --dry-run to actually import the data"))
            
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR(f"❌ File not found: {json_file}"))
        except json.JSONDecodeError as e:
            self.stdout.write(self.style.ERROR(f"❌ Invalid JSON file: {e}"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ Import failed: {e}"))
            import traceback
            traceback.print_exc()