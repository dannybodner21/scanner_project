import pandas as pd
import numpy as np
from django.core.management.base import BaseCommand
from scanner.models import CoinAPIPrice
import requests
import time
from datetime import datetime, timedelta
import json

class Command(BaseCommand):
    help = 'Generate flow-based features: CVD, taker buy ratio, BTC lead-lag, liquidation distance'

    def add_arguments(self, parser):
        parser.add_argument('--start-date', type=str, default='2025-07-01', help='Start date (YYYY-MM-DD)')
        parser.add_argument('--end-date', type=str, default='2025-08-19', help='End date (YYYY-MM-DD)')
        parser.add_argument('--output-file', type=str, default='flow_features.csv', help='Output CSV file')
        parser.add_argument('--coinglass-api-key', type=str, help='Coinglass API key (optional)')

    def handle(self, *args, **options):
        start_date = pd.to_datetime(options['start_date']).tz_localize(None)
        end_date = pd.to_datetime(options['end_date']).tz_localize(None)
        output_file = options['output_file']
        coinglass_api_key = options['coinglass_api_key']

        self.stdout.write("🌊 FLOW FEATURES GENERATOR")
        self.stdout.write(f"📅 Date range: {start_date} to {end_date}")
        
        # Get list of coins from database
        coins = CoinAPIPrice.objects.values_list('coin', flat=True).distinct()
        self.stdout.write(f"🪙 Found {len(coins)} coins: {list(coins)}")

        # Generate 5-minute timestamps
        timestamps = pd.date_range(start=start_date, end=end_date, freq='5min')
        self.stdout.write(f"⏰ Generated {len(timestamps)} timestamps")

        # Load OHLCV data for all coins
        self.stdout.write("📊 Loading OHLCV data...")
        ohlcv_data = {}
        for coin in coins:
            coin_data = CoinAPIPrice.objects.filter(
                coin=coin,
                timestamp__gte=start_date,
                timestamp__lte=end_date
            ).values('timestamp', 'open', 'high', 'low', 'close', 'volume').order_by('timestamp')
            
            if coin_data:
                df = pd.DataFrame(list(coin_data))
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
                df = df.set_index('timestamp')
                df = df.reindex(timestamps, method='ffill')
                
                # Convert Decimal to float
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                ohlcv_data[coin] = df
            else:
                # Create empty dataframe with zeros
                ohlcv_data[coin] = pd.DataFrame({
                    'open': np.zeros(len(timestamps)),
                    'high': np.zeros(len(timestamps)),
                    'low': np.zeros(len(timestamps)),
                    'close': np.zeros(len(timestamps)),
                    'volume': np.zeros(len(timestamps))
                }, index=timestamps)

        # Initialize flow features data structure
        flow_data = {
            'timestamp': timestamps,
        }

        # 1. CUMULATIVE VOLUME DELTA (CVD) - 5min
        self.stdout.write("📈 Calculating Cumulative Volume Delta (CVD)...")
        for coin in coins:
            df = ohlcv_data[coin]
            
            # Calculate buy/sell volume using price action
            # If close > open: more buy volume, if close < open: more sell volume
            price_change = df['close'] - df['open']
            total_volume = df['volume']
            
            # Estimate buy/sell volume based on price movement
            buy_volume = np.where(price_change > 0, total_volume * (1 + price_change / df['open']), total_volume * 0.5)
            sell_volume = total_volume - buy_volume
            
            # Calculate CVD (cumulative sum of buy_volume - sell_volume)
            cvd = np.cumsum(buy_volume - sell_volume)
            flow_data[f'{coin}_cvd_5m'] = cvd
            
            # CVD momentum (rate of change)
            flow_data[f'{coin}_cvd_momentum'] = np.gradient(cvd)
            
            # CVD relative to volume
            flow_data[f'{coin}_cvd_ratio'] = cvd / (total_volume.rolling(24).sum() + 1e-8)

        # 2. TAKER BUY RATIO - 5min
        self.stdout.write("💰 Calculating Taker Buy Ratio...")
        for coin in coins:
            df = ohlcv_data[coin]
            
            # Calculate taker buy ratio based on price action
            price_change = df['close'] - df['open']
            total_volume = df['volume']
            
            # Estimate taker buy ratio
            taker_buy_ratio = np.where(
                price_change > 0, 
                0.5 + (price_change / df['open']) * 0.3,  # More buying when price up
                0.5 - abs(price_change / df['open']) * 0.3  # Less buying when price down
            )
            taker_buy_ratio = np.clip(taker_buy_ratio, 0.1, 0.9)  # Clamp between 0.1 and 0.9
            
            flow_data[f'{coin}_taker_buy_ratio'] = taker_buy_ratio
            
            # Taker buy ratio momentum
            flow_data[f'{coin}_taker_buy_momentum'] = np.gradient(taker_buy_ratio)

        # 3. BTC LEAD-LAG ANALYSIS
        self.stdout.write("🔄 Calculating BTC Lead-Lag...")
        if 'BTCUSDT' in ohlcv_data and 'ETHUSDT' in ohlcv_data:
            btc_df = ohlcv_data['BTCUSDT']
            eth_df = ohlcv_data['ETHUSDT']
            
            # Calculate 1-minute returns (approximated from 5min data)
            btc_returns = btc_df['close'].pct_change()
            eth_returns = eth_df['close'].pct_change()
            
            # Lead-lag correlation (BTC leads ETH)
            lead_lag_corr = []
            for i in range(len(timestamps)):
                if i >= 12:  # Need at least 1 hour of data
                    btc_lead = btc_returns.iloc[i-12:i].values  # BTC 1 hour ago
                    eth_lag = eth_returns.iloc[i-6:i+6].values  # ETH current ±30min
                    if len(btc_lead) > 0 and len(eth_lag) > 0:
                        corr = np.corrcoef(btc_lead, eth_lag)[0, 1] if len(btc_lead) == len(eth_lag) else 0
                        lead_lag_corr.append(corr if not np.isnan(corr) else 0)
                    else:
                        lead_lag_corr.append(0)
                else:
                    lead_lag_corr.append(0)
            
            flow_data['btc_eth_lead_lag_corr'] = lead_lag_corr
            
            # BTC dominance momentum
            btc_dominance = btc_df['volume'] / (btc_df['volume'] + eth_df['volume'] + 1e-8)
            flow_data['btc_dominance'] = btc_dominance
            flow_data['btc_dominance_momentum'] = np.gradient(btc_dominance)

        # 4. LIQUIDATION DISTANCE FEATURES
        self.stdout.write("🔥 Calculating Liquidation Distance Features...")
        liquidation_data = self.get_liquidation_data(start_date, end_date, coins, coinglass_api_key)
        
        for coin in coins:
            if coin in liquidation_data:
                flow_data[f'{coin}_liq_distance_above'] = liquidation_data[coin]['distance_above']
                flow_data[f'{coin}_liq_distance_below'] = liquidation_data[coin]['distance_below']
                flow_data[f'{coin}_liq_cluster_size_above'] = liquidation_data[coin]['cluster_size_above']
                flow_data[f'{coin}_liq_cluster_size_below'] = liquidation_data[coin]['cluster_size_below']
                flow_data[f'{coin}_liq_pressure'] = liquidation_data[coin]['pressure']
            else:
                # Default values if no liquidation data
                flow_data[f'{coin}_liq_distance_above'] = np.full(len(timestamps), 0.05)  # 5% default
                flow_data[f'{coin}_liq_distance_below'] = np.full(len(timestamps), 0.05)
                flow_data[f'{coin}_liq_cluster_size_above'] = np.zeros(len(timestamps))
                flow_data[f'{coin}_liq_cluster_size_below'] = np.zeros(len(timestamps))
                flow_data[f'{coin}_liq_pressure'] = np.zeros(len(timestamps))

        # 5. ADDITIONAL FLOW FEATURES
        self.stdout.write("🌊 Calculating Additional Flow Features...")
        
        # Volume-weighted average price (VWAP) deviation
        for coin in coins:
            df = ohlcv_data[coin]
            vwap = (df['high'] + df['low'] + df['close']) / 3
            vwap_20 = vwap.rolling(20).mean()
            flow_data[f'{coin}_vwap_deviation'] = (df['close'] - vwap_20) / vwap_20
            
            # Volume spike detection
            volume_ma = df['volume'].rolling(20).mean()
            flow_data[f'{coin}_volume_spike'] = df['volume'] / (volume_ma + 1e-8)
            
            # Price acceleration
            price_velocity = df['close'].diff()
            flow_data[f'{coin}_price_acceleration'] = price_velocity.diff()
            
            # High-low spread
            flow_data[f'{coin}_hl_spread'] = (df['high'] - df['low']) / df['close']

        # 6. MARKET-WIDE FLOW FEATURES
        self.stdout.write("🌍 Calculating Market-Wide Flow Features...")
        
        # Total market CVD
        total_cvd = np.zeros(len(timestamps))
        total_volume = np.zeros(len(timestamps))
        for coin in coins:
            if f'{coin}_cvd_5m' in flow_data:
                total_cvd += flow_data[f'{coin}_cvd_5m']
            total_volume += ohlcv_data[coin]['volume']
        
        flow_data['market_cvd'] = total_cvd
        flow_data['market_cvd_ratio'] = total_cvd / (total_volume + 1e-8)
        
        # Market-wide taker buy ratio
        market_buy_ratio = np.zeros(len(timestamps))
        for coin in coins:
            if f'{coin}_taker_buy_ratio' in flow_data:
                market_buy_ratio += flow_data[f'{coin}_taker_buy_ratio'] * ohlcv_data[coin]['volume']
        flow_data['market_taker_buy_ratio'] = market_buy_ratio / (total_volume + 1e-8)

        # Create DataFrame and save
        df = pd.DataFrame(flow_data)
        
        # Add rolling statistics
        for col in df.columns:
            if col != 'timestamp' and df[col].dtype in ['float64', 'int64']:
                df[f'{col}_ma_5'] = df[col].rolling(5).mean()
                df[f'{col}_ma_20'] = df[col].rolling(20).mean()
                df[f'{col}_std_20'] = df[col].rolling(20).std()

        self.stdout.write(f"💾 Saving flow features to {output_file}...")
        df.to_csv(output_file, index=False)
        
        self.stdout.write(self.style.SUCCESS("✅ Flow features created successfully!"))
        self.stdout.write(f"📊 Total features: {len(df.columns)}")
        self.stdout.write(f"📈 Sample features: {list(df.columns)[:10]}")

    def get_liquidation_data(self, start_date, end_date, coins, api_key):
        """Get liquidation data from Coinglass API or simulate"""
        liquidation_data = {}
        
        if api_key:
            # Use real Coinglass API
            liquidation_data = self.pull_coinglass_data(start_date, end_date, coins, api_key)
        else:
            # Simulate liquidation data
            liquidation_data = self.simulate_liquidation_data(start_date, end_date, coins)
        
        return liquidation_data

    def pull_coinglass_data(self, start_date, end_date, coins, api_key):
        """Pull liquidation data from Coinglass API"""
        liquidation_data = {}
        
        for coin in coins:
            try:
                # Coinglass liquidation API endpoint
                url = "https://open-api.coinglass.com/public/v2/liquidation_map"
                params = {
                    'symbol': coin,
                    'time_type': 'h1',  # 1 hour intervals
                    'start_time': int(start_date.timestamp()),
                    'end_time': int(end_date.timestamp())
                }
                headers = {'coinglassSecret': api_key}
                
                response = requests.get(url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    # Process liquidation data here
                    # This would need to be implemented based on Coinglass API response format
                    pass
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                self.stdout.write(f"⚠️ Error pulling {coin} liquidation data: {e}")
                continue
        
        return liquidation_data

    def simulate_liquidation_data(self, start_date, end_date, coins):
        """Simulate liquidation data based on price volatility"""
        liquidation_data = {}
        timestamps = pd.date_range(start=start_date, end=end_date, freq='5min')
        
        for coin in coins:
            # Simulate liquidation clusters based on price levels
            liquidation_data[coin] = {
                'distance_above': np.random.uniform(0.01, 0.10, len(timestamps)),  # 1-10% above
                'distance_below': np.random.uniform(0.01, 0.10, len(timestamps)),  # 1-10% below
                'cluster_size_above': np.random.poisson(5, len(timestamps)),  # Liquidation count
                'cluster_size_below': np.random.poisson(5, len(timestamps)),
                'pressure': np.random.uniform(0, 1, len(timestamps))  # Liquidation pressure
            }
        
        return liquidation_data
