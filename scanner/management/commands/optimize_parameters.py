from django.core.management.base import BaseCommand
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import itertools
from decimal import Decimal
import json

class Command(BaseCommand):
    help = 'Systematically test different TP/SL/leverage combinations to find optimal parameters'

    def add_arguments(self, parser):
        parser.add_argument('--model', type=str, default='short_three', 
                          help='Model to optimize (short_one, short_two, short_three)')
        parser.add_argument('--test-months', type=int, default=2,
                          help='Number of months of recent data to test on')

    def handle(self, *args, **options):
        model_name = options['model']
        test_months = options['test_months']
        
        # Parameter combinations to test
        PARAMETER_CONFIGS = [
            {"tp": 1.5, "sl": 0.75, "leverage": 15},  # 2:1 ratio, aggressive
            {"tp": 1.8, "sl": 0.8, "leverage": 20},   # 2.25:1 ratio, very aggressive  
            {"tp": 2.0, "sl": 1.0, "leverage": 15},   # 2:1 ratio, balanced
            {"tp": 2.2, "sl": 1.0, "leverage": 12},   # 2.2:1 ratio, conservative leverage
            {"tp": 2.5, "sl": 1.2, "leverage": 10},   # Current ratio, lower leverage
            {"tp": 2.5, "sl": 1.5, "leverage": 10},   # Current settings
            {"tp": 3.0, "sl": 1.2, "leverage": 12},   # Higher TP, moderate SL
            {"tp": 3.5, "sl": 1.5, "leverage": 8},    # Conservative, high reward
        ]
        
        self.stdout.write(f"🔍 Optimizing parameters for {model_name} model")
        self.stdout.write(f"📊 Testing {len(PARAMETER_CONFIGS)} configurations")
        
        # Load predictions data
        predictions_file = f"{model_name}_enhanced_predictions.csv"
        try:
            df = pd.read_csv(predictions_file)
            self.stdout.write(f"✅ Loaded {len(df)} predictions from {predictions_file}")
        except FileNotFoundError:
            self.stderr.write(f"❌ Could not find {predictions_file}")
            return
        
        # Prepare data
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Use only recent data for testing
        cutoff_date = df['timestamp'].max() - timedelta(days=30 * test_months)
        test_df = df[df['timestamp'] >= cutoff_date].copy()
        
        self.stdout.write(f"📅 Testing on {len(test_df)} predictions from {cutoff_date.date()}")
        
        results = []
        
        for i, config in enumerate(PARAMETER_CONFIGS):
            self.stdout.write(f"\n🧪 Testing config {i+1}/{len(PARAMETER_CONFIGS)}: {config}")
            
            # Run simulation with this configuration
            performance = self.simulate_trading(test_df, config)
            performance['config'] = config
            results.append(performance)
            
            # Print immediate results
            self.stdout.write(f"   📈 Win Rate: {performance['win_rate']:.1f}%")
            self.stdout.write(f"   💰 Total PnL: {performance['total_pnl']:.2f}%")
            self.stdout.write(f"   📊 Profit Factor: {performance['profit_factor']:.2f}")
            self.stdout.write(f"   📉 Max Drawdown: {performance['max_drawdown']:.2f}%")
        
        # Analyze and rank results
        self.analyze_results(results, model_name)

    def simulate_trading(self, df, config):
        """Simulate trading with given TP/SL/leverage configuration"""
        tp_pct = config['tp'] / 100
        sl_pct = config['sl'] / 100
        leverage = config['leverage']
        
        # Filter for high confidence predictions only
        high_confidence = df[df['prediction_prob'] > 0.65].copy()
        
        trades = []
        account_balance = 10000  # Starting balance
        peak_balance = account_balance
        max_drawdown = 0
        
        for _, row in high_confidence.iterrows():
            if row['prediction'] == 1:  # Model predicts profitable trade
                # Calculate trade outcome
                entry_price = row['close']
                
                # Simulate price movement for next 24 periods (2 hours)
                # For now, use simplified logic - in real implementation, 
                # you'd look ahead in the data
                
                # Simplified outcome based on whether prediction was correct
                if row.get('correct', 0) == 1:
                    # Successful trade - hit TP
                    pnl_pct = tp_pct * leverage
                    exit_reason = 'take_profit'
                else:
                    # Failed trade - hit SL  
                    pnl_pct = -sl_pct * leverage
                    exit_reason = 'stop_loss'
                
                # Apply PnL to account
                trade_amount = account_balance * 0.1  # Risk 10% per trade
                pnl_amount = trade_amount * pnl_pct
                account_balance += pnl_amount
                
                # Track drawdown
                if account_balance > peak_balance:
                    peak_balance = account_balance
                
                current_drawdown = (peak_balance - account_balance) / peak_balance * 100
                max_drawdown = max(max_drawdown, current_drawdown)
                
                trades.append({
                    'timestamp': row['timestamp'],
                    'entry_price': entry_price,
                    'pnl_pct': pnl_pct * 100,
                    'pnl_amount': pnl_amount,
                    'exit_reason': exit_reason,
                    'balance': account_balance,
                    'confidence': row['prediction_prob']
                })
        
        # Calculate performance metrics
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'avg_win': 0,
                'avg_loss': 0
            }
        
        trades_df = pd.DataFrame(trades)
        
        winning_trades = trades_df[trades_df['pnl_amount'] > 0]
        losing_trades = trades_df[trades_df['pnl_amount'] < 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100
        total_pnl = (account_balance - 10000) / 10000 * 100
        
        avg_win = winning_trades['pnl_amount'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['pnl_amount'].mean()) if len(losing_trades) > 0 else 1
        
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        return {
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_balance': account_balance
        }

    def analyze_results(self, results, model_name):
        """Analyze and rank parameter configurations"""
        self.stdout.write("\n" + "="*80)
        self.stdout.write("📊 PARAMETER OPTIMIZATION RESULTS")
        self.stdout.write("="*80)
        
        # Sort by total PnL (primary metric)
        results_sorted = sorted(results, key=lambda x: x['total_pnl'], reverse=True)
        
        self.stdout.write("\n🏆 TOP CONFIGURATIONS (by Total PnL):")
        self.stdout.write("-" * 80)
        
        for i, result in enumerate(results_sorted[:5]):
            config = result['config']
            self.stdout.write(f"\n#{i+1} - TP: {config['tp']}% | SL: {config['sl']}% | Leverage: {config['leverage']}x")
            self.stdout.write(f"     💰 Total PnL: {result['total_pnl']:.2f}%")
            self.stdout.write(f"     📈 Win Rate: {result['win_rate']:.1f}%")
            self.stdout.write(f"     📊 Profit Factor: {result['profit_factor']:.2f}")
            self.stdout.write(f"     📉 Max Drawdown: {result['max_drawdown']:.2f}%")
            self.stdout.write(f"     🎯 Total Trades: {result['total_trades']}")
        
        # Sort by profit factor (risk-adjusted returns)
        self.stdout.write("\n\n🎯 TOP CONFIGURATIONS (by Profit Factor):")
        self.stdout.write("-" * 80)
        
        results_by_pf = sorted(results, key=lambda x: x['profit_factor'], reverse=True)
        
        for i, result in enumerate(results_by_pf[:3]):
            config = result['config']
            self.stdout.write(f"\n#{i+1} - TP: {config['tp']}% | SL: {config['sl']}% | Leverage: {config['leverage']}x")
            self.stdout.write(f"     📊 Profit Factor: {result['profit_factor']:.2f}")
            self.stdout.write(f"     💰 Total PnL: {result['total_pnl']:.2f}%")
            self.stdout.write(f"     📈 Win Rate: {result['win_rate']:.1f}%")
        
        # Sort by win rate
        self.stdout.write("\n\n🎪 TOP CONFIGURATIONS (by Win Rate):")
        self.stdout.write("-" * 80)
        
        results_by_wr = sorted(results, key=lambda x: x['win_rate'], reverse=True)
        
        for i, result in enumerate(results_by_wr[:3]):
            config = result['config']
            self.stdout.write(f"\n#{i+1} - TP: {config['tp']}% | SL: {config['sl']}% | Leverage: {config['leverage']}x")
            self.stdout.write(f"     📈 Win Rate: {result['win_rate']:.1f}%")
            self.stdout.write(f"     💰 Total PnL: {result['total_pnl']:.2f}%")
            self.stdout.write(f"     📊 Profit Factor: {result['profit_factor']:.2f}")
        
        # Save detailed results
        results_file = f"optimization_results_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to JSON-serializable format
        for result in results:
            result['timestamp'] = datetime.now().isoformat()
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.stdout.write(f"\n💾 Detailed results saved to: {results_file}")
        
        # Recommendations
        best_overall = results_sorted[0]
        best_config = best_overall['config']
        
        self.stdout.write("\n" + "="*80)
        self.stdout.write("🎯 RECOMMENDATIONS")
        self.stdout.write("="*80)
        self.stdout.write(f"\n🏆 OPTIMAL CONFIGURATION:")
        self.stdout.write(f"   Take Profit: {best_config['tp']}%")
        self.stdout.write(f"   Stop Loss: {best_config['sl']}%")
        self.stdout.write(f"   Leverage: {best_config['leverage']}x")
        self.stdout.write(f"   Expected PnL: {best_overall['total_pnl']:.2f}%")
        self.stdout.write(f"   Expected Win Rate: {best_overall['win_rate']:.1f}%")
        
        self.stdout.write(f"\n📝 Next steps:")
        self.stdout.write(f"   1. Update {model_name}_trade_simulator.py with optimal parameters")
        self.stdout.write(f"   2. Update live_pipeline.py with optimal parameters")
        self.stdout.write(f"   3. Run extended backtest to validate results")
        
        self.stdout.write("\n✅ Parameter optimization complete!")