from django.core.management.base import BaseCommand
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

# Remove OpenAI imports - we're going local!
# from openai import OpenAI
# from dotenv import load_dotenv
# import base64
# from PIL import Image
# import io
import time

# ==================== MULTI-AGENT DECISION SYSTEM ====================

class TechnicalAnalysisAgent:
    """Analyzes chart patterns and technical indicators"""
    
    def __init__(self):
        self.confidence_weights = {
            'trend_alignment': 0.3,
            'support_resistance': 0.25,
            'momentum_signals': 0.25,
            'volume_confirmation': 0.2
        }
    
    def analyze(self, features: Dict, recent_data: pd.DataFrame = None) -> float:
        """Analyze technical setup and return confidence score 0-1"""
        score = 0.0
        
        try:
            # Trend alignment - multiple EMA alignment
            if (features.get('ema_9', 0) > features.get('ema_21', 0) > features.get('ema_50', 0)):
                score += 0.3
            elif features.get('ema_9', 0) > features.get('ema_21', 0):
                score += 0.15
                
            # RSI momentum (avoid extremes)
            rsi = features.get('rsi_14', 50)
            if 35 < rsi < 65:
                score += 0.25
            elif 30 < rsi < 70:
                score += 0.15
                
            # MACD bullish
            if features.get('macd_bullish', 0) == 1:
                score += 0.1
                
            # Volume confirmation
            volume_ratio = features.get('volume_ratio', 1.0)
            if volume_ratio > 1.2:
                score += 0.2
            elif volume_ratio > 1.0:
                score += 0.1
                
            # Bollinger Band position (avoid extremes)
            bb_pos = features.get('bb_position', 0.5)
            if 0.2 < bb_pos < 0.8:
                score += 0.15
                
        except Exception as e:
            print(f"⚠️ Technical analysis error: {e}")
            return 0.0
            
        return min(score, 1.0)


class MicrostructureAgent:
    """Analyzes market microstructure and volatility"""
    
    def analyze(self, features: Dict) -> float:
        """Analyze market microstructure and return confidence score 0-1"""
        score = 0.0
        
        try:
            # Volatility regime (sweet spot for day trading)
            atr_percent = features.get('atr_percent', 0)
            if 2 < atr_percent < 6:  # Good volatility for profits
                score += 0.4
            elif 1.5 < atr_percent < 8:  # Acceptable
                score += 0.2
                
            # Bollinger Band squeeze (avoid low volatility)
            if not features.get('bb_squeeze', False):
                score += 0.3
                
            # Market structure - not at resistance
            bb_position = features.get('bb_position', 0.5)
            if bb_position < 0.75:  # Not near upper band resistance
                score += 0.2
                
            # ATR trend (increasing volatility is good)
            atr_14 = features.get('atr_14', 0)
            atr_21 = features.get('atr_21', 0)
            if atr_14 > atr_21 and atr_21 > 0:
                score += 0.1
                
        except Exception as e:
            print(f"⚠️ Microstructure analysis error: {e}")
            return 0.0
            
        return min(score, 1.0)


class RiskAgent:
    """Assesses trade and portfolio risk"""
    
    def __init__(self):
        self.recent_trades = deque(maxlen=20)  # Track last 20 trades
        
    def add_trade_result(self, won: bool):
        """Add trade result to recent history"""
        self.recent_trades.append(1 if won else 0)
        
    def analyze(self, features: Dict, portfolio_state: Dict) -> float:
        """Analyze risk factors and return confidence score 0-1"""
        score = 1.0  # Start at max, reduce for risks
        
        try:
            # Recent performance check
            if len(self.recent_trades) >= 10:
                recent_win_rate = sum(list(self.recent_trades)[-10:]) / 10
                if recent_win_rate < 0.4:  # Model degrading
                    score -= 0.4
                elif recent_win_rate < 0.5:
                    score -= 0.2
                    
            # Time-based risk
            hour = features.get('hour', 12)
            is_weekend = features.get('is_weekend', 0)
            
            if is_weekend:
                score -= 0.2  # Weekend liquidity risk
            if hour in [22, 23, 0, 1, 2, 3]:  # Low liquidity hours
                score -= 0.15
                
            # Volatility risk (too high is dangerous)
            atr_percent = features.get('atr_percent', 0)
            if atr_percent > 10:  # Extremely volatile
                score -= 0.5
            elif atr_percent > 8:  # Very volatile
                score -= 0.3
                
            # RSI extremes risk
            rsi = features.get('rsi_14', 50)
            if rsi > 80:  # Extremely overbought
                score -= 0.3
            elif rsi < 20:  # Extremely oversold
                score -= 0.2
                
            # Portfolio concentration risk
            open_positions = portfolio_state.get('open_positions', 0)
            if open_positions > 3:  # Too many positions
                score -= 0.2
                
        except Exception as e:
            print(f"⚠️ Risk analysis error: {e}")
            return 0.0
            
        return max(score, 0.0)


class RegimeAgent:
    """Detects overall market regime and conditions"""
    
    def analyze(self, features: Dict, btc_context: Dict = None) -> float:
        """Analyze market regime and return confidence score 0-1"""
        score = 0.0
        
        try:
            # BTC trend context (affects all crypto)
            if btc_context:
                if btc_context.get('btc_bull_trend', False):
                    score += 0.4
                elif btc_context.get('btc_strong_trend', False):
                    score += 0.2
                else:
                    score -= 0.1  # BTC bearish = risky for alts
                    
            # Volume regime
            volume_trend = features.get('volume_trend', 0)
            if volume_trend > 0:  # Increasing volume
                score += 0.3
            elif volume_trend > -0.1:  # Stable volume
                score += 0.1
                
            # Volatility percentile (medium is best)
            vol_percentile = features.get('vol_percentile', 0.5)
            if 0.3 < vol_percentile < 0.8:
                score += 0.3
            elif 0.2 < vol_percentile < 0.9:
                score += 0.15
                
        except Exception as e:
            print(f"⚠️ Regime analysis error: {e}")
            return 0.0
            
        return min(score, 1.0)


class MasterDecisionEngine:
    """Coordinates all agents and makes final trading decisions"""
    
    def __init__(self):
        self.technical_agent = TechnicalAnalysisAgent()
        self.microstructure_agent = MicrostructureAgent()
        self.risk_agent = RiskAgent()
        self.regime_agent = RegimeAgent()
        
        # Agent weights (tune these based on backtesting)
        self.agent_weights = {
            'ml_model': 0.35,        # Your LightGBM model
            'technical': 0.25,       # Technical analysis
            'microstructure': 0.20,  # Market microstructure
            'regime': 0.15,          # Market regime
            'risk': 0.05             # Risk assessment (more of a filter)
        }
        
        # Track decisions for analysis
        self.decision_history = []
        
    def make_decision(self, ml_confidence: float, features: Dict, 
                     context_data: Dict = None) -> Tuple[float, str, Dict]:
        """
        Make final trading decision based on all agents
        
        Returns:
            final_confidence: float (0-1)
            decision_reason: str
            agent_scores: dict
        """
        
        if context_data is None:
            context_data = {}
            
        # Get all agent scores
        agent_scores = {
            'ml_model': ml_confidence,
            'technical': self.technical_agent.analyze(features, context_data.get('recent_data')),
            'microstructure': self.microstructure_agent.analyze(features),
            'regime': self.regime_agent.analyze(features, context_data.get('btc_context')),
            'risk': self.risk_agent.analyze(features, context_data.get('portfolio_state', {}))
        }
        
        # Risk agent can veto trades
        if agent_scores['risk'] < 0.3:
            return 0.0, f"Risk veto (risk_score={agent_scores['risk']:.2f})", agent_scores
        
        # Calculate weighted ensemble score
        final_score = sum(agent_scores[agent] * self.agent_weights[agent] 
                         for agent in agent_scores)
        
        # Require agent agreement - reduce confidence if agents disagree
        high_confidence_agents = sum(1 for score in agent_scores.values() if score > 0.6)
        if high_confidence_agents < 3:
            final_score *= 0.8  # Reduce confidence if agents disagree
            
        # Very low scores get heavily penalized
        very_low_agents = sum(1 for score in agent_scores.values() if score < 0.3)
        if very_low_agents > 1:
            final_score *= 0.6
            
        decision_reason = f"Agents: ML={agent_scores['ml_model']:.2f}, " \
                         f"Tech={agent_scores['technical']:.2f}, " \
                         f"Micro={agent_scores['microstructure']:.2f}, " \
                         f"Regime={agent_scores['regime']:.2f}, " \
                         f"Risk={agent_scores['risk']:.2f}"
        
        # Store decision for analysis
        self.decision_history.append({
            'timestamp': datetime.now(),
            'final_score': final_score,
            'agent_scores': agent_scores.copy(),
            'reason': decision_reason
        })
        
        return final_score, decision_reason, agent_scores
    
    def add_trade_result(self, won: bool):
        """Update risk agent with trade result"""
        self.risk_agent.add_trade_result(won)


# ==================== ENHANCED TRADE CLASS ====================

class Trade:
    def __init__(self, coin, entry_time, entry_price, direction, confidence, 
                 trade_id, leverage, agent_scores=None, decision_reason=""):
        self.coin = coin
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.direction = direction
        self.confidence = confidence
        self.trade_id = trade_id
        self.leverage = leverage
        self.agent_scores = agent_scores or {}
        self.decision_reason = decision_reason

        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl_pct = 0.0
        self.pnl = 0.0
        self.duration_minutes = 0

    def close_trade(self, exit_time, exit_price, reason):
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = reason
        self.duration_minutes = (exit_time - self.entry_time).total_seconds() / 60

        if self.direction == 'long':
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price

        self.pnl = self.pnl_pct * self.leverage * 100


# ==================== ENHANCED COMMAND ====================

class Command(BaseCommand):
    help = 'Simulate trading using multi-agent decision system'

    def add_arguments(self, parser):
        parser.add_argument('--predictions-file', type=str, default='four_enhanced_predictions.csv')
        parser.add_argument('--baseline-file', type=str, default='baseline_ohlcv.csv')
        parser.add_argument('--initial-balance', type=float, default=5000)
        
        # Multi-agent thresholds
        parser.add_argument('--ml-threshold', type=float, default=0.5, 
                          help='Minimum ML model confidence')
        parser.add_argument('--final-threshold', type=float, default=0.55, 
                          help='Minimum final ensemble confidence')
        
        parser.add_argument('--position-size', type=float, default=0.10)
        parser.add_argument('--stop-loss', type=float, default=0.015)
        parser.add_argument('--take-profit', type=float, default=0.025)
        parser.add_argument('--max-hold-hours', type=int, default=48)
        parser.add_argument('--output-dir', type=str, default='.')
        parser.add_argument('--leverage', type=float, default=10.0)
        
        # Advanced options
        parser.add_argument('--max-concurrent-trades', type=int, default=3)
        parser.add_argument('--dynamic-sizing', action='store_true', 
                          help='Use confidence-based position sizing')

    def handle(self, *args, **options):
        pred_path = options['predictions_file']
        base_path = options['baseline_file']

        if not os.path.exists(pred_path) or not os.path.exists(base_path):
            self.stderr.write("❌ Missing required files.")
            return

        self.stdout.write("🤖 Loading Multi-Agent Trading System...")
        
        predictions = pd.read_csv(pred_path, parse_dates=['timestamp'])
        baseline = pd.read_csv(base_path, parse_dates=['timestamp'])

        predictions.sort_values('timestamp', inplace=True)
        baseline.sort_values('timestamp', inplace=True)
        baseline.set_index(['coin', 'timestamp'], inplace=True)

        # Initialize multi-agent system
        decision_engine = MasterDecisionEngine()
        
        trades = []
        open_trades = []
        balance = options['initial_balance']
        trade_id_counter = 0

        stop_loss = options['stop_loss']
        take_profit = options['take_profit']
        max_hold_minutes = options['max_hold_hours'] * 60
        leverage = options['leverage']
        ml_threshold = options['ml_threshold']
        final_threshold = options['final_threshold']
        position_size_pct = options['position_size']
        max_concurrent = options['max_concurrent_trades']

        all_timestamps = sorted(predictions['timestamp'].unique())
        
        # Track BTC context for regime analysis
        btc_context = {}
        
        self.stdout.write(f"🎯 Thresholds: ML>{ml_threshold}, Final>{final_threshold}")
        self.stdout.write(f"📊 Processing {len(all_timestamps)} timestamps...")

        for i, timestamp in enumerate(all_timestamps):
            if i % 1000 == 0:
                self.stdout.write(f"⏳ Processed {i}/{len(all_timestamps)} timestamps...")
                
            # Exit logic for open trades
            still_open = []
            for trade in open_trades:
                key = (trade.coin, timestamp)
                if key not in baseline.index:
                    still_open.append(trade)
                    continue

                row = baseline.loc[key]
                high = row['high']
                low = row['low']
                close = row['close']
                duration = (timestamp - trade.entry_time).total_seconds() / 60

                tp_price = trade.entry_price * (1 + take_profit)
                sl_price = trade.entry_price * (1 - stop_loss)

                if high >= tp_price:
                    trade.close_trade(timestamp, tp_price, 'take_profit')
                    trades.append(trade)
                    decision_engine.add_trade_result(True)  # Won
                elif low <= sl_price:
                    trade.close_trade(timestamp, sl_price, 'stop_loss')
                    trades.append(trade)
                    decision_engine.add_trade_result(False)  # Lost
                elif duration >= max_hold_minutes:
                    trade.close_trade(timestamp, close, 'max_hold')
                    trades.append(trade)
                    # Determine if max_hold was win or loss
                    was_win = close > trade.entry_price
                    decision_engine.add_trade_result(was_win)
                else:
                    still_open.append(trade)

            open_trades = still_open

            # Update BTC context for regime analysis
            btc_rows = predictions[(predictions['timestamp'] == timestamp) & 
                                 (predictions['coin'].str.contains('BTC'))]
            if not btc_rows.empty:
                btc_row = btc_rows.iloc[0]
                btc_context = {
                    'btc_bull_trend': btc_row.get('ema_21', 0) > btc_row.get('ema_50', 0),
                    'btc_strong_trend': btc_row.get('ema_9', 0) > btc_row.get('ema_21', 0),
                }

            # Entry logic with multi-agent system
            current_rows = predictions[predictions['timestamp'] == timestamp]
            for _, row in current_rows.iterrows():
                coin = row['coin']
                ml_confidence = row['prediction_prob']
                ml_prediction = row['prediction']
                entry_price = row['open']

                # Skip if ML model not confident enough
                if ml_prediction != 1 or ml_confidence < ml_threshold:
                    continue
                    
                # Skip if already have position in this coin
                already_open = any(t.coin == coin for t in open_trades)
                if already_open:
                    continue
                    
                # Skip if too many open positions
                if len(open_trades) >= max_concurrent:
                    continue

                # Prepare features for agent analysis
                features = row.to_dict()
                
                # Context data for agents
                context_data = {
                    'btc_context': btc_context,
                    'portfolio_state': {
                        'open_positions': len(open_trades),
                        'balance': balance
                    }
                }

                # Get multi-agent decision
                final_confidence, decision_reason, agent_scores = decision_engine.make_decision(
                    ml_confidence, features, context_data
                )

                # Execute trade if final confidence is high enough
                if final_confidence >= final_threshold:
                    trade_id_counter += 1
                    
                    # Dynamic position sizing based on confidence
                    if options['dynamic_sizing']:
                        # Scale position size by confidence (0.5x to 1.5x base size)
                        size_multiplier = 0.5 + (final_confidence * 1.0)
                        actual_position_size = position_size_pct * size_multiplier
                    else:
                        actual_position_size = position_size_pct
                    
                    trade = Trade(
                        coin=coin,
                        entry_time=timestamp,
                        entry_price=entry_price,
                        direction='long',
                        confidence=final_confidence,
                        trade_id=trade_id_counter,
                        leverage=leverage,
                        agent_scores=agent_scores,
                        decision_reason=decision_reason
                    )
                    
                    open_trades.append(trade)
                    
                    print(f"🚀 Trade #{trade_id_counter}: {coin} @ ${entry_price:.4f} "
                          f"(Conf: {final_confidence:.2f}) - {decision_reason}")

        # Close remaining trades at end of data
        for trade in open_trades:
            try:
                future_rows = baseline.loc[trade.coin]
                future_rows = future_rows[future_rows.index > trade.entry_time]

                if not future_rows.empty:
                    last_time = future_rows.index[-1]
                    last_price = future_rows.iloc[-1]['close']
                    trade.close_trade(last_time, last_price, 'end_of_data')
                    was_win = last_price > trade.entry_price
                    decision_engine.add_trade_result(was_win)
                    trades.append(trade)
            except Exception as e:
                print(f"⚠️ Error closing trade {trade.trade_id}: {e}")

        # Generate enhanced results
        self._generate_results(trades, options, balance, position_size_pct)
        
        # Generate agent performance analysis
        self._analyze_agent_performance(trades, decision_engine)

    def _generate_results(self, trades, options, initial_balance, position_size_pct):
        """Generate detailed trading results"""
        os.makedirs(options['output_dir'], exist_ok=True)
        out_path = os.path.join(options['output_dir'], 'enhanced_trading_results.csv')

        results = pd.DataFrame([{
            'trade_id': t.trade_id,
            'coin': t.coin,
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'pnl_pct': t.pnl_pct,
            'pnl': t.pnl,
            'exit_reason': t.exit_reason or 'unknown',
            'ml_confidence': t.agent_scores.get('ml_model', 0),
            'final_confidence': t.confidence,
            'technical_score': t.agent_scores.get('technical', 0),
            'microstructure_score': t.agent_scores.get('microstructure', 0),
            'regime_score': t.agent_scores.get('regime', 0),
            'risk_score': t.agent_scores.get('risk', 0),
            'decision_reason': t.decision_reason,
            'leverage': t.leverage,
            'duration_minutes': t.duration_minutes
        } for t in trades])

        results.to_csv(out_path, index=False)

        # Calculate performance metrics
        total_trades = len(trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        losses = total_trades - wins
        win_pct = (wins / total_trades) * 100 if total_trades > 0 else 0

        # Calculate final balance
        balance = initial_balance
        for t in trades:
            position_size = balance * position_size_pct
            pl = (t.pnl / 100) * position_size
            balance += pl

        # Performance by exit reason
        exit_reasons = results['exit_reason'].value_counts()
        tp_trades = results[results['exit_reason'] == 'take_profit']
        tp_rate = len(tp_trades) / total_trades * 100 if total_trades > 0 else 0

        self.stdout.write(self.style.SUCCESS(f"\n🎉 Multi-Agent Trading Results:"))
        self.stdout.write(self.style.SUCCESS(f"📊 Total Trades: {total_trades}"))
        self.stdout.write(self.style.SUCCESS(f"✅ Wins: {wins} ({win_pct:.2f}%)"))
        self.stdout.write(self.style.SUCCESS(f"❌ Losses: {losses}"))
        self.stdout.write(self.style.SUCCESS(f"🎯 Take Profit Rate: {tp_rate:.2f}%"))
        self.stdout.write(self.style.SUCCESS(f"💰 Final Balance: ${balance:,.2f}"))
        self.stdout.write(self.style.SUCCESS(f"📈 Total Return: {((balance/initial_balance)-1)*100:.2f}%"))
        self.stdout.write(self.style.SUCCESS(f"💾 Results saved to {out_path}"))

    def _analyze_agent_performance(self, trades, decision_engine):
        """Analyze which agents performed best"""
        if not trades:
            return
            
        df = pd.DataFrame([{
            'won': t.pnl > 0,
            'ml_score': t.agent_scores.get('ml_model', 0),
            'technical_score': t.agent_scores.get('technical', 0),
            'microstructure_score': t.agent_scores.get('microstructure', 0),
            'regime_score': t.agent_scores.get('regime', 0),
            'risk_score': t.agent_scores.get('risk', 0),
            'final_confidence': t.confidence
        } for t in trades])
        
        self.stdout.write(self.style.SUCCESS(f"\n🤖 Agent Performance Analysis:"))
        
        for agent in ['ml_score', 'technical_score', 'microstructure_score', 'regime_score', 'risk_score']:
            # Correlation between agent score and trade success
            if len(df) > 10:
                corr = df[agent].corr(df['won'].astype(float))
                win_rate_high = df[df[agent] > df[agent].median()]['won'].mean()
                win_rate_low = df[df[agent] <= df[agent].median()]['won'].mean()
                
                self.stdout.write(f"  📈 {agent}: Correlation={corr:.3f}, "
                                f"High Score WR={win_rate_high*100:.1f}%, "
                                f"Low Score WR={win_rate_low*100:.1f}%")