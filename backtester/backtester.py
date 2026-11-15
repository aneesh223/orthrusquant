"""
Alpaca Backtesting System for Orthrus
Runs trading strategies against historical Alpaca data with realistic simulation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import sys
import os

# Conditional matplotlib import
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, plotting disabled")

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Add parent directory to path to import src modules
parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_dir)

from src import config, market, analyzer, scraper
try:
    from backtester.metrics import affordable_shares, buy_fill, sell_fill
except ImportError:  # script-style launch puts this directory on sys.path
    from metrics import affordable_shares, buy_fill, sell_fill
from src.logic import (
    detect_market_regime,
    get_adaptive_risk_params,
    calculate_price_volatility,
    calculate_bull_market_duration,
    calculate_adaptive_thresholds,
    calculate_adaptive_profit_target,
    calculate_conviction_score,
    detect_momentum_reversal
)
from typing import Dict, List, Tuple, Optional
import bisect
import warnings
warnings.filterwarnings("ignore")


def news_window_days(simulation_days: int, news_lookback_days: Optional[int] = None) -> int:
    """How many days of news to request.

    ``None`` means the whole simulation plus a short buffer, with no 90-day
    cap. Pass an int to cap the request (the old behavior was a hard 90).
    """
    if news_lookback_days is None:
        return max(1, int(simulation_days) + 10)
    return max(1, int(news_lookback_days))


def _as_calendar_date(value):
    if isinstance(value, datetime):
        return value.date()
    if hasattr(value, "date") and not isinstance(value, datetime):
        try:
            return value.date()
        except (TypeError, ValueError):
            return value
    return value


def sentiment_coverage(trading_days, observation_dates) -> Dict:
    """Share of sessions whose score can come from a real sentiment row.

    A day is covered when some observation falls on or before it. Days with
    nothing yet fall back to the neutral score of 5.0.
    """
    days = list(trading_days)
    if not days:
        return {
            "sentiment_coverage_pct": 0.0,
            "sentiment_covered_days": 0,
            "neutral_fallback_days": 0,
            "sentiment_observation_days": 0,
        }
    observations = sorted({_as_calendar_date(d) for d in observation_dates})
    covered = 0
    for day in days:
        key = _as_calendar_date(day)
        if bisect.bisect_right(observations, key) > 0:
            covered += 1
    return {
        "sentiment_coverage_pct": 100.0 * covered / len(days),
        "sentiment_covered_days": covered,
        "neutral_fallback_days": len(days) - covered,
        "sentiment_observation_days": len(observations),
    }


class AlpacaBacktester:
    """
    Backtesting system that simulates trading strategies using historical Alpaca data
    """
    
    def __init__(self, ticker: str, strategy: str = "momentum", initial_cash: float = 10000.0,
                 news_lookback_days: Optional[int] = None,
                 commission_bps: float = 5.0, slippage_bps: float = 5.0):
        """
        Initialize the backtester
        
        Args:
            ticker: Stock symbol to backtest
            strategy: Trading strategy ("momentum" or "value")
            initial_cash: Starting cash amount
        """
        if initial_cash <= 0:
            raise ValueError("Initial cash must be greater than 0")
        
        self.ticker = ticker.upper()
        self.strategy = strategy.lower()
        self.initial_cash = initial_cash
        # None requests news for the whole simulation. An int caps the window.
        self.news_lookback_days = news_lookback_days
        # Per-side costs. 5 bps of commission and 5 bps of slippage by default.
        if commission_bps < 0 or slippage_bps < 0:
            raise ValueError("Commission and slippage must be non-negative")
        self.commission_bps = float(commission_bps)
        self.slippage_bps = float(slippage_bps)
        
        # Portfolio tracking
        self.cash = initial_cash
        self.shares = 0
        self.portfolio_history = []
        self.trade_log = []
        self.score_history = []  # Track Final_Buy_Scores for conviction calculation
        self.last_buy_date = None  # Track last buy date for minimum holding period
        self.first_buy_date = None  # Track first buy date for fair buy-and-hold comparison
        self.first_buy_price = None  # Track first buy price for fair buy-and-hold comparison
        
        # Risk management
        self.entry_price = None  # Track entry price for stop-loss/take-profit
        self.max_portfolio_value = initial_cash  # Track peak for trailing stop
        
        # Bull market tracking for adaptive scaling
        self.bull_market_start_date = None
        self.current_bull_duration = 0
        
        # Overtrading protection
        self.recent_trades = []  # Track recent trades for overtrading protection
        self.last_trade_date = None
        
        # Data storage
        self.full_price_data = None
        self.full_news_data = None
        self.benchmark_prices = None

        # Alpaca client is created on the first live request so constructing a
        # backtester with injected prices does not require real API keys.
        self._alpaca_client = None

    @property
    def alpaca_client(self):
        if self._alpaca_client is None:
            api_key, api_secret = config.require_api_keys()
            self._alpaca_client = StockHistoricalDataClient(
                api_key=api_key,
                secret_key=api_secret
            )
            print("Alpaca client initialized successfully")
        return self._alpaca_client
    
    def fetch_historical_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical daily bars from Alpaca
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            print(f"Fetching historical data for {self.ticker} from {start_date.date()} to {end_date.date()}...")
            
            # Create request for daily bars
            request_params = StockBarsRequest(
                symbol_or_symbols=[self.ticker],
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            # Fetch data from Alpaca
            bars = self.alpaca_client.get_stock_bars(request_params)
            
            if not bars.df.empty:
                # Convert to standard format
                price_data = bars.df.copy()
                
                # Reset index to get timestamp as column, then set as index
                if 'timestamp' not in price_data.columns:
                    price_data = price_data.reset_index()
                
                price_data.set_index('timestamp', inplace=True)
                
                # Rename columns to match yfinance format
                column_mapping = {
                    'open': 'Open',
                    'high': 'High', 
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }
                price_data.rename(columns=column_mapping, inplace=True)
                
                # Ensure timezone is UTC
                if price_data.index.tz is None:
                    price_data.index = price_data.index.tz_localize('UTC')
                elif price_data.index.tz != pytz.UTC:
                    price_data.index = price_data.index.tz_convert('UTC')
                
                print(f"Fetched {len(price_data)} days of price data")
                return price_data
            else:
                print(f"No price data found for {self.ticker}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def fetch_news_data(self, days: int = 90, end_date: datetime = None) -> List[Tuple]:
        """
        Fetch historical news data using the scraper with support for historical dates
        
        Args:
            days: Number of days of news to fetch
            end_date: End date for news search (for historical backtesting)
            
        Returns:
            List of news tuples (timestamp, headline, source)
        """
        try:
            print(f"Fetching {days} days of news data for {self.ticker}...")
            if end_date:
                print(f"   Historical news ending at: {end_date.date()}")
            
            # Alpaca/Benzinga only. GNews does not honor a historical end date.
            profile_data, news_data = scraper.scrape_historical_news(
                self.ticker, days=days, end_date=end_date
            )
            
            print(f"Fetched {len(news_data)} news articles")
            return news_data
            
        except Exception as e:
            print(f"Error fetching news data: {e}")
            return []
    
    def prepare_simulation_data(self, start_date: datetime, end_date: datetime, news_days: int = 90):
        """
        Prepare all data needed for simulation
        
        Args:
            start_date: Simulation start date (should be timezone-aware)
            end_date: Simulation end date (should be timezone-aware)
            news_days: Days of news data to fetch (should be >= simulation period)
        """
        # Ensure dates are timezone-aware
        if start_date.tzinfo is None:
            start_date = pytz.utc.localize(start_date)
        if end_date.tzinfo is None:
            end_date = pytz.utc.localize(end_date)
            
        # Fetch price data with some buffer before start_date for technical indicators.
        # Injected frames are kept so offline runs do not open a client.
        if self.full_price_data is None or self.full_price_data.empty:
            price_start = start_date - timedelta(days=200)  # Buffer for 200-day SMA
            self.full_price_data = self.fetch_historical_data(price_start, end_date)
        
        # Fetch news data with historical end date. An empty list is a real
        # result (no articles) and is not replaced by a live call.
        if self.full_news_data is None:
            self.full_news_data = self.fetch_news_data(days=news_days, end_date=end_date)
        
        if self.full_price_data.empty:
            raise ValueError(f"No price data available for {self.ticker}")
        
        print(f"Simulation data prepared:")
        print(f"   Price data: {len(self.full_price_data)} records")
        print(f"   News data: {len(self.full_news_data)} articles")
    
    def filter_news_by_date(self, target_date: datetime) -> List[Tuple]:
        """
        Filter news data to only include articles up to target_date
        
        Args:
            target_date: Cut-off date for news
            
        Returns:
            Filtered news list
        """
        if not self.full_news_data:
            return []
        
        # Ensure target_date is timezone-aware (UTC)
        if target_date.tzinfo is None:
            target_date = pytz.utc.localize(target_date)
        else:
            target_date = target_date.astimezone(pytz.utc)
        
        filtered_news = []
        for news_item in self.full_news_data:
            news_date, headline, source = news_item
            
            # Convert news_date to datetime if it's not already
            if isinstance(news_date, str):
                try:
                    news_date = datetime.fromisoformat(news_date.replace('Z', '+00:00'))
                except ValueError:
                    continue
            elif not isinstance(news_date, datetime):
                continue
            
            # Ensure timezone awareness for news_date
            if news_date.tzinfo is None:
                news_date = pytz.utc.localize(news_date)
            else:
                news_date = news_date.astimezone(pytz.utc)
            
            # Include news up to target_date
            if news_date <= target_date:
                filtered_news.append((news_date, headline, source))
        
        return filtered_news
    
    def execute_trade(self, action: str, price: float, date: datetime, buy_score: float):
        """
        Execute a buy or sell trade with risk management
        
        Args:
            action: "BUY" or "SELL"
            price: Current stock price
            date: Trade date
            buy_score: The buy score that triggered the trade
        """
        if action == "BUY" and self.cash > 0:
            # Buy as many shares as possible
            if price <= 0:
                print(f"Warning: Invalid price {price} for BUY order, skipping...")
                return
                
            shares_to_buy = affordable_shares(
                self.cash, price, self.commission_bps, self.slippage_bps
            )
            if shares_to_buy > 0:
                fill_price, notional, commission, cost = buy_fill(
                    price, shares_to_buy, self.commission_bps, self.slippage_bps
                )
                self.cash -= cost
                self.shares += shares_to_buy
                self.entry_price = fill_price  # Track entry price for risk management
                
                # Track first buy for fair buy-and-hold comparison
                if self.first_buy_date is None:
                    self.first_buy_date = date
                    self.first_buy_price = price
                
                self.trade_log.append({
                    'Date': date,
                    'Action': 'BUY',
                    'Shares': shares_to_buy,
                    'Price': price,
                    'Fill_Price': fill_price,
                    'Commission': commission,
                    'Notional': notional,
                    'Cost': cost,
                    'Buy_Score': buy_score,
                    'Cash_After': self.cash,
                    'Shares_After': self.shares
                })
                
                print(f"🟢 BUY: {shares_to_buy} shares at ${price:.2f} (Score: {buy_score:.1f}) - Cash: ${self.cash:.2f}")
        
        elif action == "SELL" and self.shares > 0:
            # Sell all shares
            sold_shares = self.shares
            fill_price, notional, commission, proceeds = sell_fill(
                price, sold_shares, self.commission_bps, self.slippage_bps
            )
            self.cash += proceeds
            
            # Calculate P&L for this trade
            if self.entry_price and self.entry_price > 0:
                pnl_pct = ((fill_price - self.entry_price) / self.entry_price) * 100
                pnl_dollar = (fill_price - self.entry_price) * sold_shares - commission
            else:
                pnl_pct = 0
                pnl_dollar = 0
            
            self.trade_log.append({
                'Date': date,
                'Action': 'SELL',
                'Shares': sold_shares,
                'Price': price,
                'Fill_Price': fill_price,
                'Commission': commission,
                'Notional': notional,
                'Proceeds': proceeds,
                'Buy_Score': buy_score,
                'Cash_After': self.cash,
                'Shares_After': 0,
                'PnL_Pct': pnl_pct,
                'PnL_Dollar': pnl_dollar
            })
            
            print(f"🔴 SELL: {self.shares} shares at ${price:.2f} (Score: {buy_score:.1f}) - P&L: {pnl_pct:+.1f}% (${pnl_dollar:+.0f}) - Cash: ${self.cash:.2f}")
            self.shares = 0
            self.entry_price = None  # Reset entry price
    
    def detect_market_regime(self, price_data: pd.DataFrame, current_date: datetime = None, lookback_days: int = 60) -> str:
        """
        Detect current market regime usingshared logic.
        Slices price_data to only include dates up to current_date to prevent future data leakage. 
        """
        # Prevent future data leakage by slicing up to current date if provided
        if current_date is not None:
            sliced_data = price_data[price_data.index <= current_date]
        else:
            sliced_data = price_data
            
        return detect_market_regime(price_data=sliced_data, lookback_days=lookback_days)
    
    def get_adaptive_risk_params(self, market_regime: str, current_date: datetime = None) -> Dict:
        """
        Get adaptive risk management parameters using shared logic with price volatility, strategy, and bull market duration
        """
        # Prevent future data leakage by slicing up to current date if provided
        if current_date is not None:
            sliced_data = self.full_price_data[self.full_price_data.index <= current_date]
        else:
            sliced_data = self.full_price_data
            
        # Calculate price volatility from historical price data only
        price_volatility = calculate_price_volatility(price_data=sliced_data)
        
        # Calculate bull market duration from historical price data only
        bull_market_duration = calculate_bull_market_duration(sliced_data, market_regime)
        
        return get_adaptive_risk_params(market_regime, price_volatility, strategy=self.strategy, bull_market_duration=bull_market_duration)
    
    def calculate_adaptive_thresholds(self, score_history: List[float], market_regime: str, lookback: int = 20) -> Tuple[float, float]:
        """
        Calculate adaptive buy/sell thresholds using shared logic with strategy-specific optimizations and bull market duration
        """
        return calculate_adaptive_thresholds(score_history, market_regime, lookback, strategy=self.strategy, bull_market_duration=self.current_bull_duration)
    
    def check_adaptive_risk_management(self, current_price: float, portfolio_value: float, market_regime: str, current_date: datetime = None) -> Tuple[bool, str]:
        """
        Check if adaptive risk management rules trigger a sell based on market regime with adaptive profit targets
        
        Args:
            current_price: Current stock price
            portfolio_value: Current portfolio value
            market_regime: Current market regime
            current_date: Current simulation date for accurate trailing calculations
            
        Returns:
            Tuple of (should_sell, reason)
        """
        if self.shares == 0 or self.entry_price is None or self.entry_price <= 0:
            return False, ""
        
        # Get adaptive risk parameters without data leak
        risk_params = self.get_adaptive_risk_params(market_regime, current_date)
        
        # Update max portfolio value for trailing stop
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value
        
        # Calculate current P&L
        current_pnl_pct = ((current_price - self.entry_price) / self.entry_price)
        
        # Adaptive stop-loss
        if current_pnl_pct <= risk_params['stop_loss_pct']:
            return True, f"STOP-LOSS ({risk_params['stop_loss_pct']*100:.0f}%): {current_pnl_pct*100:.1f}%"
        
        # ADAPTIVE PROFIT TARGET: Use bull market duration scaling for take-profit
        base_profit_pct = risk_params['take_profit_pct']
        bull_duration = risk_params.get('bull_duration_days', 0)
        
        # Import adaptive profit target calculation
        adaptive_profit_target = calculate_adaptive_profit_target(
            self.entry_price, current_price, bull_duration, base_profit_pct, market_regime
        )
        
        # Adaptive take-profit with scaling
        if current_pnl_pct >= adaptive_profit_target:
            scaling_info = f" (scaled from {base_profit_pct*100:.0f}% due to {bull_duration}d bull market)" if bull_duration > 30 else ""
            return True, f"ADAPTIVE-PROFIT (+{adaptive_profit_target*100:.0f}%): {current_pnl_pct*100:.1f}%{scaling_info}"
        
        # Adaptive trailing stop
        trailing_stop_pct = ((portfolio_value - self.max_portfolio_value) / self.max_portfolio_value)
        if trailing_stop_pct <= risk_params['trailing_stop_pct']:
            return True, f"TRAILING-STOP ({risk_params['trailing_stop_pct']*100:.0f}% from peak): {trailing_stop_pct*100:.1f}%"
        
        return False, ""
    
    def check_overtrading_protection(self, current_date: datetime, market_regime: str) -> bool:
        """
        Check if overtrading protection should prevent a trade
        
        Args:
            current_date: Current simulation date
            market_regime: Current market regime
            
        Returns:
            True if trade should be blocked due to overtrading protection
        """
        # Get risk parameters to check if overtrading protection is enabled
        risk_params = self.get_adaptive_risk_params(market_regime, current_date)
        if not risk_params.get('overtrading_protection', False):
            return False
        
        # Clean old trades (older than 7 days)
        cutoff_date = current_date - timedelta(days=7)
        self.recent_trades = [trade_date for trade_date in self.recent_trades if trade_date > cutoff_date]
        
        # Bull market overtrading protection: max 1 trade per 3 days in bull markets
        if market_regime in ["BULL", "STRONG_BULL"]:
            if self.last_trade_date and (current_date - self.last_trade_date).days < 3:
                return True  # Block trade
            
            # Additional protection: max 2 trades per week in bull markets
            if len(self.recent_trades) >= 2:
                return True  # Block trade
        
        return False  # Allow trade
    
    def execute_adaptive_trade(self, action: str, price: float, date: datetime, buy_score: float, market_regime: str):
        """
        Execute a buy or sell trade with adaptive position sizing based on market regime and overtrading protection
        
        Args:
            action: "BUY" or "SELL"
            price: Current stock price
            date: Trade date
            buy_score: The buy score that triggered the trade
            market_regime: Current market regime
        """
        # Check overtrading protection
        if action == "BUY" and self.check_overtrading_protection(date, market_regime):
            print(f"OVERTRADING PROTECTION: Blocking BUY signal (Score: {buy_score:.1f}) - too many recent trades")
            return
        
        if action == "BUY" and self.cash > 0:
            # Get adaptive position sizing with conviction scoring
            risk_params = self.get_adaptive_risk_params(market_regime, date)
            base_position_multiplier = risk_params['position_size_multiplier']
            
            # Validate price
            if price <= 0:
                print(f"Warning: Invalid price {price} for BUY order, skipping...")
                return
            
            # Calculate conviction score for dynamic position sizing
            conviction_score = calculate_conviction_score(buy_score, self.score_history, market_regime)
            
            # Apply conviction to position sizing
            final_position_multiplier = base_position_multiplier * conviction_score
            
            # Size off the all-in cost so commission and slippage fit in cash.
            base_shares = affordable_shares(
                self.cash, price, self.commission_bps, self.slippage_bps
            )
            adjusted_shares = int(base_shares * final_position_multiplier)
            shares_to_buy = min(adjusted_shares, base_shares)  # Don't exceed available cash
            
            if shares_to_buy > 0:
                fill_price, notional, commission, cost = buy_fill(
                    price, shares_to_buy, self.commission_bps, self.slippage_bps
                )
                self.cash -= cost
                self.shares += shares_to_buy
                self.entry_price = fill_price  # Economic entry, including slippage
                self.last_buy_date = date  # Track buy date for minimum holding period
                
                # Track first buy for fair buy-and-hold comparison
                if self.first_buy_date is None:
                    self.first_buy_date = date
                    self.first_buy_price = fill_price
                
                self.trade_log.append({
                    'Date': date,
                    'Action': 'BUY',
                    'Shares': shares_to_buy,
                    'Price': price,
                    'Fill_Price': fill_price,
                    'Commission': commission,
                    'Notional': notional,
                    'Cost': cost,
                    'Buy_Score': buy_score,
                    'Market_Regime': market_regime,
                    'Position_Multiplier': base_position_multiplier,
                    'Conviction_Score': conviction_score,
                    'Final_Multiplier': final_position_multiplier,
                    'Cash_After': self.cash,
                    'Shares_After': self.shares
                })
                
                # Track trade for overtrading protection
                self.recent_trades.append(date)
                self.last_trade_date = date
                
                print(f"🟢 BUY: {shares_to_buy} shares at ${price:.2f} (Score: {buy_score:.1f}, {market_regime}, Conviction: {conviction_score:.1f}x) - Cash: ${self.cash:.2f}")
        
        elif action == "SELL" and self.shares > 0:
            # Sell all shares
            sold_shares = self.shares
            fill_price, notional, commission, proceeds = sell_fill(
                price, sold_shares, self.commission_bps, self.slippage_bps
            )
            self.cash += proceeds
            
            # Calculate P&L for this trade against the cost-adjusted entry
            if self.entry_price and self.entry_price > 0:
                pnl_pct = ((fill_price - self.entry_price) / self.entry_price) * 100
                pnl_dollar = (fill_price - self.entry_price) * sold_shares - commission
            else:
                pnl_pct = 0
                pnl_dollar = 0
            
            self.trade_log.append({
                'Date': date,
                'Action': 'SELL',
                'Shares': sold_shares,
                'Price': price,
                'Fill_Price': fill_price,
                'Commission': commission,
                'Notional': notional,
                'Proceeds': proceeds,
                'Buy_Score': buy_score,
                'Market_Regime': market_regime,
                'Cash_After': self.cash,
                'Shares_After': 0,
                'PnL_Pct': pnl_pct,
                'PnL_Dollar': pnl_dollar
            })
            
            # Track trade for overtrading protection
            self.recent_trades.append(date)
            self.last_trade_date = date
            
            print(f"🔴 SELL: {self.shares} shares at ${price:.2f} (Score: {buy_score:.1f}, {market_regime}) - P&L: {pnl_pct:+.1f}% (${pnl_dollar:+.0f}) - Cash: ${self.cash:.2f}")
            self.shares = 0
            self.entry_price = None  # Reset entry price
    
    def run(self, start_date: datetime, end_date: datetime, lookback_days: int = 30) -> Dict:
        """
        Run the backtesting simulation with fully optimized pre-calculated scores and momentum reversal detection
        
        Args:
            start_date: Simulation start date
            end_date: Simulation end date
            lookback_days: Days to look back for analysis
            
        Returns:
            Dictionary with simulation results
        """
        print(f"\nStarting backtest simulation for {self.ticker}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Initial cash: ${self.initial_cash:,.2f}")
        print(f"Strategy: {self.strategy.upper()}")
        print("=" * 60)
        
        # Ensure start_date and end_date are timezone-aware (UTC)
        if start_date.tzinfo is None:
            start_date = pytz.utc.localize(start_date)
        else:
            start_date = start_date.astimezone(pytz.utc)
            
        if end_date.tzinfo is None:
            end_date = pytz.utc.localize(end_date)
        else:
            end_date = end_date.astimezone(pytz.utc)
        
        # News covers the whole simulation unless the caller set a cap.
        # The previous hard 90-day cap left every earlier session on a
        # neutral 5.0 because no article had been requested for it.
        simulation_days = (end_date - start_date).days
        news_days = news_window_days(simulation_days, self.news_lookback_days)
        self.prepare_simulation_data(start_date, end_date, news_days)
        self._news_days_requested = news_days
        
        # Get trading days from price data
        trading_days = self.full_price_data[
            (self.full_price_data.index >= start_date) & 
            (self.full_price_data.index <= end_date)
        ].index
        
        import logging
        if len(trading_days) > 0:
            bars_before = len(self.full_price_data[self.full_price_data.index < trading_days[0]])
            if bars_before < 200:
                logging.warning('fewer than 200 price bars exist before trading_days[0]')
        
        if len(trading_days) == 0:
            raise ValueError("No trading days found in the specified period")
        
        print(f"Simulating {len(trading_days)} trading days...")
        
        observation_dates = []
        # OPTIMIZATION: Skip sentiment analysis for very short periods (< 7 days)
        # Use neutral scores to avoid expensive NLP processing. Coverage stays
        # at zero so the report shows that those days are not real sentiment.
        if simulation_days < 7 or len(self.full_news_data) == 0:
            print(f"⚡ Fast mode: Using neutral scores for {simulation_days}-day period (no sentiment analysis)")
            score_lookup = {date.date() if hasattr(date, 'date') else date: 5.0 for date in trading_days}
        else:
            # OPTIMIZATION: Analyze ALL news articles ONCE and calculate ALL scores ONCE
            print(f"🧠 Analyzing {len(self.full_news_data)} news articles (one-time processing)...")
            try:
                # Analyze all news at once to get complete sentiment DataFrame
                full_sentiment_df = analyzer.get_sentiment(self.full_news_data, self.ticker, lookback_days)
                print(f"Sentiment analysis complete: {len(full_sentiment_df)} sentiment records")
            except Exception as e:
                print(f"Error in sentiment analysis: {e}")
                full_sentiment_df = pd.DataFrame()
            
            if full_sentiment_df.empty:
                print("No sentiment data available, using neutral scores")
                # Create a simple lookup with neutral scores for all trading days
                score_lookup = {date.date() if hasattr(date, 'date') else date: 5.0 for date in trading_days}
            else:
                # PRE-CALCULATE ALL Final_Buy_Scores using the complete dataset
                print(f"🔢 Pre-calculating all Final_Buy_Scores for entire period...")
                
                # Calculate verdict using the FULL sentiment and price data
                verdict = market.calculate_verdict(
                    ticker=self.ticker,
                    sentiment_df=full_sentiment_df,
                    strategy=self.strategy,
                    lookback_days=simulation_days + 210,
                    custom_date=None,  # Use end date as reference
                    price_data=self.full_price_data  # Use full price data
                )
                
                # Extract all Final_Buy_Scores and create a lookup dictionary
                final_buy_scores_over_time = verdict.get('Final_Buy_Scores_Over_Time', [])
                score_lookup = {}
                
                for timestamp, score in final_buy_scores_over_time:
                    # Create lookup by date for fast access
                    if hasattr(timestamp, 'date'):
                        date_key = timestamp.date()
                    else:
                        date_key = timestamp
                    score_lookup[date_key] = score
                
                observation_dates = list(full_sentiment_df.index)
                print(f"Pre-calculated {len(score_lookup)} Final_Buy_Scores")
        
        print()
        
        # INITIAL MARKET REGIME DETECTION for adaptive strategy
        initial_market_regime = self.detect_market_regime(self.full_price_data, current_date=start_date)
        current_market_regime = initial_market_regime
        print(f"Initial Market Regime: {initial_market_regime}")
        
        risk_params = self.get_adaptive_risk_params(initial_market_regime, current_date=start_date)
        print(f"Initial Adaptive Parameters: Stop-loss: {risk_params['stop_loss_pct']*100:.0f}%, Take-profit: {risk_params['take_profit_pct']*100:.0f}%, Position size: {risk_params['position_size_multiplier']:.1f}x")
        print()
        
        # OPTIMIZATION: Reduce regime update frequency for longer backtests
        # Update every 5 days for short tests, every 10 days for long tests
        regime_update_interval = 10 if simulation_days > 180 else 5
        
        # Simulation loop - now extremely fast since everything is pre-calculated
        score_history = []  # Track scores for dynamic thresholds and momentum reversal detection
        regime_update_counter = 0  # Track how often we update regime
        
        
        pending_buy_shares = 0
        pending_buy_date = None
        pending_buy_score = 0
        pending_buy_regime = None
        pending_sell = False
        pending_sell_date = None
        pending_sell_score = 0
        pending_sell_regime = None
        pending_sell_reason = ""
        
        for i, simulation_date in enumerate(trading_days):
            try:
                # Safe DataFrame access with error handling
                try:
                    current_close = self.full_price_data.loc[simulation_date, 'Close']
                    current_open = self.full_price_data.loc[simulation_date, 'Open']
                except (KeyError, IndexError):
                    print(f"Warning: No price data for {simulation_date.date()}, skipping...")
                    continue
                
                if current_close <= 0 or current_open <= 0:
                    print(f"Warning: Invalid price for {simulation_date.date()}, skipping...")
                    continue
                
                # OPTIMIZATION V9: EXECUTION LOOKAHEAD BIAS FIX
                # Execute any pending trades from the PREVIOUS day at TODAY'S Open price.
                # This prevents the algorithm from executing trades at the Close of Day T 
                # using signals generated at the Close of Day T (which is mathematically impossible)
                if pending_buy_shares > 0 and self.cash > 0:
                    # Execute pending buy with T+1 open 
                    actual_execution_price = current_open
                    self.execute_adaptive_trade("BUY", actual_execution_price, simulation_date, pending_buy_score, pending_buy_regime)
                    
                    # Reset pending orders
                    pending_buy_shares = 0
                    pending_buy_date = None
                
                elif pending_sell and self.shares > 0:
                    # Execute pending sell with T+1 open
                    actual_execution_price = current_open
                    self.execute_adaptive_trade("SELL", actual_execution_price, simulation_date, pending_sell_score, pending_sell_regime)
                    if pending_sell_reason:
                        print(f"   ↳ Executed pending sell triggered by: {pending_sell_reason}")
                        
                    # Reset pending orders
                    pending_sell = False
                    pending_sell_date = None
                    pending_sell_reason = ""
                
                
                # FAST LOOKUP: Get pre-calculated Final_Buy_Score for this date
                sim_date_key = simulation_date.date() if hasattr(simulation_date, 'date') else simulation_date
                
                # Look for exact match first
                if sim_date_key in score_lookup:
                    buy_score_t = score_lookup[sim_date_key]
                else:
                    # Find the most recent score on or before this date
                    valid_dates = [d for d in score_lookup.keys() if d <= sim_date_key]
                    if valid_dates:
                        most_recent_date = max(valid_dates)
                        buy_score_t = score_lookup[most_recent_date]
                    else:
                        buy_score_t = 5.0  # Neutral fallback
                
                # Add to score history for dynamic thresholds and momentum reversal detection
                score_history.append(buy_score_t)
                self.score_history = score_history[-20:]  # Keep last 20 scores for conviction calculation
                
                # Calculate portfolio value using current close for daily NAV tracking
                portfolio_value = self.cash + (self.shares * current_close)
                
                # OPTIMIZATION: Reduce momentum reversal detection frequency
                # Check for regime changes at configurable intervals (not every 5 days)
                if i > 0 and i % regime_update_interval == 0 and len(score_history) >= 10:
                    # Get recent price data for reversal detection
                    try:
                        recent_price_data = self.full_price_data.loc[:simulation_date].tail(10)
                        if len(recent_price_data) >= 5:  # Need minimum data for reversal detection
                            recent_score_history = score_history[-10:]
                            
                            # Import momentum reversal detection
                            reversal_info = detect_momentum_reversal(recent_price_data, recent_score_history)
                        else:
                            reversal_info = {'reversal_detected': False}
                    except Exception as e:
                        print(f"Warning: Momentum reversal detection failed: {e}")
                        reversal_info = {'reversal_detected': False}
                    
                    if reversal_info['reversal_detected'] and reversal_info['strength'] > 0.7:
                        # Strong reversal detected - update market regime dynamically
                        reversal_type = reversal_info['type']
                        strength = reversal_info['strength']
                        old_regime = current_market_regime
                        
                        if 'BULLISH' in reversal_type:
                            if current_market_regime in ["BEAR", "STRONG_BEAR", "SIDEWAYS"]:
                                if 'ACCELERATION' in reversal_type:
                                    current_market_regime = "STRONG_BULL"
                                else:
                                    current_market_regime = "BULL"
                                regime_update_counter += 1
                                print(f"DYNAMIC REGIME UPDATE #{regime_update_counter}: {old_regime} → {current_market_regime} (reversal: {reversal_type}, strength: {strength:.1f})")
                            elif current_market_regime == "BULL" and 'ACCELERATION' in reversal_type:
                                current_market_regime = "STRONG_BULL"
                                regime_update_counter += 1
                                print(f"DYNAMIC REGIME UPDATE #{regime_update_counter}: {old_regime} → {current_market_regime} (acceleration detected)")
                                
                        elif 'BEARISH' in reversal_type:
                            if current_market_regime in ["BULL", "STRONG_BULL", "SIDEWAYS"]:
                                if 'ACCELERATION' in reversal_type:
                                    current_market_regime = "STRONG_BEAR"
                                else:
                                    current_market_regime = "BEAR"
                                regime_update_counter += 1
                                print(f"DYNAMIC REGIME UPDATE #{regime_update_counter}: {old_regime} → {current_market_regime} (reversal: {reversal_type}, strength: {strength:.1f})")
                            elif current_market_regime == "BEAR" and 'ACCELERATION' in reversal_type:
                                current_market_regime = "STRONG_BEAR"
                                regime_update_counter += 1
                                print(f"DYNAMIC REGIME UPDATE #{regime_update_counter}: {old_regime} → {current_market_regime} (acceleration detected)")
                
                # BULL MARKET DURATION TRACKING: Update bull market duration for adaptive scaling
                if current_market_regime in ["BULL", "STRONG_BULL"]:
                    if self.bull_market_start_date is None:
                        # Start of new bull market
                        self.bull_market_start_date = simulation_date
                        self.current_bull_duration = 0
                    else:
                        # Continue existing bull market
                        self.current_bull_duration = (simulation_date - self.bull_market_start_date).days
                else:
                    # Not in bull market - reset tracking
                    self.bull_market_start_date = None
                    self.current_bull_duration = 0
                
                # OPTIMIZATION: Continually re-evaluate regime without future data
                if i > 0 and i % regime_update_interval == 0:
                    current_market_regime = self.detect_market_regime(self.full_price_data, simulation_date)
                
                # ADAPTIVE RISK MANAGEMENT: Check regime-specific stop-loss/take-profit first
                should_sell_risk, risk_reason = self.check_adaptive_risk_management(current_close, portfolio_value, current_market_regime, simulation_date)
                if should_sell_risk:
                    # Queue the sell for T+1 Open
                    pending_sell = True
                    pending_sell_date = simulation_date
                    pending_sell_score = buy_score_t
                    pending_sell_regime = current_market_regime
                    pending_sell_reason = risk_reason
                    print(f"Pending {risk_reason} for T+1 Open")
                else:
                    # ADAPTIVE THRESHOLDS: Calculate regime-aware buy/sell thresholds
                    buy_threshold, sell_threshold = self.calculate_adaptive_thresholds(score_history, current_market_regime)
                    
                    # Debug: Print adaptive thresholds for first few days
                    if i < 3:
                        print(f"Day {i+1}: Final_Buy_Score = {buy_score_t:.2f} | {current_market_regime} Thresholds: Buy≥{buy_threshold:.3f}, Sell≤{sell_threshold:.3f}")
                    
                    # Trading logic using adaptive thresholds with minimum holding period
                    if buy_score_t >= buy_threshold and self.shares == 0:
                        # Strong buy signal - Queue position for T+1 open
                        pending_buy_shares = 1 # Flag to execute
                        pending_buy_date = simulation_date
                        pending_buy_score = buy_score_t
                        pending_buy_regime = current_market_regime
                        print(f"Queueing BUY signal for T+1 Open (Score: {buy_score_t:.1f})")
                        
                    elif buy_score_t <= sell_threshold and self.shares > 0:
                        # Check minimum holding period before selling
                        risk_params = self.get_adaptive_risk_params(current_market_regime, simulation_date)
                        min_hold_days = risk_params.get('min_hold_days', 0)
                        
                        if self.last_buy_date is None or (simulation_date - self.last_buy_date).days >= min_hold_days:
                            # Weak signal - Queue exit position for T+1 open
                            pending_sell = True
                            pending_sell_date = simulation_date
                            pending_sell_score = buy_score_t
                            pending_sell_regime = current_market_regime
                            pending_sell_reason = "Signal Weakness Threshold Met"
                            print(f"Queueing SELL signal for T+1 Open (Score: {buy_score_t:.1f})")
                        # else: Skip sell due to minimum holding period
                
                # Track portfolio value
                self.portfolio_history.append({
                    'Date': simulation_date,
                    'Price': current_close,
                    'Cash': self.cash,
                    'Shares': self.shares,
                    'Portfolio_Value': portfolio_value,
                    'Final_Buy_Score': buy_score_t,  # Use pre-calculated Final_Buy_Score
                    'Market_Regime': current_market_regime  # Track dynamic regime changes
                })
                
                # OPTIMIZATION: Reduce progress update frequency for longer backtests
                # Update every 10 days for short tests, every 30 days for long tests
                progress_interval = 30 if simulation_days > 180 else 10
                if (i + 1) % progress_interval == 0:
                    buy_thresh, sell_thresh = self.calculate_adaptive_thresholds(score_history, current_market_regime)
                    print(f"Day {i+1}/{len(trading_days)}: ${portfolio_value:,.2f} (Score: {buy_score_t:.1f}, {current_market_regime}) | Thresholds: {buy_thresh:.3f}/{sell_thresh:.3f}")
            
            except Exception as e:
                print(f"Error on {simulation_date.date()}: {e}")
                continue
        
        # Final portfolio value
        final_price = self.portfolio_history[-1]['Price'] if self.portfolio_history else 0
        final_portfolio_value = self.cash + (self.shares * final_price)
        
        # Calculate performance metrics
        if self.initial_cash > 0:
            total_return = ((final_portfolio_value - self.initial_cash) / self.initial_cash) * 100
        else:
            total_return = 0
        
        # Buy-and-Hold Comparison: Always start on Day 1 for fair comparison
        # Since our algorithm CAN buy on Day 1, buy-and-hold should always start on Day 1
        if len(trading_days) > 0:
            try:
                initial_price = self.full_price_data.loc[trading_days[0], 'Close']
                if initial_price > 0 and self.initial_cash > 0:
                    buy_hold_shares = self.initial_cash / initial_price
                    buy_hold_value = buy_hold_shares * final_price
                    buy_hold_return = ((buy_hold_value - self.initial_cash) / self.initial_cash) * 100
                    
                    print(f"Buy-and-Hold: Started on {trading_days[0].date()} at ${initial_price:.2f} (Day 1 - fair comparison)")
                else:
                    buy_hold_return = 0
            except (KeyError, IndexError):
                print("Warning: Could not calculate buy-and-hold return due to missing price data")
                buy_hold_return = 0
        else:
            buy_hold_return = 0
        
        # Calculate dollar-based alpha for more accurate representation
        # This shows the actual dollar difference in gains
        buy_hold_final_value = self.initial_cash * (1 + buy_hold_return / 100)
        strategy_gain = final_portfolio_value - self.initial_cash
        buy_hold_gain = buy_hold_final_value - self.initial_cash
        dollar_alpha = strategy_gain - buy_hold_gain
        
        # Calculate percentage-based alpha (relative to initial capital)
        # This shows how much better/worse the strategy performed as a % of initial investment
        pct_point_alpha = total_return - buy_hold_return
        relative_alpha = (dollar_alpha / self.initial_cash) * 100
        coverage = sentiment_coverage(trading_days, observation_dates)
        
        results = {
            'ticker': self.ticker,
            'strategy': self.strategy,
            'start_date': start_date,
            'end_date': end_date,
            'initial_cash': self.initial_cash,
            'final_portfolio_value': final_portfolio_value,
            'final_cash': self.cash,
            'final_shares': self.shares,
            'total_return_pct': total_return,
            'buy_hold_return_pct': buy_hold_return,
            'alpha': pct_point_alpha,  # Keep for backward compatibility
            'dollar_alpha': dollar_alpha,  # New: actual dollar difference
            'relative_alpha': relative_alpha,  # New: alpha as % of initial capital
            'strategy_gain': strategy_gain,  # New: total $ gained by strategy
            'buy_hold_gain': buy_hold_gain,  # New: total $ gained by buy-hold
            'num_trades': len(self.trade_log),
            'portfolio_history': self.portfolio_history,
            'trade_log': self.trade_log,
            'initial_regime': initial_market_regime,
            'final_regime': current_market_regime,
            'regime_updates': regime_update_counter,
            'news_source': 'alpaca',
            'news_days_requested': getattr(self, '_news_days_requested', None),
            'commission_bps': self.commission_bps,
            'slippage_bps': self.slippage_bps,
            **coverage,
        }
        
        self.print_results(results)
        return results
    
    def print_results(self, results: Dict):
        """Print backtesting results summary with momentum reversal information"""
        print("\n" + "=" * 60)
        print("BACKTESTING RESULTS")
        print("=" * 60)
        print(f"Ticker: {results['ticker']}")
        print(f"Strategy: {results['strategy'].upper()}")
        print(f"Period: {results['start_date'].date()} to {results['end_date'].date()}")
        print(f"Initial Cash: ${results['initial_cash']:,.2f}")
        print(f"Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
        print(f"Final Cash: ${results['final_cash']:,.2f}")
        print(f"Final Shares: {results['final_shares']}")
        print()
        print(f"Strategy Return: {results['total_return_pct']:+.2f}%")
        print(f"Buy & Hold Return: {results['buy_hold_return_pct']:+.2f}%")
        print(f"Alpha (Percentage Points): {results['alpha']:+.2f}%")
        print()
        print(f"Dollar Performance:")
        print(f"   Strategy Gain: ${results['strategy_gain']:+,.2f}")
        print(f"   Buy-Hold Gain: ${results['buy_hold_gain']:+,.2f}")
        print(f"   Dollar Alpha: ${results['dollar_alpha']:+,.2f}")
        print(f"   Relative Alpha: {results['relative_alpha']:+.2f}% of initial capital")
        print()
        print(f"Number of Trades: {results['num_trades']}")
        print(
            f"Transaction Costs: {results.get('commission_bps', 0):.1f} bps commission "
            f"+ {results.get('slippage_bps', 0):.1f} bps slippage, per side"
        )
        coverage_pct = results.get('sentiment_coverage_pct')
        if coverage_pct is not None:
            print(
                f"Sentiment Coverage: {coverage_pct:.1f}% "
                f"({results.get('sentiment_covered_days', 0)} covered, "
                f"{results.get('neutral_fallback_days', 0)} neutral fallback, "
                f"{results.get('sentiment_observation_days', 0)} observation days, "
                f"source={results.get('news_source', 'alpaca')})"
            )
        
        # Market regime information
        initial_regime = results.get('initial_regime', 'N/A')
        final_regime = results.get('final_regime', 'N/A')
        regime_updates = results.get('regime_updates', 0)
        print()
        print(f"Market Regime Analysis:")
        print(f"   Initial Regime: {initial_regime}")
        print(f"   Final Regime: {final_regime}")
        print(f"   Dynamic Updates: {regime_updates}")
        if regime_updates > 0:
            print(f"   Momentum reversal detection successfully adapted to {regime_updates} regime changes")
        
        # Add buy score statistics
        if results['portfolio_history']:
            buy_scores = [h['Final_Buy_Score'] for h in results['portfolio_history']]
            print()
            print(f"Buy Score Stats:")
            print(f"   Max: {max(buy_scores):.2f}")
            print(f"   Min: {min(buy_scores):.2f}")
            print(f"   Avg: {sum(buy_scores)/len(buy_scores):.2f}")
            print(f"   Days ≥5.02 (Buy): {sum(1 for s in buy_scores if s >= 5.02)}")
            print(f"   Days ≤4.98 (Sell): {sum(1 for s in buy_scores if s <= 4.98)}")
        
        print()
        
        if results['trade_log']:
            print("TRADE LOG:")
            for trade in results['trade_log']:
                action_emoji = "🟢" if trade['Action'] == 'BUY' else "🔴"
                regime = trade.get('Market_Regime', 'N/A')
                if trade['Action'] == 'SELL' and 'PnL_Pct' in trade:
                    pnl_info = f" | P&L: {trade['PnL_Pct']:+.1f}%"
                else:
                    pnl_info = ""
                print(f"{action_emoji} {trade['Date'].date()} | {trade['Action']} {trade['Shares']} @ ${trade['Price']:.2f} | Score: {trade['Buy_Score']:.1f} ({regime}){pnl_info}")
        
        print("=" * 60)
    
    def plot_results(self, results: Dict):
        """Plot portfolio performance vs stock price"""
        if not results['portfolio_history']:
            print("No portfolio history to plot")
            return
        
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for plotting")
            return
        
        # Check if we're in a headless environment or batch mode
        import os
        if os.environ.get('DISPLAY') is None and os.name != 'nt':
            print("No display available for plotting (headless environment)")
            return
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(results['portfolio_history'])
        df.set_index('Date', inplace=True)
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'{results["ticker"]} Backtesting Results - {results["strategy"].upper()} Strategy', fontsize=16)
        
        # Plot 1: Portfolio Value vs Stock Price
        ax1_twin = ax1.twinx()
        
        # Portfolio value (left axis)
        ax1.plot(df.index, df['Portfolio_Value'], 'b-', linewidth=2, label='Portfolio Value')
        ax1.set_ylabel('Portfolio Value ($)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        
        # Stock price (right axis)
        ax1_twin.plot(df.index, df['Price'], 'r-', linewidth=1, alpha=0.7, label='Stock Price')
        ax1_twin.set_ylabel('Stock Price ($)', color='r')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        
        # Add trade markers
        if results['trade_log']:
            for trade in results['trade_log']:
                color = 'green' if trade['Action'] == 'BUY' else 'red'
                marker = '^' if trade['Action'] == 'BUY' else 'v'
                ax1.scatter(trade['Date'], df.loc[trade['Date'], 'Portfolio_Value'], 
                           color=color, marker=marker, s=100, zorder=5)
        
        ax1.set_title('Portfolio Value vs Stock Price')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # Plot 2: Buy Scores
        ax2.plot(df.index, df['Final_Buy_Score'], 'g-', linewidth=1)
        ax2.axhline(y=5.02, color='green', linestyle='--', alpha=0.7, label='Buy Threshold (5.02)')
        ax2.axhline(y=4.98, color='red', linestyle='--', alpha=0.7, label='Sell Threshold (4.98)')
        ax2.axhline(y=5.0, color='gray', linestyle='-', alpha=0.5, label='Neutral (5.0)')
        ax2.set_ylabel('Buy Score')
        ax2.set_title('Buy Score Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 10)
        
        # Plot 3: Cash and Shares
        ax3_twin = ax3.twinx()
        
        # Cash (left axis)
        ax3.plot(df.index, df['Cash'], 'b-', linewidth=2, label='Cash')
        ax3.set_ylabel('Cash ($)', color='b')
        ax3.tick_params(axis='y', labelcolor='b')
        
        # Shares (right axis)
        ax3_twin.plot(df.index, df['Shares'], 'orange', linewidth=2, label='Shares')
        ax3_twin.set_ylabel('Shares Held', color='orange')
        ax3_twin.tick_params(axis='y', labelcolor='orange')
        
        ax3.set_title('Cash and Shares Over Time')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot: {e}")
            print("Plot generated but display failed (possibly headless environment)")
        
        # Print summary statistics
        print(f"\nPerformance Summary:")
        print(f"Max Portfolio Value: ${df['Portfolio_Value'].max():,.2f}")
        print(f"Min Portfolio Value: ${df['Portfolio_Value'].min():,.2f}")
        print(f"Volatility (Portfolio): {df['Portfolio_Value'].pct_change().std() * 100:.2f}%")
        print(f"Volatility (Stock): {df['Price'].pct_change().std() * 100:.2f}%")


def run_backtest_example():
    """Example function showing how to use the backtester"""
    
    # Configuration
    ticker = "TSLA"
    strategy = "momentum"  # or "value"
    initial_cash = 10000.0
    
    # Date range (last 3 months for example)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    try:
        # Initialize backtester
        backtester = AlpacaBacktester(ticker, strategy, initial_cash)
        
        # Run simulation
        results = backtester.run(start_date, end_date, lookback_days=30)
        
        # Plot results
        backtester.plot_results(results)
        
        return results
        
    except Exception as e:
        print(f"Backtesting failed: {e}")
        return None


if __name__ == "__main__":
    # Run example backtest
    run_backtest_example()