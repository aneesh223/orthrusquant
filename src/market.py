import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, date, time, timedelta
import pytz
from functools import lru_cache
from threading import Lock
import logging

# Thread-safe cache with lock for concurrent access
_stock_cache = {}
_cache_lock = Lock()

@lru_cache(maxsize=100)
def get_stock_data(ticker, period="1y"):
    """Fetch stock data using yfinance with thread-safe caching"""
    cache_key = f"{ticker}_{period}"
    
    # Thread-safe cache check
    with _cache_lock:
        if cache_key in _stock_cache:
            cached_data, cache_time = _stock_cache[cache_key]
            if datetime.now() - cache_time < timedelta(hours=1):
                return cached_data
    
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, auto_adjust=True, prepost=False)
        
        if data.empty:
            logging.warning(f"No market data found for {ticker}")
            return pd.DataFrame()
        
        # Optimize memory usage with float32 precision
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in data.columns:
                data[col] = data[col].astype('float32')
        
        if 'Volume' in data.columns:
            data['Volume'] = data['Volume'].astype('int32')
        
        # Thread-safe cache update
        with _cache_lock:
            _stock_cache[cache_key] = (data, datetime.now())
        
        return data
    
    except Exception as e:
        logging.error(f"Error fetching market data for {ticker}: {e}")
        return pd.DataFrame()

@lru_cache(maxsize=100)
def get_intraday_data(ticker: str) -> pd.DataFrame:
    """
    Fetch 1-minute intraday data for the last 5 trading days.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL")
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex with 1-minute frequency
        Returns empty DataFrame on error
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="5d", interval="1m")
        
        if data.empty:
            logging.warning(f"No intraday data found for {ticker}")
            return pd.DataFrame()
        
        return data
    
    except Exception as e:
        logging.error(f"Error fetching intraday data for {ticker}: {e}")
        return pd.DataFrame()

@lru_cache(maxsize=500)
def calculate_z_score_cached(prices_tuple, window=50):
    """Calculate Z-Score with caching - prices as tuple for hashing"""
    prices = pd.Series(prices_tuple)
    return calculate_z_score(prices, window)

def calculate_z_score(prices, window=50):
    """Calculate Z-Score using vectorized numpy operations"""
    if len(prices) < window:
        return 0.0
    
    prices_array = prices.values
    
    if len(prices_array) == 0:
        return 0.0
    
    # Vectorized rolling calculations
    rolling_mean = pd.Series(prices_array).rolling(window=window).mean()
    rolling_std = pd.Series(prices_array).rolling(window=window).std()
    
    if len(rolling_mean) == 0 or len(rolling_std) == 0:
        return 0.0
        
    current_price = prices_array[-1]
    current_mean = rolling_mean.iloc[-1]
    current_std = rolling_std.iloc[-1]
    
    if current_std == 0 or np.isnan(current_std):
        return 0.0
    
    z_score = (current_price - current_mean) / current_std
    return float(z_score)

def calculate_golden_death_cross(prices, short_period=50, long_period=200):
    """Calculate Golden Cross and Death Cross signals"""
    if len(prices) < long_period + 1:
        return 0.0, False, False, None, None
    
    # Calculate SMAs
    sma_short = prices.rolling(window=short_period).mean()
    sma_long = prices.rolling(window=long_period).mean()
    
    if len(sma_short) < 2 or len(sma_long) < 2:
        return 0.0, False, False, sma_short, sma_long
    
    # Check for crossovers with bounds checking
    try:
        prev_short = sma_short.iloc[-2]
        prev_long = sma_long.iloc[-2]
        curr_short = sma_short.iloc[-1]
        curr_long = sma_long.iloc[-1]
        
        # Validate values
        if pd.isna(prev_short) or pd.isna(prev_long) or pd.isna(curr_short) or pd.isna(curr_long):
            return 0.0, False, False, sma_short, sma_long
            
    except IndexError:
        return 0.0, False, False, sma_short, sma_long
    
    # Golden Cross: SMA(50) crosses above SMA(200)
    golden_cross = (prev_short <= prev_long) and (curr_short > curr_long)
    
    # Death Cross: SMA(50) crosses below SMA(200)
    death_cross = (prev_short >= prev_long) and (curr_short < curr_long)
    
    # Calculate position score based on current relationship
    if curr_short > curr_long:
        # Bullish position (short SMA above long SMA)
        gap_ratio = (curr_short - curr_long) / curr_long
        cross_score = min(3.0, gap_ratio * 100)  # Scale the gap
    else:
        # Bearish position (short SMA below long SMA)
        gap_ratio = (curr_long - curr_short) / curr_long
        cross_score = max(-3.0, -gap_ratio * 100)  # Negative score
    
    return cross_score, golden_cross, death_cross, sma_short, sma_long

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD using vectorized pandas operations"""
    if len(prices) < slow + signal:
        return 0.0, 0.0, 0.0, False
    
    # Vectorized EMA calculations
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    # MACD Line = Fast EMA - Slow EMA
    macd_line = ema_fast - ema_slow
    
    # Signal Line = EMA of MACD Line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Histogram = MACD - Signal
    histogram = macd_line - signal_line
    
    # Vectorized crossover detection
    bullish_crossover = False
    if len(macd_line) >= 2:
        try:
            # Check if MACD crossed above signal line
            prev_diff = macd_line.iloc[-2] - signal_line.iloc[-2]
            curr_diff = macd_line.iloc[-1] - signal_line.iloc[-1]
            
            if not (pd.isna(prev_diff) or pd.isna(curr_diff)):
                bullish_crossover = prev_diff <= 0 and curr_diff > 0
        except IndexError:
            bullish_crossover = False
    
    try:
        return (float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), 
                float(histogram.iloc[-1]), bullish_crossover)
    except (IndexError, ValueError):
        return (0.0, 0.0, 0.0, False)

def calculate_rsi(prices, window=14):
    """Calculate RSI using vectorized operations"""
    if len(prices) < window + 1:
        return 50.0  # Neutral RSI
    
    # Vectorized price change calculation
    delta = prices.diff()
    
    # Separate gains and losses using numpy where
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Convert back to Series for rolling operations
    gains_series = pd.Series(gains, index=prices.index)
    losses_series = pd.Series(losses, index=prices.index)
    
    # Vectorized rolling averages
    avg_gains = gains_series.rolling(window=window, min_periods=window).mean()
    avg_losses = losses_series.rolling(window=window, min_periods=window).mean()
    
    # Avoid division by zero
    avg_losses = avg_losses.replace(0, np.finfo(float).eps)
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    try:
        final_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    except (IndexError, ValueError):
        final_rsi = 50.0
        
    return final_rsi

@lru_cache(maxsize=200)
def calculate_value_score(z_score):
    """Calculate VALUE strategy score with caching"""
    # VALUE Strategy: Look for oversold conditions (Z < -2.0)
    if z_score <= -2.5:
        technical_score = 9.0
        explanation = f"Stock is {abs(z_score):.1f} standard deviations below mean - deep value opportunity."
    elif z_score <= -2.0:
        technical_score = 8.0
        explanation = f"Stock is {abs(z_score):.1f} standard deviations below mean - strong BUY signal."
    elif z_score <= -1.5:
        technical_score = 7.0
        explanation = f"Stock is {abs(z_score):.1f} standard deviations below mean - good value entry."
    elif z_score <= -1.0:
        technical_score = 6.0
        explanation = f"Stock is {abs(z_score):.1f} standard deviations below mean - moderate value."
    elif z_score <= 0.5:
        technical_score = 5.0
        explanation = f"Stock is near mean (Z-Score: {z_score:.1f}) - neutral valuation."
    elif z_score <= 1.5:
        technical_score = 4.0
        explanation = f"Stock is {z_score:.1f} standard deviations above mean - slightly expensive."
    elif z_score <= 2.0:
        technical_score = 3.0
        explanation = f"Stock is {z_score:.1f} standard deviations above mean - expensive."
    else:
        technical_score = 2.0
        explanation = f"Stock is {z_score:.1f} standard deviations above mean - very expensive."
    
    return technical_score, explanation

@lru_cache(maxsize=200)
def calculate_momentum_score_cached(macd_line, signal_line, histogram, bullish_crossover, rsi, cross_score=0.0, golden_cross=False, death_cross=False):
    """Calculate MOMENTUM strategy score with caching"""
    return calculate_momentum_score(macd_line, signal_line, histogram, bullish_crossover, rsi, cross_score, golden_cross, death_cross)

def calculate_momentum_score(macd_line, signal_line, histogram, bullish_crossover, rsi, cross_score=0.0, golden_cross=False, death_cross=False):
    """Calculate MOMENTUM strategy score based on MACD, RSI, and Golden/Death Cross"""
    # Base score starts at 5 (neutral)
    technical_score = 5.0
    explanations = []
    
    # Golden Cross / Death Cross Analysis (Major signals)
    if golden_cross:
        technical_score += 2.5
        explanations.append("GOLDEN CROSS detected - major bullish signal")
    elif death_cross:
        technical_score -= 2.5
        explanations.append("DEATH CROSS detected - major bearish signal")
    else:
        # Add position score based on SMA relationship
        technical_score += cross_score
        if cross_score > 1.0:
            explanations.append(f"SMA(50) well above SMA(200) - strong bullish trend")
        elif cross_score > 0:
            explanations.append(f"SMA(50) above SMA(200) - bullish trend")
        elif cross_score < -1.0:
            explanations.append(f"SMA(50) well below SMA(200) - strong bearish trend")
        elif cross_score < 0:
            explanations.append(f"SMA(50) below SMA(200) - bearish trend")
    
    # MACD Analysis
    if bullish_crossover:
        if rsi < 70:  # Not overbought
            technical_score += 2.0
            explanations.append("MACD bullish crossover detected")
        else:
            technical_score += 0.5
            explanations.append("MACD bullish crossover but RSI overbought")
    elif macd_line > signal_line:
        if histogram > 0:
            technical_score += 1.5
            explanations.append("MACD above signal with positive momentum")
        else:
            technical_score += 0.5
            explanations.append("MACD above signal but weakening")
    else:
        if histogram < 0:
            technical_score -= 1.0
            explanations.append("MACD below signal with negative momentum")
    
    # RSI Analysis
    if rsi < 30:
        technical_score += 0.5
        explanations.append(f"RSI oversold at {rsi:.0f}")
    elif rsi > 70:
        technical_score -= 0.5
        explanations.append(f"RSI overbought at {rsi:.0f}")
    else:
        explanations.append(f"RSI neutral at {rsi:.0f}")
    
    # Cap the score between 1-10
    technical_score = max(1.0, min(10.0, technical_score))
    
    explanation = ". ".join(explanations) + "."
    
    return technical_score, explanation

def calculate_verdict(ticker, sentiment_df, strategy="value", lookback_days=30, custom_date=None, price_data=None):
    """Calculate trading verdict with B(t) calculation and rolling window Final_Buy_Scores
    
    Args:
        ticker: Stock ticker symbol
        sentiment_df: DataFrame with sentiment analysis results
        strategy: "value" or "momentum" strategy
        lookback_days: Number of days to look back
        custom_date: Custom date for analysis (for backtesting)
        price_data: Optional DataFrame with price data (for backtesting injection)
    """
    if sentiment_df.empty:
        return {
            'Sentiment_Health': 5.0,
            'Technical_Score': 5.0,
            'Final_Buy_Score': 5.0,
            'Explanation': "No sentiment data available for analysis.",
            'Strategy': "N/A"
        }
    
    # Period selection using dictionary lookup
    period_map = {
        1: "1d", 2: "2d", 5: "5d", 7: "7d", 
        30: "1mo", 90: "3mo", 180: "6mo"
    }
    
    period = period_map.get(lookback_days, "1y" if lookback_days > 180 else "1mo")
    
    # Use injected price data if provided (for backtesting), otherwise fetch from yfinance
    if price_data is not None:
        logging.info(f"Using injected price data for backtesting analysis...")
        stock_data = price_data.copy()
    else:
        logging.info(f"Fetching {period} of market data for analysis...")
        # Get stock data with caching
        stock_data = get_stock_data(ticker, period=period)
        
    if stock_data.empty:
        return {
            'Sentiment_Health': 5.0,
            'Technical_Score': 5.0,
            'Final_Buy_Score': 5.0,
            'Explanation': "No stock price data available for technical analysis.",
            'Strategy': "N/A"
        }
    
    # Date filtering (still apply filtering logic even with injected data)
    stock_data_filtered = _filter_stock_data(stock_data, lookback_days, custom_date)
    
    if stock_data_filtered.empty:
        stock_data_filtered = stock_data
    
    # Get closing prices for technical analysis
    prices = stock_data_filtered['Close']
    strategy_name = "MOMENTUM" if strategy.lower() == "momentum" else "VALUE"
    
    # Calculate all B(t) values first (one pass through sentiment data)
    buy_scores_at_t = []  # List of (timestamp, B(t)) tuples
    
    logging.info(f"Calculating B(t) values for {len(sentiment_df)} time periods...")
    
    for timestamp, row in sentiment_df.iterrows():
        sentiment_score = row['Compound_Score']
        sentiment_health = (sentiment_score + 1) * 5  # Convert to 0-10 scale
        
        # Handle timezone-aware comparisons for price data
        if hasattr(timestamp, 'tz_localize') and timestamp.tz is None:
            # Make timestamp timezone-aware to match price data
            if not prices.index.empty and prices.index.tz is not None:
                timestamp_tz = timestamp.tz_localize(prices.index.tz)
            else:
                timestamp_tz = timestamp
        elif hasattr(timestamp, 'tz_convert') and timestamp.tz is not None:
            # Convert to price data timezone if needed
            if not prices.index.empty and prices.index.tz is not None:
                timestamp_tz = timestamp.tz_convert(prices.index.tz)
            else:
                timestamp_tz = timestamp.tz_localize(None)  # Remove timezone
        else:
            timestamp_tz = timestamp
        
        # Get technical score at time t (using price data up to that point)
        if strategy.lower() == "momentum":
            # For momentum, use recent price data up to time t
            try:
                price_data_up_to_t = prices[prices.index <= timestamp_tz]
            except (TypeError, ValueError):
                # Fallback: use all price data if comparison fails
                price_data_up_to_t = prices
                
            if len(price_data_up_to_t) >= 200:  # Need enough data for Golden/Death Cross
                # Calculate Golden/Death Cross
                cross_score, golden_cross, death_cross, sma_short, sma_long = calculate_golden_death_cross(price_data_up_to_t)
                
                # Calculate MACD and RSI
                macd_line, signal_line, histogram, bullish_crossover = calculate_macd(price_data_up_to_t)
                rsi = calculate_rsi(price_data_up_to_t)
                
                technical_score_t, _ = calculate_momentum_score_cached(
                    macd_line, signal_line, histogram, bullish_crossover, rsi, cross_score, golden_cross, death_cross
                )
            elif len(price_data_up_to_t) >= 26:  # Need enough data for MACD only
                macd_line, signal_line, histogram, bullish_crossover = calculate_macd(price_data_up_to_t)
                rsi = calculate_rsi(price_data_up_to_t)
                technical_score_t, _ = calculate_momentum_score_cached(
                    macd_line, signal_line, histogram, bullish_crossover, rsi, 0.0, False, False
                )
            else:
                technical_score_t = 5.0  # Neutral if not enough data
        else:
            # For value, use Z-score up to time t
            try:
                price_data_up_to_t = prices[prices.index <= timestamp_tz]
            except (TypeError, ValueError):
                # Fallback: use all price data if comparison fails
                price_data_up_to_t = prices
                
            if len(price_data_up_to_t) >= 50:  # Need enough data for Z-score
                z_score = calculate_z_score(price_data_up_to_t, window=50)
                technical_score_t, _ = calculate_value_score(z_score)
            else:
                technical_score_t = 5.0  # Neutral if not enough data
        
        # Calculate B(t) for this timestamp
        sentiment_penalty = 0.5 if sentiment_score < -0.5 else 1.0
        buy_score_t = ((technical_score_t * 0.6) + (sentiment_health * 0.4)) * sentiment_penalty
        buy_score_t = np.clip(buy_score_t, 1.0, 10.0)
        
        buy_scores_at_t.append((timestamp, buy_score_t))
    
    if not buy_scores_at_t:
        return {
            'Sentiment_Health': 5.0,
            'Technical_Score': 5.0,
            'Final_Buy_Score': 5.0,
            'Explanation': "No valid buy scores could be calculated.",
            'Strategy': strategy_name
        }
    
    # Calculate Final_Buy_Score at each time t using rolling window
    final_buy_scores_over_time = []
    timestamps = []
    
    # Define rolling window size (in days) - shorter window for more responsiveness
    rolling_window_days = min(lookback_days if lookback_days > 1 else 3, 5)  # Max 5 days, min 3 days
    
    logging.info(f"Calculating Final_Buy_Scores using {rolling_window_days}-day rolling window...")
    
    # Use sliding window with deque for O(N) complexity instead of O(N²)
    from collections import deque
    import time
    
    start_time = time.time()
    
    # Sliding window to store (timestamp, buy_score) tuples within the rolling window
    window = deque()
    
    for i, (timestamp, buy_score_t) in enumerate(buy_scores_at_t):
        timestamps.append(timestamp)
        
        # Define the rolling window: from (t - rolling_window_days) to t
        window_start = timestamp - pd.Timedelta(days=rolling_window_days)
        
        # Remove expired entries from LEFT side of deque - O(k) where k << N
        while window and window[0][0] < window_start:
            window.popleft()  # O(1) operation
        
        # Add current score to RIGHT side - O(1)
        window.append((timestamp, buy_score_t))
        
        # Calculate recency-weighted average of current window - O(k) where k = window size
        if len(window) == 1:
            final_buy_score_t = window[0][1]  # Single score
        else:
            # Extract scores and timestamps from current window
            window_timestamps = [ts for ts, _ in window]
            window_buy_scores = [score for _, score in window]
            
            # Calculate days ago for each timestamp in the window (relative to current timestamp)
            days_ago = []
            for window_ts in window_timestamps:
                if hasattr(window_ts, 'date') and hasattr(timestamp, 'date'):
                    days_diff = (timestamp.date() - window_ts.date()).days
                else:
                    days_diff = len(window_timestamps) - window_timestamps.index(window_ts) - 1
                days_ago.append(max(0, days_diff))
            
            # Calculate recency weights (more recent = higher weight)
            decay_rate = 0.5  # Much higher decay rate - recent data gets much more weight
            recency_weights = np.exp(-decay_rate * np.array(days_ago))
            recency_weights = recency_weights / np.sum(recency_weights)  # Normalize
            
            # Final Buy Score at time t = Recency-weighted average of window B(t) scores
            final_buy_score_t = np.sum(np.array(window_buy_scores) * recency_weights)
        
        final_buy_scores_over_time.append(final_buy_score_t)
    
    optimization_time = time.time() - start_time
    logging.info(f"Calculation completed in {optimization_time:.3f} seconds")
    
    # For the main program display, use the most recent Final_Buy_Score
    final_buy_score = final_buy_scores_over_time[-1]
    
    # Calculate summary statistics for explanation
    avg_sentiment = sentiment_df['Compound_Score'].mean()
    avg_technical = np.mean([score for score in final_buy_scores_over_time])  # Average of Final_Buy_Scores
    
    # Sentiment description using numpy
    sentiment_thresholds = np.array([-0.3, -0.1, 0.1, 0.3])
    sentiment_labels = ["VERY NEGATIVE", "MILDLY NEGATIVE", "NEUTRAL", "MILDLY POSITIVE", "VERY POSITIVE"]
    sentiment_idx = np.searchsorted(sentiment_thresholds, avg_sentiment)
    sentiment_description = sentiment_labels[sentiment_idx]
    
    explanation = f"{strategy_name} Strategy: Analysis complete based on market sentiment and technical indicators. Overall sentiment is {sentiment_description}."
    
    return {
        'Sentiment_Health': round((avg_sentiment + 1) * 5, 1),
        'Technical_Score': round(avg_technical, 1),
        'Final_Buy_Score': round(final_buy_score, 1),
        'Explanation': explanation,
        'Strategy': strategy_name,
        'Final_Buy_Scores_Over_Time': list(zip(timestamps, final_buy_scores_over_time))  # Final_Buy_Score at each time t
    }

def _filter_stock_data(stock_data, lookback_days, custom_date):
    """Optimized stock data filtering helper function with robust timezone handling for Alpaca data"""
    if lookback_days <= 1:
        # For 1-day analysis, filter to only that specific day
        target_date = custom_date.date() if custom_date else datetime.now().date()
        
        if not stock_data.empty:
            if stock_data.index.tz is not None:
                # Timezone-aware filtering (handles Alpaca UTC timestamps)
                target_start = datetime.combine(target_date, datetime.min.time())
                target_end = datetime.combine(target_date, datetime.max.time())
                
                # Convert to UTC first, then to data timezone
                try:
                    if hasattr(stock_data.index.tz, 'zone') and stock_data.index.tz.zone == 'UTC':
                        # Data is already in UTC
                        target_start = pytz.utc.localize(target_start)
                        target_end = pytz.utc.localize(target_end)
                    elif str(stock_data.index.tz) == 'UTC':
                        # Data is already in UTC (fallback check)
                        target_start = pytz.utc.localize(target_start)
                        target_end = pytz.utc.localize(target_end)
                    else:
                        # Convert to data timezone
                        target_start = pytz.utc.localize(target_start).astimezone(stock_data.index.tz)
                        target_end = pytz.utc.localize(target_end).astimezone(stock_data.index.tz)
                except AttributeError:
                    # Fallback for timezone objects without 'zone' attribute
                    target_start = pytz.utc.localize(target_start)
                    target_end = pytz.utc.localize(target_end)
                
                return stock_data[(stock_data.index >= target_start) & (stock_data.index <= target_end)]
            else:
                # Non-timezone aware filtering
                return stock_data[stock_data.index.date == target_date]
    else:
        # Multi-day analysis
        cutoff_date = (custom_date or datetime.now()) - timedelta(days=lookback_days)
        
        # Handle timezone awareness with robust Alpaca UTC support
        if not stock_data.empty and stock_data.index.tz is not None:
            if cutoff_date.tzinfo is None:
                # Convert naive datetime to UTC first, then to data timezone
                try:
                    if hasattr(stock_data.index.tz, 'zone') and stock_data.index.tz.zone == 'UTC':
                        cutoff_date = pytz.utc.localize(cutoff_date)
                    elif str(stock_data.index.tz) == 'UTC':
                        cutoff_date = pytz.utc.localize(cutoff_date)
                    else:
                        cutoff_date = pytz.utc.localize(cutoff_date).astimezone(stock_data.index.tz)
                except AttributeError:
                    # Fallback for timezone objects without 'zone' attribute
                    cutoff_date = pytz.utc.localize(cutoff_date)
            else:
                # Convert timezone-aware datetime to data timezone
                try:
                    if hasattr(stock_data.index.tz, 'zone') and stock_data.index.tz.zone == 'UTC':
                        cutoff_date = cutoff_date.astimezone(pytz.utc)
                    elif str(stock_data.index.tz) == 'UTC':
                        cutoff_date = cutoff_date.astimezone(pytz.utc)
                    else:
                        cutoff_date = cutoff_date.astimezone(stock_data.index.tz)
                except AttributeError:
                    # Fallback for timezone objects without 'zone' attribute
                    cutoff_date = cutoff_date.astimezone(pytz.utc)
        
        return stock_data[stock_data.index >= cutoff_date]
    
    return stock_data

# Legacy function for backward compatibility
def get_financials(ticker, sentiment_df, timeframe_days=30):
    """Legacy function - now just returns sentiment_df for compatibility"""
    return sentiment_df

def get_visualization_data(ticker, sentiment_df, timeframe_days=30):
    """Get combined sentiment and price data for visualization with buy/sell signals"""
    if sentiment_df.empty:
        return sentiment_df
    
    try:
        # Get the date range from sentiment data
        start_date = sentiment_df.index.min()
        end_date = sentiment_df.index.max()
        
        # Handle different index types
        if hasattr(start_date, 'date'):
            start_date = start_date.date()
        if hasattr(end_date, 'date'):
            end_date = end_date.date()
            
        # For price data, always fetch up to the current date to get the latest market data
        # This ensures we show the most recent price even if sentiment data is older
        price_end_date = datetime.now().date()
        
        # Get stock data
        stock = yf.Ticker(ticker)
        
        # Determine interval based on timeframe
        if timeframe_days <= 1:
            interval = "1h"
            # Use specific date range instead of period to avoid extra buffer
            price_data = stock.history(start=start_date, end=price_end_date + timedelta(days=1), interval=interval)
            # print(f"Fetched price data using date range {start_date} to {price_end_date}, interval='{interval}': {len(price_data) if not price_data.empty else 0} records")  # Hidden from user
        else:
            interval = "1d"
            price_data = stock.history(start=start_date, end=price_end_date + timedelta(days=1), interval=interval)
            # print(f"Fetched price data using date range {start_date} to {price_end_date}, interval='{interval}': {len(price_data) if not price_data.empty else 0} records")  # Hidden from user
        
        if price_data.empty:
            print("No market data found for the given date range")
            return sentiment_df
        
        # Filter out any future dates from price data
        now = datetime.now()
        if price_data.index.tz is not None:
            now = pytz.utc.localize(now).astimezone(price_data.index.tz)
        price_data = price_data[price_data.index <= now]
        
        # Filter price data for intraday if needed
        if timeframe_days <= 1:
            # For 1-day analysis, filter to only the specific day from sentiment data
            # Get the target date from sentiment data
            target_date = start_date
            if hasattr(target_date, 'date'):
                target_date = target_date.date()
            
            # Filter price data to only the target date
            if price_data.index.tz is not None:
                # Convert target date to timezone-aware datetime for comparison
                target_start = datetime.combine(target_date, datetime.min.time())
                target_end = datetime.combine(target_date, datetime.max.time())
                target_start = pd.Timestamp(target_start).tz_localize('UTC').tz_convert(price_data.index.tz)
                target_end = pd.Timestamp(target_end).tz_localize('UTC').tz_convert(price_data.index.tz)
                price_data = price_data[(price_data.index >= target_start) & (price_data.index <= target_end)]
            else:
                # For non-timezone aware data, filter by date
                price_data = price_data[price_data.index.date == target_date]
        
        # Calculate buy/sell signals based on strategy
        buy_signals = []
        sell_signals = []
        
        if len(price_data) >= 200:  # Enough data for Golden/Death Cross
            # Calculate Golden/Death Cross for the entire period
            cross_score, _, _, sma_short, sma_long = calculate_golden_death_cross(price_data['Close'])
            
            # Find crossover points
            for i in range(1, len(sma_short)):
                if pd.notna(sma_short.iloc[i]) and pd.notna(sma_long.iloc[i]) and pd.notna(sma_short.iloc[i-1]) and pd.notna(sma_long.iloc[i-1]):
                    prev_short = sma_short.iloc[i-1]
                    prev_long = sma_long.iloc[i-1]
                    curr_short = sma_short.iloc[i]
                    curr_long = sma_long.iloc[i]
                    
                    # Golden Cross
                    if prev_short <= prev_long and curr_short > curr_long:
                        buy_signals.append((sma_short.index[i], price_data['Close'].iloc[i]))
                    
                    # Death Cross
                    elif prev_short >= prev_long and curr_short < curr_long:
                        sell_signals.append((sma_short.index[i], price_data['Close'].iloc[i]))
        
        # Also check for high buy scores (>7) and low buy scores (<4) from sentiment analysis
        for timestamp, row in sentiment_df.iterrows():
            if 'Buy_Score' in row and pd.notna(row['Buy_Score']):
                if row['Buy_Score'] > 7.0:
                    # Find corresponding price
                    try:
                        if timestamp in price_data.index:
                            price = price_data.loc[timestamp, 'Close']
                            buy_signals.append((timestamp, price))
                    except (KeyError, TypeError, ValueError):
                        pass
                elif row['Buy_Score'] < 4.0:
                    try:
                        if timestamp in price_data.index:
                            price = price_data.loc[timestamp, 'Close']
                            sell_signals.append((timestamp, price))
                    except (KeyError, TypeError, ValueError):
                        pass
        
        # Resample price data to match sentiment frequency
        if timeframe_days <= 1 and interval == "1h":
            price_resampled = price_data.resample('h').last()
        else:
            price_resampled = price_data.resample('D').last()
            price_resampled.index = price_resampled.index.date
        
        start_time = price_data.index.min()
        end_time = price_data.index.max()
        
        sentiment_start = sentiment_df.index.min()
        sentiment_end = sentiment_df.index.max()
        
        # Merge sentiment and price data first
        merged_df = sentiment_df.join(price_resampled[['Close', 'Volume']], how='left')
        
        # Forward fill missing prices
        merged_df['Close'] = merged_df['Close'].ffill()
        merged_df['Volume'] = merged_df['Volume'].fillna(0)
        
        # Store data info for display in visualizer (after merged_df is created)
        data_info = {
            'price_records': len(price_data),
            'price_start': start_time.strftime('%Y-%m-%d %H:%M'),
            'price_end': end_time.strftime('%Y-%m-%d %H:%M'),
            'resampled_records': len(price_resampled),
            'sentiment_records': len(sentiment_df),
            'sentiment_start': sentiment_start.strftime('%Y-%m-%d %H:%M') if hasattr(sentiment_start, 'strftime') else str(sentiment_start),
            'sentiment_end': sentiment_end.strftime('%Y-%m-%d %H:%M') if hasattr(sentiment_end, 'strftime') else str(sentiment_end),
            'merged_records': len(merged_df),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals
        }
        
        # Store data info in the dataframe as metadata
        merged_df.attrs['data_info'] = data_info
        
        return merged_df
        
    except Exception as e:
        print(f"Error fetching market data for visualization: {e}")
        return sentiment_df

def detect_market_regime(ticker: str, lookback_days: int = 60) -> str:
    """
    Detect current market regime using shared logic
    Automatically stays in sync with any improvements
    """
    from logic import detect_market_regime as shared_detect_market_regime
    return shared_detect_market_regime(ticker=ticker, lookback_days=lookback_days)

def get_adaptive_risk_params(market_regime: str, strategy: str = "momentum") -> dict:
    """
    Get risk parameters using shared logic with strategy-specific optimizations
    Automatically stays in sync with any improvements
    """
    from logic import get_adaptive_risk_params as shared_get_adaptive_risk_params
    return shared_get_adaptive_risk_params(market_regime, strategy=strategy)

def calculate_adaptive_thresholds(score_history: list, market_regime: str, lookback: int = 20, strategy: str = "momentum") -> tuple:
    """
    Calculate adaptive thresholds using shared logic with strategy-specific optimizations
    Automatically stays in sync with any improvements
    """
    from logic import calculate_adaptive_thresholds as shared_calculate_adaptive_thresholds
    return shared_calculate_adaptive_thresholds(score_history, market_regime, lookback, strategy)

def get_trading_recommendation(ticker: str, final_buy_score: float, sentiment_df: pd.DataFrame, strategy: str = "momentum") -> dict:
    """
    Convert Final_Buy_Score into concrete BUY/HOLD/SELL recommendation
    Uses shared logic for consistency with strategy-specific optimizations and bull market duration scaling
    """
    from logic import get_trading_recommendation as shared_get_trading_recommendation
    
    # Get price data for bull market duration calculation
    import yfinance as yf
    try:
        stock = yf.Ticker(ticker)
        price_data = stock.history(period="6mo")  # 6 months for regime detection
    except Exception as exc:
        logging.warning(f"Could not load price history for {ticker}: {exc}")
        price_data = None
    
    return shared_get_trading_recommendation(ticker, final_buy_score, sentiment_df, price_data, strategy=strategy)