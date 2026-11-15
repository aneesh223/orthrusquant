import yfinance as yf
from gnews import GNews
from datetime import datetime, timedelta
import pytz
from functools import lru_cache
import concurrent.futures
from threading import Lock
import time
from src import config
import logging

# Thread-safe cache
_profile_cache = {}
_cache_lock = Lock()

@lru_cache(maxsize=100)
def get_profile_data(ticker):
    """Get profile data with caching and error handling"""
    cache_key = ticker.upper()
    
    # Thread-safe cache check
    with _cache_lock:
        if cache_key in _profile_cache:
            cached_data, cache_time = _profile_cache[cache_key]
            # Cache for 1 hour
            if time.time() - cache_time < 3600:
                return cached_data
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        name = info.get('longName', ticker.upper())
        industry = info.get('industry', 'N/A')
        
        raw_cap = info.get('marketCap', None)
        if raw_cap:
            if raw_cap >= 1_000_000_000_000:
                mkt_cap = f"${raw_cap / 1_000_000_000_000:.2f}T"
            elif raw_cap >= 1_000_000_000:
                mkt_cap = f"${raw_cap / 1_000_000_000:.2f}B"
            else:
                mkt_cap = f"${raw_cap / 1_000_000:.2f}M"
        else:
            mkt_cap = "N/A"
        
        result = (name, industry, mkt_cap)
        
        # Thread-safe cache update
        with _cache_lock:
            _profile_cache[cache_key] = (result, time.time())
        
        return result
        
    except Exception as e:
        logging.error(f"API Error: {e}")
        fallback = (ticker.upper(), "N/A", "N/A")
        
        with _cache_lock:
            _profile_cache[cache_key] = (fallback, time.time())
        
        return fallback

def scrape_alpaca_news(ticker, days=30, end_date=None):
    """
    Fetches historical news using Alpaca's Official News API (Benzinga).
    Uses RESTClient for direct API access as recommended in Alpaca documentation.
    
    Args:
        ticker: Stock symbol
        days: Number of days to look back
        end_date: End date for news search (defaults to now)
    """
    profile_data = get_profile_data(ticker)
    
    try:
        from alpaca.common.rest import RESTClient
        
        # Create RESTClient for news API (uses v1beta1 endpoint)
        news_client = RESTClient(
            base_url='https://data.alpaca.markets',
            api_version='v1beta1',
            api_key=config.API_KEY,
            secret_key=config.API_SECRET
        )
    except Exception as e:
        logging.error(f"Alpaca Client Error: {e}. Check config.py.")
        return profile_data, []
    
    # Calculate date range - support historical backtesting
    if end_date is None:
        end_date = datetime.now(pytz.utc)
    elif not hasattr(end_date, 'tzinfo') or end_date.tzinfo is None:
        end_date = pytz.utc.localize(end_date)
    
    start_date = end_date - timedelta(days=days)
    
    try:
        parsed = []
        page_token = None
        
        # Paginate through all results
        while True:
            news_endpoint = '/news'
            parameters = {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'symbols': ticker,
                'limit': 50
            }
            
            # Add page token if we have one
            if page_token:
                parameters['page_token'] = page_token
            
            # Make the API call
            response = news_client.get(news_endpoint, parameters)
            
            # Extract news articles
            news_articles = response.get('news', [])
            if not news_articles:
                break
                
            # Parse each article
            for article in news_articles:
                created_at = article.get('created_at')
                headline = article.get('headline', 'No headline')
                source = article.get('source', 'benzinga')
                
                # Convert created_at string to datetime if needed
                if isinstance(created_at, str):
                    try:
                        created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    except ValueError:
                        logging.warning("Skipping Alpaca article with an unparseable timestamp")
                        continue
                
                if created_at and headline:
                    parsed.append([created_at, headline, source])
            
            # Check for next page
            page_token = response.get('next_page_token')
            if not page_token:
                break
                
        return profile_data, parsed

    except Exception as e:
        logging.error(f"Alpaca News Error: {e}")
        return profile_data, []

def scrape_gnews(ticker, days=30, end_date=None):
    """Legacy GNews scraping logic with historical date support"""
    profile_data = get_profile_data(ticker)
    parsed = []
    
    # Calculate date range - support historical backtesting
    if end_date is None:
        end_date = datetime.now(pytz.utc)
    elif not hasattr(end_date, 'tzinfo') or end_date.tzinfo is None:
        end_date = pytz.utc.localize(end_date)
    
    cutoff_date = end_date - timedelta(days=days + 1)

    # Helper to parse GNews dates
    def _parse_gnews_date(date_str):
        if not date_str: return datetime.now(pytz.utc)
        formats = ['%a, %d %b %Y %H:%M:%S GMT', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S']
        for fmt in formats:
            try:
                if fmt == '%a, %d %b %Y %H:%M:%S GMT':
                    dt_obj = datetime.strptime(date_str, fmt)
                    return dt_obj.replace(tzinfo=pytz.utc)
                elif 'Z' in date_str:
                    return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                else:
                    dt_obj = datetime.strptime(date_str, fmt)
                    return pytz.utc.localize(dt_obj)
            except ValueError:
                continue
        return datetime.now(pytz.utc)

    try:
        google_news = GNews(language='en', country='US')
        search_queries = [f"{ticker} stock", f"{ticker} finance"]
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(google_news.get_news, q) for q in search_queries]
            for f in concurrent.futures.as_completed(futures):
                try:
                    results.extend(f.result())
                except Exception as exc:
                    logging.error(f"GNews query failed: {exc}")
        
        for item in results:
            title = item.get('title')
            date_str = item.get('published date')
            publisher = item.get('publisher', {}).get('title', 'Unknown')
            dt_obj = _parse_gnews_date(date_str)
            
            if dt_obj < cutoff_date: continue
            parsed.append([dt_obj, title, publisher])
            
        return profile_data, parsed
    except Exception as e:
        logging.error(f"GNews Error: {e}")
        return profile_data, []

def scrape_hybrid(ticker, days=30, end_date=None):
    """
    Combined scraper: Fetches from both Alpaca and Google News.
    Uses deduplication and parallel processing.
    
    Args:
        ticker: Stock symbol
        days: Number of days to look back
        end_date: End date for news search (defaults to now, supports historical backtesting)
    """
    # Fetch from Alpaca (Fast, reliable, structured) - parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        alpaca_future = executor.submit(scrape_alpaca_news, ticker, days, end_date)
        gnews_future = executor.submit(scrape_gnews, ticker, days, end_date)
        
        # Get results
        profile_data, alpaca_news = alpaca_future.result()
        _, gnews_news = gnews_future.result()
    
    # Merge and deduplication using sets
    all_news = alpaca_news + gnews_news
    
    if not all_news:
        return profile_data, []
    
    # Use deduplication with generator expression
    seen_headlines = set()
    unique_news = []
    
    for item in all_news:
        date, headline, source = item
        headline_key = headline.lower().strip()[:100]  # Limit key size for memory efficiency
        
        if headline_key not in seen_headlines:
            unique_news.append(item)
            seen_headlines.add(headline_key)
    
    # Sort by date (Newest first) - use key function for better performance
    unique_news.sort(key=lambda x: x[0], reverse=True)
    
    return profile_data, unique_news
