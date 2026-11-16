from datetime import datetime, timedelta
import pytz
import re
import sys
import os
import logging

# C0/C1 controls, including ESC and BEL. Tabs and newlines are left alone.
_TERMINAL_CONTROLS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def strip_terminal_controls(value):
    """Drop bytes a provider could use to rewrite the terminal."""
    if value is None:
        return ""
    return _TERMINAL_CONTROLS.sub("", str(value))

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import scraper
import analyzer
import market
import visualizer

def get_last_trading_day(date=None):
    """
    Get the last trading day (Monday-Friday, excluding weekends)
    
    Args:
        date: Optional date to check (defaults to today)
        
    Returns:
        datetime: Last trading day
    """
    if date is None:
        date = datetime.now()
    
    # Get the weekday (0=Monday, 6=Sunday)
    weekday = date.weekday()
    
    if weekday < 5:  # Monday-Friday (0-4)
        return date
    elif weekday == 5:  # Saturday
        return date - timedelta(days=1)  # Go back to Friday
    else:  # Sunday (weekday == 6)
        return date - timedelta(days=2)  # Go back to Friday

def analyze_stock(ticker, lookback_days, strategy_mode):
    """Stock analysis with memory efficiency"""
    logging.info(f"\nAnalyzing {ticker}...")
    
    try:
        profile_data, raw_data = scraper.scrape_hybrid(ticker, days=lookback_days)
        company_name, industry, mkt_cap = profile_data
        
    except Exception as e:
        logging.error(f"Error: {e}")
        return
    
    if not raw_data:
        logging.warning("No data found. Check your connection.")
        return
    
    # Sentiment analysis
    sentiment_df = analyzer.get_sentiment(raw_data, ticker, lookback_days)
    best_news, worst_news = analyzer.get_top_headlines_wrapper(raw_data, ticker, lookback_days)
    
    # Date filtering with trading day awareness
    today = get_last_trading_day()  # Use last trading day instead of raw datetime.now()
    if today.date() != datetime.now().date():
        logging.info(f"Note: Using last trading day ({today.strftime('%A, %B %d')}) - Markets closed on weekends")
        
    if lookback_days <= 1:
        cutoff_date = today.date()
        sentiment_df = sentiment_df[sentiment_df.index.date == cutoff_date]
    else:
        cutoff_date = (today - timedelta(days=lookback_days)).date()
        if hasattr(sentiment_df.index[0], 'date'):
            sentiment_df.index = sentiment_df.index.date
        sentiment_df = sentiment_df[sentiment_df.index >= cutoff_date]

    if sentiment_df.empty:
        logging.warning(f"No relevant news found for {ticker}.")
        return
    
    # Calculate verdict and display results
    verdict = market.calculate_verdict(ticker, sentiment_df, strategy_mode.lower(), lookback_days)
    
    if verdict:
        # Get trading recommendation using the same signal code as the backtester
        trading_rec = market.get_trading_recommendation(ticker, verdict['Final_Buy_Score'], sentiment_df, strategy=strategy_mode)
        print_analysis_report(
            company_name=company_name,
            ticker=ticker,
            industry=industry,
            mkt_cap=mkt_cap,
            verdict=verdict,
            trading_rec=trading_rec,
            best_news=best_news,
            worst_news=worst_news,
        )

    # Generate visualization
    final_df = market.get_visualization_data(ticker, sentiment_df, lookback_days)
    visualizer.plot_graph(ticker, final_df, profile_data, lookback_days, strategy_mode)


def print_analysis_report(*, company_name, ticker, industry, mkt_cap, verdict, trading_rec, best_news, worst_news):
    """Print the analysis. Provider strings are stripped of terminal controls."""
    company_name = strip_terminal_controls(company_name)
    ticker = strip_terminal_controls(ticker)
    industry = strip_terminal_controls(industry)
    mkt_cap = strip_terminal_controls(mkt_cap)
    best_news = strip_terminal_controls(best_news)
    worst_news = strip_terminal_controls(worst_news)
    explanation = strip_terminal_controls(verdict.get("Explanation"))
    reasoning = strip_terminal_controls(trading_rec.get("reasoning"))
    regime = strip_terminal_controls(trading_rec.get("market_regime"))
    recommendation = strip_terminal_controls(trading_rec.get("recommendation"))

    print("\n" + "="*50)
    print(f"   ANALYSIS FOR {company_name} ({ticker})")
    print("="*50)
    print(f"Industry:   {industry}")
    print(f"Market Cap: {mkt_cap}")
    print("-" * 50)
    print(f"Strategy:           {verdict['Strategy']}")
    print(f"Market Sentiment:   {verdict['Sentiment_Health']:.1f}/10")
    print(f"Market Analysis:    {verdict['Technical_Score']:.1f}/10")
    print("-" * 50)
    print(f"FINAL BUY SCORE:    {verdict['Final_Buy_Score']:.1f}/10")

    rec_color = "🟢" if recommendation == "BUY" else "🔴" if recommendation == "SELL" else "🟡"
    print(f"TRADING SIGNAL:     {rec_color} {recommendation} ({trading_rec['confidence']:.0f}% confidence)")
    print(f"Market Regime:      {regime}")

    risk_params = trading_rec['risk_params']
    print(f"Adaptive Thresholds: Buy≥{trading_rec['buy_threshold']:.3f}, Sell≤{trading_rec['sell_threshold']:.3f}")
    print(f"Risk Management:    Stop-loss {risk_params['stop_loss_pct']*100:+.0f}%, Take-profit {risk_params['take_profit_pct']*100:+.0f}%")

    print(f"\nREASONING:\n{explanation}")
    print(f"Signal Logic: {reasoning}")
    print("-" * 50)
    print(f"TOP POSITIVE NEWS:\n>> {best_news}")
    print(f"\nTOP NEGATIVE NEWS:\n>> {worst_news}")
    print("="*50 + "\n")

def main():
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("   ORTHRUS: AI SENTIMENT ANALYZER")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    while True:
        ticker = input("\nEnter ticker (or 'exit'): ").upper().strip()
        
        if ticker in ['EXIT', 'QUIT', 'Q']: 
            break

        if not ticker: 
            continue

        # Strategy selection first
        print("\nSelect Investment Strategy:")
        print("1. VALUE    (Buy the Dip - Long-term Analysis)")
        print("2. MOMENTUM (Swing Trading - Short-term Signals)")
        
        strategy_choice = input("Enter strategy choice (1-2): ").strip()
        
        if strategy_choice == '1':
            strategy_mode = "VALUE"
        elif strategy_choice == '2':
            strategy_mode = "MOMENTUM"
        else:
            print("Invalid selection. Defaulting to VALUE.")
            strategy_mode = "VALUE"

        # Show appropriate timeframes based on strategy
        print(f"\nSelect Timeframe for {strategy_mode} Strategy:")
        
        if strategy_mode == "VALUE":
            print("1. 1M  (1 Month)")
            print("2. 6M  (6 Months) - Recommended")
            print("3. YTD (Year-To-Date)")
            print("4. 1Y  (1 Year) - Best for Statistical Analysis")
            
            choice = input("Enter choice (1-4): ").strip()
            
            # Use last trading day for YTD calculation
            today = get_last_trading_day()
            
            if choice == '1':
                lookback_days = 30
            elif choice == '2':
                lookback_days = 180
            elif choice == '3':
                start_of_year = datetime(today.year, 1, 1)
                delta = today - start_of_year
                lookback_days = delta.days
            elif choice == '4':
                lookback_days = 365
            else:
                print("Invalid selection. Defaulting to 6 Months.")
                lookback_days = 180
                
        else:  # MOMENTUM strategy
            print("1. 1D  (1 Day) - Intraday Signals")
            print("2. 5D  (5 Days) - Short-term Swings")
            print("3. 1M  (1 Month) - Recommended")
            print("4. 6M  (6 Months) - Longer Trends")
            
            choice = input("Enter choice (1-4): ").strip()
            
            if choice == '1':
                lookback_days = 1
            elif choice == '2':
                lookback_days = 5
            elif choice == '3':
                lookback_days = 30
            elif choice == '4':
                lookback_days = 180
            else:
                print("Invalid selection. Defaulting to 1 Month.")
                lookback_days = 30

        analyze_stock(ticker, lookback_days, strategy_mode)

if __name__ == "__main__":
    main()