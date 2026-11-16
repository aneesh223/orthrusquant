#!/usr/bin/env python3
"""
Orthrus Quick Backtesting CLI
Usage: python run_backtest.py <ticker> <strategy> [period] [cash] [--plot] [--save]

Optimal Backtesting Timeframes:
  MOMENTUM: 3-5 years (default: 5 years)
    - Requires ~10 months to warm up 200-day SMA for Golden/Death Cross
    - Captures bull runs, bear markets, and choppy periods
    - Provides 100+ trades for statistical significance
  
  VALUE: 5-10+ years (default: 10 years)
    - Tests long-term mean reversion across full market cycles
    - Validates conservative risk management during crashes
    - Captures multiple regime transitions

Examples:
  python run_backtest.py TSLA m              # TSLA momentum, default 5 years
  python run_backtest.py AAPL v              # AAPL value, default 10 years
  python run_backtest.py NVDA m 1095         # NVDA momentum, 3 years (1095 days)
  python run_backtest.py MSFT v 2015-01-01 2024-12-31  # Custom date range
  python run_backtest.py TSLA m 1825 15000 --plot --save  # 5 years, custom cash
"""

import sys
import os
import pytz
import argparse
from datetime import datetime, timedelta

# Add parent directory to path to import src modules
parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

# Import the backtester class directly from the module file
sys.path.insert(0, os.path.dirname(__file__))
from backtester import AlpacaBacktester

def get_default_period_days(strategy):
    """
    Get optimal default period in days based on strategy.
    
    MOMENTUM: 1825 days (5 years)
      - Requires ~10 months to warm up 200-day SMA
      - Captures multiple market regimes
      - Provides statistical significance (100+ trades)
    
    VALUE: 3650 days (10 years)
      - Tests long-term mean reversion
      - Validates risk management across full cycles
      - Captures multiple bull/bear transitions
    """
    if strategy.lower() == 'momentum':
        return 1825  # 5 years
    else:  # value
        return 3650  # 10 years

def parse_period(period_arg, next_arg=None, strategy='momentum'):
    """
    Parse period argument which can be:
    - None (use strategy default): None
    - Number (days): 30, 90, 1825
    - Date range: 2023-01-20 (requires next_arg as end date)
    """
    if period_arg is None:
        # Use strategy-specific default
        days = get_default_period_days(strategy)
        end_date = datetime.now(pytz.utc) - timedelta(days=15)  # 15 days buffer for API
        start_date = end_date - timedelta(days=days)
        return start_date, end_date, 0  # Return 0 to indicate no arguments consumed
    
    try:
        # Try to parse as number of days
        days = int(period_arg)
        # Use 15 days ago as end date for API stability
        end_date = datetime.now(pytz.utc) - timedelta(days=15)
        start_date = end_date - timedelta(days=days)
        return start_date, end_date, 1  # Return 1 to indicate we used 1 argument
    except ValueError:
        # Try to parse as start date (expecting end date as next argument)
        if next_arg is None:
            raise ValueError("When using date format, both start and end dates are required")
        
        try:
            start_date = pytz.utc.localize(datetime.strptime(period_arg, "%Y-%m-%d"))
            end_date = pytz.utc.localize(datetime.strptime(next_arg, "%Y-%m-%d"))
            return start_date, end_date, 2  # Return 2 to indicate we used 2 arguments
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD")

def main():
    """Command-line backtesting interface"""
    
    if len(sys.argv) < 3:
        print("Usage: python run_backtest.py <ticker> <strategy> [period] [cash] [--plot] [--no-plot] [--save]")
        print("\nOptimal Backtesting Timeframes:")
        print("  MOMENTUM: 3-5 years (default: 5 years / 1825 days)")
        print("    • Requires ~10 months to warm up 200-day SMA for Golden/Death Cross")
        print("    • Captures bull runs, bear markets, and choppy periods")
        print("    • Provides 100+ trades for statistical significance")
        print("\n  VALUE: 5-10+ years (default: 10 years / 3650 days)")
        print("    • Tests long-term mean reversion across full market cycles")
        print("    • Validates conservative risk management during crashes")
        print("    • Captures multiple regime transitions")
        print("\nExamples:")
        print("  python run_backtest.py TSLA m              # TSLA momentum, default 5 years")
        print("  python run_backtest.py AAPL v              # AAPL value, default 10 years")
        print("  python run_backtest.py NVDA m 1095         # NVDA momentum, 3 years")
        print("  python run_backtest.py MSFT v 2015-01-01 2024-12-31  # Custom date range")
        print("  python run_backtest.py TSLA m 1825 15000 --no-plot --save  # 5 years, custom cash")
        print("\nStrategy: m/momentum or v/value")
        print("Period: [optional] number of days OR start-date end-date (uses strategy default if omitted)")
        print("Flags: --plot (force plot), --no-plot (disable plot), --save (save results)")
        print("Note: Plots are shown by default for single backtests")
        return 1
    
    # Parse arguments
    ticker = sys.argv[1].upper()
    
    # Parse strategy
    strategy_arg = sys.argv[2].lower()
    if strategy_arg in ['m', 'momentum']:
        strategy = 'momentum'
    elif strategy_arg in ['v', 'value']:
        strategy = 'value'
    else:
        print("Strategy must be 'm'/'momentum' or 'v'/'value'")
        return 1
    
    # Parse period (might consume 0, 1, or 2 arguments)
    try:
        period_arg = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else None
        next_arg = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].startswith('--') else None
        
        # Check if period_arg looks like a number or date
        if period_arg and not period_arg.replace('.', '').isdigit():
            # Might be a date, check if it's a flag
            try:
                datetime.strptime(period_arg, "%Y-%m-%d")
                # It's a date, proceed normally
            except ValueError:
                # Not a date, might be cash or flag, treat as no period
                period_arg = None
        
        start_date, end_date, args_consumed = parse_period(period_arg, next_arg, strategy)
    except (ValueError, IndexError) as e:
        print(f"Period error: {e}")
        return 1
    
    # Parse remaining arguments
    remaining_args = sys.argv[3 + args_consumed:]
    
    # Parse cash (optional)
    initial_cash = 10000.0
    cash_idx = 0
    if remaining_args and not remaining_args[0].startswith('--'):
        try:
            initial_cash = float(remaining_args[0])
            cash_idx = 1
        except ValueError:
            pass
    
    # Parse flags
    flag_args = remaining_args[cash_idx:]
    show_plot = '--plot' in flag_args or '--no-plot' not in flag_args  # Default to True unless --no-plot
    save_results = '--save' in flag_args
    benchmark = None
    split_date = None
    if '--benchmark' in flag_args:
        bench_idx = flag_args.index('--benchmark')
        if bench_idx + 1 >= len(flag_args):
            print("--benchmark requires a ticker, for example SPY")
            return 1
        benchmark = flag_args[bench_idx + 1]
    if '--split-date' in flag_args:
        split_idx = flag_args.index('--split-date')
        if split_idx + 1 >= len(flag_args):
            print("--split-date requires YYYY-MM-DD")
            return 1
        split_date = pytz.utc.localize(datetime.strptime(flag_args[split_idx + 1], "%Y-%m-%d"))
    
    # Display configuration
    period_days = (end_date - start_date).days
    period_years = period_days / 365.25
    default_indicator = " (strategy default)" if args_consumed == 0 else ""
    print(f"Orthrus Backtest")
    print(f"{ticker} | {strategy.upper()} | {start_date.date()} to {end_date.date()} ({period_years:.1f} years{default_indicator}) | ${initial_cash:,.0f}")
    
    try:
        # Run backtest
        backtester = AlpacaBacktester(ticker, strategy, initial_cash, benchmark=benchmark)
        results = backtester.run(start_date, end_date, lookback_days=30, split_date=split_date)
        
        # Show plot by default (unless --no-plot is specified)
        if show_plot:
            backtester.plot_results(results)
        
        # Save results if requested
        if save_results:
            import json
            filename = f"backtest_{ticker}_{strategy}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
            
            # Convert datetime objects to strings for JSON serialization
            results_copy = results.copy()
            results_copy['start_date'] = results_copy['start_date'].isoformat()
            results_copy['end_date'] = results_copy['end_date'].isoformat()
            
            # Convert portfolio history dates
            for item in results_copy['portfolio_history']:
                item['Date'] = item['Date'].isoformat()
            
            # Convert trade log dates
            for item in results_copy['trade_log']:
                item['Date'] = item['Date'].isoformat()
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results_copy, f, indent=2, default=str)
            
            print(f"Results saved to: {filename}")
        
        return 0
        
    except Exception as e:
        print(f"Backtesting failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())