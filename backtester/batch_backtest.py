#!/usr/bin/env python3
"""
Orthrus Random Batch Backtesting
Randomly selects from pools of tickers, strategies, and date ranges for comprehensive testing
Features parallel execution with configurable threading for faster batch processing
"""

import csv
import json
import sys
import os
import random
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def get_random_date_range(strategy=None):
    """
    Generate random start and end dates within Alpaca free tier support
    Respects strategy-specific timeframe restrictions from input.py
    
    Args:
        strategy: 'm' for momentum, 'v' for value (affects max period length)
    """
    # Define the available date range (Alpaca free tier with good data quality)
    start_year = 2020
    end_year = 2024
    
    # Generate random start date
    start_date = datetime(
        year=random.randint(start_year, end_year - 1),
        month=random.randint(1, 12),
        day=random.randint(1, 28)  # Use 28 to avoid month-end issues
    )
    
    # Strategy-specific timeframe restrictions (matching input.py)
    if strategy == 'm':  # MOMENTUM strategy
        # Allowed: 1D, 5D, 1M, 6M (1-180 days max)
        min_days = 1
        max_days = 180
        # Common periods: 1, 5, 30, 180 days
        period_options = [1, 5, 30, 60, 90, 120, 150, 180]
        period_days = random.choice(period_options)
    elif strategy == 'v':  # VALUE strategy  
        # Allowed: 1M, 6M, YTD, 1Y (30-365 days)
        min_days = 30
        max_days = 365
        # Common periods: 30, 90, 180, 270, 365 days
        period_options = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365]
        period_days = random.choice(period_options)
    else:
        # Fallback for backward compatibility
        min_days = 30
        max_days = 365
        period_days = random.randint(min_days, max_days)
    
    end_date = start_date + timedelta(days=period_days)
    
    # Ensure end date doesn't exceed our range
    if end_date.year > end_year:
        end_date = datetime(end_year, 12, 31)
    
    # Ensure we don't go into the future (leave some buffer)
    max_end_date = datetime.now() - timedelta(days=15)  # 15 days buffer for API limits
    if end_date > max_end_date:
        end_date = max_end_date
        # Adjust start date if needed
        if (end_date - start_date).days < min_days:
            start_date = end_date - timedelta(days=min_days)
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def get_ticker_pools():
    """Define pools of tickers by category for diverse testing"""
    return {
        'mega_cap_tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
        'large_cap_tech': ['NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'AVGO'],
        'financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK.B'],
        'healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'DHR'],
        'consumer': ['KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX'],
        'energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO'],
        'industrials': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX'],
        'volatile_growth': ['TSLA', 'NVDA', 'AMD', 'NFLX', 'ZOOM', 'ROKU', 'PLTR']
    }

def select_random_parameters():
    """Randomly select ticker, strategy, and date range with strategy-aware timeframes"""
    ticker_pools = get_ticker_pools()
    
    # Randomly select a category and then a ticker from that category
    category = random.choice(list(ticker_pools.keys()))
    ticker = random.choice(ticker_pools[category])
    
    # Randomly select strategy
    strategy = random.choice(['m', 'v'])  # momentum or value
    
    # Generate random date range respecting strategy timeframe restrictions
    start_date, end_date = get_random_date_range(strategy)
    
    # Create description
    strategy_name = "MOMENTUM" if strategy == 'm' else "VALUE"
    description = f"RANDOM TEST: {ticker} {strategy_name} strategy ({category.replace('_', ' ').title()})"
    
    return ticker, strategy, start_date, end_date, description, category

SUMMARY_FIELDS = [
    "success", "ticker", "strategy", "category", "start_date", "end_date", "period",
    "total_return_pct", "buy_hold_return_pct", "percent_alpha", "cagr_pct", "sharpe",
    "max_drawdown_pct", "num_trades", "benchmark_ticker", "benchmark_return_pct",
    "sentiment_coverage_pct", "error",
]


def _iso(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def summarize_result(result):
    """One JSON/CSV row from a structured backtest result. No stdout scraping."""
    row = {field: result.get(field) for field in SUMMARY_FIELDS}
    row["success"] = bool(result.get("success"))
    row["start_date"] = _iso(result.get("start_date"))
    row["end_date"] = _iso(result.get("end_date"))
    if row["percent_alpha"] is None and result.get("total_return_pct") is not None and result.get("buy_hold_return_pct") is not None:
        row["percent_alpha"] = result["total_return_pct"] - result["buy_hold_return_pct"]
    if row["num_trades"] is None and result.get("trades") is not None:
        row["num_trades"] = result.get("trades")
    return row


def aggregate_percent_alpha(results):
    alphas = [
        r["percent_alpha"] for r in results
        if r.get("success") and r.get("percent_alpha") is not None
    ]
    if not alphas:
        return {"count": 0, "mean_percent_alpha": None, "positive_alpha": 0}
    return {
        "count": len(alphas),
        "mean_percent_alpha": sum(alphas) / len(alphas),
        "positive_alpha": sum(1 for value in alphas if value > 0),
    }


def write_batch_results(path_prefix, results, seed, params):
    """Write structured rows plus the seed and parameters to JSON and CSV."""
    rows = [summarize_result(r) for r in results]
    payload = {
        "seed": seed,
        "params": params,
        "summary": aggregate_percent_alpha(rows),
        "results": rows,
    }
    json_path = path_prefix if str(path_prefix).endswith(".json") else f"{path_prefix}.json"
    csv_path = json_path[:-5] + ".csv"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return json_path, csv_path


def run_backtest(ticker, strategy, start_date, end_date, description="", print_header=True,
                 initial_cash=10000.0, commission_bps=5.0, slippage_bps=5.0, benchmark=None,
                 news_lookback_days=None):
    """Run one backtest in-process and return its result dict."""
    if print_header:
        print(f"\n{'='*80}")
        print(f"TEST: {ticker} {strategy.upper()} | {start_date} to {end_date}")
        if description:
            print(f"{description}")
        print('='*80)

    parent = os.path.join(os.path.dirname(__file__), "..")
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from backtester.backtester import AlpacaBacktester

    strategy_name = "momentum" if str(strategy).lower() in ("m", "momentum") else "value"
    try:
        engine = AlpacaBacktester(
            ticker,
            strategy=strategy_name,
            initial_cash=initial_cash,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            benchmark=benchmark,
            news_lookback_days=news_lookback_days,
        )
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        results = engine.run(start, end)
        results["success"] = True
        results["percent_alpha"] = results["total_return_pct"] - results["buy_hold_return_pct"]
        results["strategy"] = strategy_name.upper() if len(strategy_name) > 1 else strategy_name
        results["period"] = f"{start_date} to {end_date}"
        results["description"] = description
        results["num_trades"] = results.get("num_trades", len(results.get("trade_log", [])))
        return results
    except Exception as exc:
        print(f"ERROR: {exc}")
        return {
            "success": False,
            "ticker": ticker,
            "strategy": strategy_name.upper(),
            "error": str(exc),
            "description": description,
            "period": f"{start_date} to {end_date}",
        }

def main():
    """Run random batch tests with threading support"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Orthrus Random Batch Backtesting')
    parser.add_argument('num_tests', type=int, nargs='?', default=10, 
                       help='Number of random backtests to run (default: 10)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible results')
    parser.add_argument('--threads', type=int, default=4, 
                       help='Number of concurrent threads (default: 4)')
    parser.add_argument('--benchmark', default=None,
                       help='Benchmark ticker bought at the same next-open convention (for example SPY)')
    parser.add_argument('--output', default='batch_results',
                       help='Prefix for the JSON and CSV result files (default: batch_results)')
    parser.add_argument('--commission-bps', type=float, default=5.0,
                       help='Commission in basis points per side (default: 5)')
    parser.add_argument('--slippage-bps', type=float, default=5.0,
                       help='Slippage in basis points per side (default: 5)')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    print("ORTHRUS RANDOM BATCH BACKTESTING")
    print(f"Running {args.num_tests} random backtests across diverse market conditions")
    print(f"Using {args.threads} concurrent threads for parallel execution")
    print("=" * 80)
    
    # Generate all test parameters upfront for thread safety
    test_params = []
    for i in range(args.num_tests):
        ticker, strategy, start_date, end_date, description, category = select_random_parameters()
        test_params.append((i+1, ticker, strategy, start_date, end_date, description, category))
    
    results = []
    successful = 0
    completed = 0
    
    # Thread-safe print lock
    print_lock = threading.Lock()
    
    def run_test_with_progress(params):
        """Wrapper function to run test with progress tracking"""
        nonlocal completed, successful
        
        test_num, ticker, strategy, start_date, end_date, description, category = params
        
        with print_lock:
            print(f"\n[TEST {test_num}/{args.num_tests}] Starting: {ticker} {strategy.upper()}")
            print(f"{'='*80}")
            print(f"TEST: {ticker} {strategy.upper()} | {start_date} to {end_date}")
            if description:
                print(f"{description}")
            print('='*80)
        
        # Run the backtest without printing header (we already printed it above)
        result = run_backtest(
            ticker, strategy, start_date, end_date, description, print_header=False,
            commission_bps=args.commission_bps, slippage_bps=args.slippage_bps,
            benchmark=args.benchmark,
        )
        result['category'] = category  # Add category for analysis
        result['test_num'] = test_num
        
        with print_lock:
            completed += 1
            if result['success']:
                successful += 1
                # Show quick result
                percent_alpha = result.get('percent_alpha')
                alpha_text = "n/a" if percent_alpha is None else f"{percent_alpha:+.2f}%"
                trades = result.get('num_trades', 'N/A')
                print(f"✅ [{test_num}/{args.num_tests}] {ticker}: {alpha_text} percent alpha, {trades} trades ({completed}/{args.num_tests} complete)")
            else:
                print(f"❌ [{test_num}/{args.num_tests}] {ticker}: Failed - {result.get('error', 'Unknown error')} ({completed}/{args.num_tests} complete)")
        
        return result
    
    # Execute tests in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Submit all jobs
        future_to_params = {executor.submit(run_test_with_progress, params): params for params in test_params}
        
        # Collect results as they complete
        for future in as_completed(future_to_params):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                params = future_to_params[future]
                test_num, ticker, strategy = params[0], params[1], params[2]
                with print_lock:
                    print(f"❌ [{test_num}/{args.num_tests}] {ticker}: Exception - {str(e)}")
                results.append({
                    'success': False, 
                    'ticker': ticker, 
                    'strategy': strategy.upper(), 
                    'error': str(e),
                    'test_num': test_num
                })
    
    # Sort results by test number for consistent output
    results.sort(key=lambda x: x.get('test_num', 0))
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print(f"RANDOM BATCH TEST RESULTS")
    print(f"{'='*80}")
    print(f"Successful Tests: {successful}/{args.num_tests} ({successful/args.num_tests*100:.1f}%)")
    print()
    
    # Analyze results by category
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        # Group by category
        categories = {}
        for result in successful_results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)
        
        # Print category analysis
        for category, cat_results in categories.items():
            print(f"\n{category.replace('_', ' ').upper()} ({len(cat_results)} tests)")
            print("-" * 60)
            
            for result in cat_results:
                percent_alpha = result.get('percent_alpha')
                if percent_alpha is None:
                    alpha_color = "🟡"
                    alpha_text = "n/a"
                else:
                    alpha_color = "🟢" if percent_alpha > 0 else "🔴" if percent_alpha < -1 else "🟡"
                    alpha_text = f"{percent_alpha:+.2f}%"
                
                print(f"{alpha_color} {result['ticker']} {result['strategy']}: {alpha_text} percent alpha | {result.get('num_trades')} trades | {result.get('initial_regime')}")
                print(f"   Strategy: {result.get('total_return_pct')}% vs Buy-Hold: {result.get('buy_hold_return_pct')}%")
                print(f"   Period: {result['period']}")
                print()
        
        summary = aggregate_percent_alpha(successful_results)
        momentum = [r['percent_alpha'] for r in successful_results if r.get('strategy') in ('M', 'MOMENTUM') and r.get('percent_alpha') is not None]
        value = [r['percent_alpha'] for r in successful_results if r.get('strategy') in ('V', 'VALUE') and r.get('percent_alpha') is not None]
        
        print(f"\nOVERALL STATISTICS")
        print("-" * 40)
        if summary["count"]:
            mean_alpha = summary["mean_percent_alpha"]
            print(f"Average Percent Alpha: {mean_alpha:+.2f}%")
            print(f"Positive Alpha: {summary['positive_alpha']}/{summary['count']} ({summary['positive_alpha']/summary['count']*100:.1f}%)")
            if momentum:
                print(f"Momentum Strategy Avg: {sum(momentum)/len(momentum):+.2f}% ({len(momentum)} tests)")
            if value:
                print(f"Value Strategy Avg: {sum(value)/len(value):+.2f}% ({len(value)} tests)")
        else:
            print("No percent-alpha figures to aggregate")
    
    params = {
        "num_tests": args.num_tests,
        "threads": args.threads,
        "benchmark": args.benchmark,
        "commission_bps": args.commission_bps,
        "slippage_bps": args.slippage_bps,
        "seed": args.seed,
    }
    json_path, csv_path = write_batch_results(args.output, results, args.seed, params)
    print(f"\nWrote {json_path} and {csv_path}")

    print(f"\n{'='*80}")
    print("TIP: Run with --seed <number> for reproducible results")
    print("TIP: Use --threads <number> to control parallel execution (default: 4)")
    print("TIP: Increase number of tests for more comprehensive analysis")
    
    return 0 if successful > 0 else 1

if __name__ == "__main__":
    exit(main())