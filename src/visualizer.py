import matplotlib
try:
    matplotlib.use('TkAgg')  # Interactive windows when Tk is installed
except Exception:
    matplotlib.use('Agg')  # Headless fallback so imports do not require Tk
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime
import gc  # For garbage collection
import logging

# Pre-configure matplotlib for better performance
plt.rcParams['figure.max_open_warning'] = 0  # Disable warning for multiple figures
plt.rcParams['axes.formatter.useoffset'] = False  # Better date formatting
plt.rcParams['figure.dpi'] = 100  # Optimize DPI for performance
plt.rcParams['savefig.dpi'] = 100

def plot_graph(ticker, merged_df, company_info, timeframe_days=30, strategy_mode="VALUE"):
    """Plotting function with memory efficiency and better performance"""
    name, industry, mkt_cap = company_info
    
    # Pre-filter data to avoid unnecessary operations
    if merged_df.empty:
        logging.warning("No data to plot")
        return
    
    # Create figure with optimized settings
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                                   facecolor='white', tight_layout=True)

    # Main sentiment plot with bars - optimized data handling
    score_column = 'Compound_Score' if 'Compound_Score' in merged_df.columns else 'Buy_Score'
    
    # Create sentiment bars with vectorized color assignment
    sentiment_data = merged_df[score_column].dropna()
    if not sentiment_data.empty:
        # Vectorized color assignment using numpy
        colors = np.where(sentiment_data.values > 0, '#2E8B57', '#DC143C')
        
        # Calculate bar width based on timeframe
        bar_width = 0.02 if timeframe_days <= 1 else 0.8
        
        ax1.bar(sentiment_data.index, sentiment_data.values, color=colors, 
                alpha=0.7, width=bar_width, label='Sentiment Score', rasterized=True)

    # Style the main plot
    ax1.set_ylabel('Sentiment Score', fontsize=12, fontweight='bold')
    ax1.axhline(0, color='#696969', linewidth=1, linestyle='--', alpha=0.7)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_facecolor('#FAFAFA')

    # Price plot on secondary axis
    if 'Close' in merged_df.columns and not merged_df['Close'].isna().all():
        ax1_twin = ax1.twinx()
        price_data = merged_df['Close'].dropna()
        if not price_data.empty:
            ax1_twin.plot(price_data.index, price_data.values, color='#800080', linewidth=2.5, 
                    linestyle='-', alpha=0.8, zorder=10, label='Stock Price', rasterized=True)
            
            ax1_twin.set_ylabel('Stock Price ($)', color='#800080', fontsize=12, fontweight='bold')
            ax1_twin.tick_params(axis='y', labelcolor='#800080', labelsize=10)
            ax1_twin.grid(False)

        # Add buy/sell signals if available
        if hasattr(merged_df, 'attrs') and 'data_info' in merged_df.attrs:
            data_info = merged_df.attrs['data_info']
            
            # Plot buy signals (Golden Cross + High Buy Scores)
            if 'buy_signals' in data_info and data_info['buy_signals']:
                buy_dates, buy_prices = zip(*data_info['buy_signals'])
                ax1_twin.scatter(buy_dates, buy_prices, color='#00FF00', marker='^', 
                               s=100, alpha=0.8, zorder=15, label='BUY Signal', edgecolors='black')
            
            # Plot sell signals (Death Cross + Low Buy Scores)  
            if 'sell_signals' in data_info and data_info['sell_signals']:
                sell_dates, sell_prices = zip(*data_info['sell_signals'])
                ax1_twin.scatter(sell_dates, sell_prices, color='#FF0000', marker='v', 
                               s=100, alpha=0.8, zorder=15, label='SELL Signal', edgecolors='black')

    # Legend handling
    lines1, labels1 = ax1.get_legend_handles_labels()
    if 'Close' in merged_df.columns and not merged_df['Close'].isna().all():
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
                  frameon=True, fancybox=True, shadow=True, fontsize=10)
    else:
        ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=10)

    # Second subplot: Total Article Volume
    source_columns = [col for col in merged_df.columns if col.endswith('_Count')]
    if source_columns:
        # Vectorized sum calculation
        total_articles = merged_df[source_columns].fillna(0).sum(axis=1)
        
        if total_articles.sum() > 0:
            ax2.bar(total_articles.index, total_articles.values, 
                   color='#4682B4', alpha=0.7, width=bar_width, 
                   label='Total Articles', rasterized=True)

    # Style the second plot
    ax2.set_ylabel('Article Count', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_facecolor('#FAFAFA')
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)

    # X-axis formatting with user-friendly timezone display
    if timeframe_days <= 1:
        # For intraday, show times in Eastern Time for better user experience
        import pytz
        et_tz = pytz.timezone('US/Eastern')
        
        # Create a custom formatter that converts UTC to ET for display
        def format_time_et(x, pos):
            try:
                dt = mdates.num2date(x)
                if dt.tzinfo is None:
                    dt = pytz.utc.localize(dt)
                dt_et = dt.astimezone(et_tz)
                return dt_et.strftime('%m/%d %H:%M ET')
            except:
                return mdates.DateFormatter('%m/%d %H:%M')(x, pos)
        
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_time_et))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    else:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, timeframe_days//10)))
    
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, fontsize=10)

    # Title and info with signal count
    signal_info = ""
    if hasattr(merged_df, 'attrs') and 'data_info' in merged_df.attrs:
        data_info = merged_df.attrs['data_info']
        buy_count = len(data_info.get('buy_signals', []))
        sell_count = len(data_info.get('sell_signals', []))
        if buy_count > 0 or sell_count > 0:
            signal_info = f" • Signals: {buy_count} BUY, {sell_count} SELL"
    
    ax1.set_title(f'{name} ({ticker}) - Sentiment Analysis & Trading Signals{signal_info}', 
                  fontsize=16, fontweight='bold', pad=20)
    
    # Company info
    info_text = f"Industry: {industry}  •  Market Cap: {mkt_cap}  •  Strategy: {strategy_mode}"
    fig.text(0.5, 0.02, info_text, ha="center", fontsize=11, 
             bbox={"facecolor":"#E6E6FA", "alpha":0.8, "pad":8, "boxstyle":"round,pad=0.5"})
    
    # Layout and display
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    # Non-blocking show
    plt.show(block=False)
    plt.draw()  # Force immediate rendering
    
    # Clean up memory
    gc.collect()
        
    # Display simplified data summary
    if hasattr(merged_df, 'attrs') and 'data_info' in merged_df.attrs:
        data_info = merged_df.attrs['data_info']
        logging.info(f"\nAnalysis Complete")
        logging.info(f"Market data: {data_info['price_records']} records")
        logging.info(f"News data: {data_info['sentiment_records']} articles analyzed")
        logging.info(f"Time periods: {data_info['merged_records']} data points")