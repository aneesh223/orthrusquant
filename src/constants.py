"""
Constants for Orthrus trading logic.
"""

# SECTOR-AWARE THRESHOLD ADJUSTMENTS
# Adjustment to buy threshold based on sector/stock characteristics
# Negative = more aggressive (lower threshold), Positive = more conservative (higher threshold)
SECTOR_THRESHOLDS = {
    # Volatile Growth - Ultra aggressive (fast movers, sentiment lags)
    'TSLA': -0.20, 'NVDA': -0.20, 'AMD': -0.20, 'ROKU': -0.20,
    'PLTR': -0.20, 'ZM': -0.20, 'NFLX': -0.15,

    # Mega Cap Tech - Very aggressive (tech rallies are fast)
    'AAPL': -0.15, 'MSFT': -0.15, 'GOOGL': -0.15,
    'AMZN': -0.15, 'META': -0.15,

    # Large Cap Tech - Aggressive
    'ADBE': -0.10, 'CRM': -0.10, 'ORCL': -0.10,
    'INTC': -0.10, 'AVGO': -0.10,

    # Financials - Moderate (cyclical, moderate speed)
    'JPM': -0.05, 'BAC': -0.05, 'WFC': -0.05,
    'GS': -0.05, 'MS': -0.05, 'C': -0.05, 'BRK.B': -0.05,

    # Energy - Moderate (cyclical, regime-dependent)
    'XOM': -0.05, 'CVX': -0.05, 'COP': -0.05,
    'EOG': -0.05, 'SLB': -0.05, 'MPC': -0.05, 'VLO': -0.05,

    # Healthcare - Conservative (slow movers, wait for confirmation)
    'JNJ': +0.05, 'PFE': +0.05, 'MRK': +0.05,
    'UNH': +0.05, 'ABBV': +0.05, 'TMO': +0.05, 'DHR': +0.05,

    # Consumer - Very Conservative (defensive, slow)
    'KO': +0.10, 'PEP': +0.10, 'WMT': +0.10,
    'HD': +0.10, 'MCD': +0.10, 'NKE': +0.10, 'SBUX': +0.10,

    # Industrials - Conservative (slow, defensive)
    'BA': +0.05, 'CAT': +0.05, 'GE': +0.05,
    'MMM': +0.05, 'HON': +0.05, 'UPS': +0.05, 'RTX': +0.05,
}

# Base thresholds by strategy and regime
# OPTIMIZATION V8: Even more aggressive momentum strategy thresholds
VALUE_STRATEGY_THRESHOLDS = {
    "STRONG_BULL": (4.30, 4.10),  # Ultra-aggressive for alpha generation
    "BULL": (4.45, 4.25),  # More aggressive (was 4.70)
    ("STRONG_BEAR", "BEAR"): (5.30, 4.20),  # Slightly more aggressive
    ("CHOPPY", "CHOPPY_SIDEWAYS"): (5.80, 3.50),  # Less restrictive - allow some choppy trades
    "default": (5.00, 4.50)  # More aggressive default
}

MOMENTUM_STRATEGY_THRESHOLDS = {
    "STRONG_BULL": (4.15, 4.05),  # Even more aggressive (was 4.25)
    "BULL": (4.25, 4.15),  # Even more aggressive (was 4.40)
    ("STRONG_BEAR", "BEAR"): (5.00, 4.20),  # More aggressive (was 5.20)
    ("CHOPPY", "CHOPPY_SIDEWAYS"): (5.50, 3.70),  # Much less restrictive (was 5.70)
    "default": (4.80, 4.45)  # More aggressive default (was 4.95)
}
