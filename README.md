# Orthrus: AI-Powered Algorithmic Trading Engine

**An advanced algorithmic trading platform that combines DistilRoBERTa transformer models with sophisticated technical analysis to generate data-driven investment insights. Features comprehensive backtesting, adaptive risk management, and market regime detection for systematic trading strategies.**

## What Makes Orthrus Special

- **Hybrid AI Sentiment Analysis (Hence the Name)**: Combines VADER + DistilRoBERTa for superior financial news interpretation
- **Adaptive Market Regime Detection**: Automatically adjusts strategy based on BULL/BEAR/SIDEWAYS market conditions  
- **Comprehensive Testing Framework**: Extensive backtesting across multiple market scenarios and timeframes
- **Advanced Risk Management**: Dynamic stop-loss, take-profit, and volatility protection systems
- **Production-Ready**: Professional-grade backtesting system with fair performance comparisons

## Features

### Convolutional Order Book - Vision Engine
- **CNN-Based Microstructure Analysis**: Converts 1-minute OHLCV data into 64x64 liquidity heatmaps
- **Pattern Detection**: Identifies liquidity voids, accumulation zones, and microstructure volatility
- **Real-Time Analysis**: Processes intraday data to detect market anomalies
- **Anomaly Scoring**: 0.0-1.0 scale with interpretable thresholds:
  - **> 0.85**: Critical liquidity void detected (dangerous conditions)
  - **0.70-0.85**: High microstructure volatility (risky)
  - **0.20-0.70**: Normal market conditions
  - **< 0.20**: Strong accumulation zone detected (bullish signal)
- **GPU Acceleration**: Automatic CUDA detection for faster inference
- **Lazy Initialization**: Model loaded on first use to minimize startup time

### Dual Trading Strategies
- **VALUE Strategy**: Rolling window Final_Buy_Score analysis for identifying statistically oversold opportunities
- **MOMENTUM Strategy**: MACD/RSI combination with Golden Cross/Death Cross detection for swing trading and trend following

### Revolutionary Hybrid Sentiment Analysis
- **Hybrid AI System**: Combines VADER's nuanced scoring (-1 to +1) with DistilRoBERTa's financial accuracy
- **Enhanced Financial Lexicon**: 40+ custom financial terms (crash: -3.0, soar: +3.0, beat: +2.5, etc.)
- **Smart Disagreement Resolution**: When models disagree, uses confidence-weighted correction
- **Source Reliability Weighting**: Four-tier categorization system (Primary, Institutional, Aggregators, Entertainment)
- **Real-Time News**: Alpaca News API integration with intelligent filtering

### Advanced Technical Analysis Algorithms
- **Rolling Window Analysis**: 3-5 day rolling window with aggressive recency weighting (decay_rate=0.5)
- **B(t) Calculation**: Individual buy scores at each time period with exponential decay
- **Final_Buy_Score**: Weighted average of recent B(t) values for stable signals
- **MACD**: Moving Average Convergence Divergence with bullish crossover detection
- **RSI**: Relative Strength Index for momentum assessment
- **Golden Cross/Death Cross**: SMA(50) vs SMA(200) major trend signals
- **Buy/Sell Signal Visualization**: Green/red triangles on price charts
- **Adaptive Risk Management**: Dynamic thresholds, stop-loss, take-profit, and trailing stops

### Interactive Visualization
- **Real-Time Charts**: Combined sentiment and price data visualization with trading signals
- **Dual Interface**: Interactive mode (`input.py`) and headless CLI mode (`headless.py`)
- **Signal Markers**: Visual buy/sell indicators on matplotlib charts
- **Source Breakdown**: Article count by reliability tier
- **Multiple Timeframes**: 1D, 5D, 1M, 6M analysis periods with strategy-specific restrictions
- **Performance Optimized**: Memory-efficient processing with intelligent caching

### Concrete Trading Recommendations
- **BUY/HOLD/SELL Signals**: Clear actionable recommendations with confidence percentages
- **Market Regime Detection**: Automatic BULL/BEAR/SIDEWAYS market identification
- **Adaptive Thresholds**: Dynamic buy/sell thresholds that adjust to market conditions
- **Risk Management**: Regime-specific stop-loss and take-profit recommendations
- **Backtester-Proven Logic**: Uses the same algorithms validated in historical testing

### Comprehensive Backtesting System
- **Historical Analysis**: Test strategies against years of Alpaca market data
- **Realistic Simulation**: Day-by-day execution with no look-ahead bias
- **Advanced Risk Management**: Adaptive stop-loss, take-profit, and trailing stops based on market regime
- **Market Regime Detection**: Automatic adaptation to BULL/BEAR/SIDEWAYS markets with volatility tolerance
- **Performance Analytics**: Fair buy-and-hold comparison (always starts Day 1) with alpha calculation
- **Bull Market Optimization**: Volatility tolerance system for maximum bull market participation
- **CLI Interface**: Streamlined command-line backtesting with comprehensive batch testing

## Technical Specifications

### High-Performance Architecture

**Parallel Data Ingestion**
- `concurrent.futures.ThreadPoolExecutor` for asynchronous API calls (Alpaca + GNews)
- Parallel news fetching reduces latency by ~50% compared to sequential requests
- Thread-safe batch backtesting with configurable worker pools (default: 4 threads)

**Intelligent Caching System**
- `@lru_cache` decorators for function-level memoization (1000+ entry capacity)
- Thread-safe global caches with `threading.Lock` for concurrent access
- Sub-millisecond signal retrieval for cached sentiment analysis
- Cache hit rate tracking: typically 70-90% hit rate after warm-up

**Vectorized Computations**
- NumPy/Pandas vectorized operations for MACD, RSI, Z-Score calculations
- Batch sentiment processing (64 headlines/batch) with GPU acceleration support
- `np.where()` for conditional operations, `np.array()` for batch transformations
- Rolling window calculations using pandas `.rolling()` and `.ewm()` methods

**Memory Optimization**
- Float32 precision for price data (50% memory reduction vs float64)
- Categorical dtypes for source classification
- Lazy model loading (DistilRoBERTa loaded on first use)
- Automatic GPU memory cleanup with `torch.cuda.empty_cache()`

### Convolutional Order Book Architecture

**CNN Model Design**
```
Input: (batch, 1, 64, 64) grayscale liquidity heatmap
├─ Conv2d(1→16, kernel=3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.3)
├─ Conv2d(16→32, kernel=3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.3)
├─ Flatten → Linear(8192→1) → Sigmoid
Output: (batch, 1) anomaly score ∈ [0, 1]
```

**Heatmap Generation**
- Converts 1-minute OHLCV data to 64×64 spatial representation
- Y-axis: Price levels (low to high)
- X-axis: Time steps (chronological)
- Pixel intensity: log(1 + volume), normalized to [0, 1]
- Handles sparse data with zero-padding for insufficient history

**Performance Characteristics**
- Inference time: ~10-50ms per ticker (CPU), ~5-15ms (GPU)
- Memory footprint: ~50MB for model weights
- Lazy initialization: Model loaded only when `analyze_liquidity()` is called
- Device detection: Automatic CUDA/CPU selection

**Integration Points**
- Called from `src.market.get_intraday_data()` for real-time analysis
- Returns structured dict: `{'anomaly_score': float, 'status': str, 'confidence': str}`
- Graceful degradation: Returns neutral score (0.5) on data insufficiency

### Shared Logic Architecture
The system uses a unified logic module (`src/logic.py`) that ensures consistency between the main program and backtester:

- **Single Source of Truth**: All trading algorithms are centralized in one module
- **Automatic Synchronization**: Changes to logic automatically apply to both systems
- **Proven Algorithms**: Logic is validated through extensive backtesting before deployment
- **Maintainable Code**: No duplication of complex trading algorithms

### Time-Series Algorithm Architecture
The system uses a sophisticated rolling window approach:

1. **B(t) Calculation**: Individual buy scores at each time period
   ```
   B(t) = Weighted average of sentiment scores at time t
   Recency weighting: exp(-decay_rate × age_in_days)
   ```

2. **Final_Buy_Score Calculation**: Rolling window analysis
   ```
   Final_Buy_Score(t) = Σ(B(i) × weight(i)) for i in [t-window, t]
   Window size: 3-5 days with aggressive decay (0.5)
   ```

3. **Market Regime Adaptation**:
   - **BULL Market**: Wider stop-loss (-40%), higher take-profit (up to 600%), larger positions (up to 3.6x)
   - **STRONG_BULL**: Maximum aggressiveness with bull market duration scaling
   - **BEAR Market**: Tighter stop-loss (-6%), lower take-profit (+12%), smaller positions (0.6x)
   - **SIDEWAYS**: Standard parameters with dynamic threshold adjustment
   - **Volatility Override**: Extreme volatility protection with bull market tolerance

### Hybrid Sentiment Analysis System
The revolutionary hybrid approach combines two complementary AI models:

1. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**:
   - Provides nuanced scoring from -1.0 to +1.0
   - Enhanced with custom financial lexicon (40+ terms)
   - Fast processing for real-time analysis (~1ms per headline)

2. **DistilRoBERTa Financial Model**:
   - Fine-tuned for financial news classification (mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis)
   - High accuracy for positive/negative/neutral detection
   - Neutral dampening logic (if Neutral > 0.6, multiply score by 0.2)
   - GPU-accelerated batch processing (64 headlines/batch)

3. **Hybrid Logic**:
   ```
   If models agree: Use VADER's nuanced score
   If models disagree: Apply confidence-weighted correction
   Formula: abs(vader_score) × ai_confidence × 0.7
   ```

4. **Performance Optimizations**:
   - Global sentiment cache with hash-based lookups (O(1) retrieval)
   - Batch processing with `get_hybrid_sentiment_batch()` for throughput
   - Lazy model initialization (loaded on first use)
   - Automatic GPU detection and utilization

### Financial Algorithms Used
1. **Convolutional Order Book CNN**: 2-layer CNN (Conv2d→BatchNorm→ReLU→MaxPool→Dropout) for liquidity heatmap analysis
2. **Rolling Window Final_Buy_Score**: 3-5 day window with exponential decay (decay_rate=0.5)
3. **MACD**: 12-day EMA - 26-day EMA with 9-day signal line (vectorized with pandas `.ewm()`)
4. **RSI**: 14-day momentum oscillator (vectorized with `np.where()` for gain/loss separation)
5. **Golden Cross**: SMA(50) > SMA(200) (+2.5 momentum points)
6. **Death Cross**: SMA(50) < SMA(200) (-2.5 momentum points)
7. **Z-Score**: Statistical deviation from rolling mean (vectorized with pandas `.rolling()`)
8. **Hybrid Sentiment**: VADER + DistilRoBERTa with financial lexicon
9. **Adaptive Risk Management**: Market regime detection with volatility-adjusted parameters
10. **Bull Market Duration Scaling**: Progressive aggressiveness in sustained bull markets
11. **Momentum Reversal Detection**: Dynamic regime updates based on price/sentiment divergence

### Source Reliability Tiers
- **PRIMARY** (Weight: 2.0): SEC filings, Company IR
- **INSTITUTIONAL** (Weight: 1.6-1.8): Bloomberg, WSJ, Reuters
- **AGGREGATORS** (Weight: 1.0-1.5): Yahoo Finance, MarketWatch, Benzinga
- **ENTERTAINMENT** (Weight: 0.3-0.8): Motley Fool, Reddit, Social Media

## Security & API Keys

**IMPORTANT**: This project requires Alpaca API keys for market data access.

### **Getting API Keys (Free)**
1. Sign up at [Alpaca Markets](https://app.alpaca.markets/signup)
2. Navigate to Paper Trading Dashboard
3. Generate API keys (paper trading is free)
4. Copy `.env.example` to `.env` and add your keys

### **Security Best Practices**
- API keys are loaded via environment variables (secure)
- `.env` file is in `.gitignore` (not committed to repo)
- Uses paper trading endpoint (no real money at risk)
- **Never commit real API keys to version control**

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/orthrus.git
   cd orthrus
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API keys** (create `.env` file)
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env with your actual Alpaca API keys
   # Get free keys at: https://app.alpaca.markets/paper/dashboard/overview
   ALPACA_API_KEY=your_actual_key_here
   ALPACA_SECRET_KEY=your_actual_secret_here
   ```

## Usage

### Real-Time Analysis

#### Interactive Mode (Recommended)
1. Run the program: `python src/main/input.py`
2. Enter a stock ticker (e.g., TSLA, AAPL, MSFT)
3. **Select strategy first** (VALUE or MOMENTUM)
4. Choose timeframe based on strategy restrictions
5. View results and interactive charts with trading signals

#### Headless Mode (CLI/Automation)
1. **Command-line analysis**: `python src/main/headless.py <ticker> <strategy> <period> [--save]`
2. **Examples**:
   ```bash
   # Quick analysis without GUI (run from project root)
   python src/main/headless.py TSLA m 30
   
   # Save chart to file
   python src/main/headless.py AAPL v 90 --save
   ```
3. **Perfect for**: Automation, scripting, server environments, or when you don't want GUI popups

### Historical Backtesting

## Optimal Backtesting Timeframes

**Critical**: Different strategies require different testing periods to validate their mathematical edge. Using the wrong timeframe will produce misleading results.

### MOMENTUM Strategy: 3-5 Years (Recommended: 5 Years)

**Why This Matters:**
1. **200-Day SMA Warm-Up**: The Golden Cross/Death Cross signals rely on a 200-day moving average. The first ~10 months of any backtest are just "warming up" this indicator, so a 1-year test only yields 2 months of actual data.

2. **Full Market Cycle Coverage**: A 3-5 year window perfectly captures:
   - A parabolic bull run (2020-2021 style)
   - A severe bear market (2022 crash)
   - Choppy sideways consolidation periods
   
   This tests the dynamic threshold system across all market regimes.

3. **Statistical Significance**: Momentum swing trades trigger less frequently than high-frequency strategies. A 3-5 year window guarantees 100+ trades per ticker, providing the sample size needed to prove the mathematical edge.

**Implementation:**
```bash
# Use strategy default (5 years)
python backtester/run_backtest.py TSLA m

# Minimum recommended (3 years = 1095 days)
python backtester/run_backtest.py NVDA m 1095

# Optimal period (5 years = 1825 days)
python backtester/run_backtest.py AAPL m 1825

# Custom date range (5 years)
python backtester/run_backtest.py GOOGL m 2019-01-01 2024-01-01
```

### VALUE Strategy: 5-10+ Years (Recommended: 10 Years)

**Why This Matters:**
1. **Long-Term Mean Reversion**: Value investing and contrarian plays rely on long-term mean reversion. Market cycles take years to play out - a stock can stay "oversold" for 18+ months before reverting.

2. **Risk Management Validation**: The VALUE strategy emphasizes capital preservation and conservative risk management. Testing on shorter timeframes only captures a single market mood and won't accurately test how the system performs during actual market crashes (2008, 2020, 2022).

3. **Multiple Regime Transitions**: A 10-year period captures:
   - Multiple bull/bear market transitions
   - Various economic cycles (expansion, recession, recovery)
   - Different volatility regimes
   - Real stress-testing of stop-loss and position sizing

**Implementation:**
```bash
# Use strategy default (10 years)
python backtester/run_backtest.py AAPL v

# Minimum recommended (5 years = 1825 days)
python backtester/run_backtest.py MSFT v 1825

# Optimal period (10 years = 3650 days)
python backtester/run_backtest.py GOOGL v 3650

# Extended testing (15 years = 5475 days)
python backtester/run_backtest.py JPM v 5475

# Custom date range (10 years)
python backtester/run_backtest.py AMZN v 2014-01-01 2024-01-01
```

### Quick Comparison Table

| Strategy | Minimum | Recommended | Maximum | Why |
|----------|---------|-------------|---------|-----|
| **MOMENTUM** | 3 years (1095 days) | 5 years (1825 days) | 10 years | SMA warm-up + full cycle coverage |
| **VALUE** | 5 years (1825 days) | 10 years (3650 days) | MAX | Mean reversion + crash testing |

### Batch Testing Note

The `batch_backtest.py` tool uses randomized periods (30-365 days) by default for **diversity testing** - validating the algorithm works across various market conditions. For **production validation** and proving the mathematical edge, always use the optimal periods above.

```bash
# Diverse testing (30-365 days, multiple scenarios)
python backtester/batch_backtest.py 50 --seed 123

# Optimal period testing (3-10 years, strategy-aware)
python backtester/batch_backtest.py 50 --seed 123 --optimal-periods
```

#### Quick Start
```bash
# Basic backtest with strategy defaults
python backtester/run_backtest.py TSLA m    # 5 years for momentum
python backtester/run_backtest.py AAPL v    # 10 years for value

# Custom period
python backtester/run_backtest.py NVDA m 1095  # 3 years

# With custom capital and visualization
python backtester/run_backtest.py MSFT v 3650 25000 --plot --save  # 10 years
```

#### Command Structure
```bash
python backtester/run_backtest.py <ticker> <strategy> [period] [cash] [--plot] [--save]
```

**Parameters:**
- **ticker**: Stock symbol (TSLA, AAPL, NVDA, MSFT, etc.)
- **strategy**: `v` (VALUE) or `m` (MOMENTUM)
- **period**: [OPTIONAL] Days (1095, 1825, 3650) OR date range (2019-01-01 2024-01-01)
  - If omitted, uses strategy default (momentum=5yr, value=10yr)
- **cash**: Starting capital (default: $10,000)
- **--plot**: Show performance charts
- **--no-plot**: Disable charts (useful for batch testing)
- **--save**: Save results to JSON file

#### Advanced Testing
```bash
# Random batch testing with diverse periods (30-365 days)
python backtester/batch_backtest.py           # Run 10 random backtests
python backtester/batch_backtest.py 25        # Run 25 random backtests  
python backtester/batch_backtest.py 50 --seed 123  # Reproducible results

# Optimal period testing (strategy-aware: momentum=3-5yr, value=5-10yr)
python backtester/batch_backtest.py 50 --seed 123 --optimal-periods

# View all usage examples and tips
python backtester/examples.py
```

### Strategy-Specific Timeframes
- **VALUE Strategy**: 1M, 6M, YTD, MAX (long-term analysis)
- **MOMENTUM Strategy**: 1D, 5D, 1M, 6M (short to medium-term signals)

### Convolutional Order Book Usage

The CNN-based microstructure analysis is **automatically integrated** into the trading recommendation system. It runs in real-time during analysis and adjusts confidence levels based on market microstructure conditions.

**Automatic Integration:**
```python
# When you run the main program, microstructure analysis happens automatically
python src/main/input.py

# Enter ticker: AAPL
# The system will:
# 1. Fetch 1-minute intraday data
# 2. Generate liquidity heatmap
# 3. Run CNN inference
# 4. Adjust trading confidence based on anomaly score
```

**Manual Usage:**
```python
from src.microstructure import analyze_liquidity

# Analyze market microstructure for a ticker
result = analyze_liquidity('TSLA')

print(f"Anomaly Score: {result['anomaly_score']:.3f}")
print(f"Status: {result['status']}")
print(f"Confidence: {result['confidence']}")

# Interpretation:
# - Score > 0.85: Critical liquidity void (avoid trading)
# - Score 0.70-0.85: High volatility (use caution)
# - Score 0.20-0.70: Normal conditions
# - Score < 0.20: Strong accumulation (bullish signal)
```

**Impact on Trading Recommendations:**
- **BUY signals** with anomaly score > 0.85: Confidence reduced by 30%
- **BUY signals** with anomaly score > 0.70: Confidence reduced by 15%
- **BUY signals** with anomaly score < 0.20: Confidence boosted by 20%
- **SELL signals** with anomaly score > 0.85: Confidence boosted by 15% (confirmation)

**Requirements:**
- Minimum 10 minutes of 1-minute intraday data
- Automatically fetches data via `get_intraday_data(ticker)`
- GPU acceleration used if available (CUDA)

**Limitations:**
- Only works for real-time trading (requires current intraday data)
- Not available for historical backtesting (no historical 1-minute data)
- Gracefully degrades if data unavailable (doesn't break existing logic)

### Output Interpretation
- **Market Sentiment**: 0-10 scale based on hybrid AI sentiment analysis
- **Market Analysis**: 0-10 based on chosen strategy algorithms
- **Final Buy Score**: Combined score with rolling window analysis
- **Trading Signal**: 🟢 BUY / 🟡 HOLD / 🔴 SELL with confidence percentage
- **Market Regime**: BULL / BEAR / SIDEWAYS detection for adaptive thresholds
- **Adaptive Thresholds**: Dynamic buy/sell thresholds based on market conditions
- **Risk Management**: Regime-specific stop-loss and take-profit recommendations
- **Top Headlines**: Best/worst sentiment-scored news with confidence scores

## Strategy Details

### VALUE Strategy
- **Objective**: Identify statistically oversold stocks using rolling window analysis
- **Algorithm**: Final_Buy_Score with 5-day rolling window and recency weighting
- **Timeframes**: 1M, 6M, YTD, MAX
- **Best For**: Long-term value investing, contrarian plays
- **Risk Management**: Conservative thresholds with capital preservation focus

### MOMENTUM Strategy  
- **Objective**: Capture trend continuation, breakouts, and major trend changes
- **Signals**: 
  - MACD bullish crossover + RSI not overbought
  - Golden Cross (SMA50 > SMA200) = major bullish signal
  - Death Cross (SMA50 < SMA200) = major bearish signal
- **Timeframes**: 1D, 5D, 1M, 6M
- **Best For**: Swing trading, trend following, momentum plays
- **Risk Management**: Dynamic thresholds with aggressive profit-taking

## Backtesting System

### Comprehensive Strategy Testing
- **Location**: `backtester/` folder
- **Capability**: Test strategies against historical Alpaca data with realistic simulation
- **Features**:
  - Real Alpaca market data integration (11+ years of data back to 2014)
  - Day-by-day simulation with no look-ahead bias
  - Pre-calculated B(t) and Final_Buy_Score for maximum efficiency
  - Advanced adaptive risk management with market regime detection
  - Bull market volatility tolerance system for enhanced participation
  - Fair buy-and-hold comparison methodology
  - Comprehensive testing framework across multiple market scenarios
  - Performance analysis with detailed categorized results
  - Visual performance charts and P&L tracking
  - **Parallel execution**: ThreadPoolExecutor for concurrent backtests (4 threads default)
  - **Thread-safe logging**: Synchronized output for parallel test runs

### Quick Reference

#### Basic Command Structure
```bash
python backtester/run_backtest.py <ticker> <strategy> <period> [cash] [--plot] [--save]
```

#### Parameters
- **ticker**: Stock symbol (TSLA, AAPL, NVDA, MSFT, etc.)
- **strategy**: `m` (momentum) or `v` (value)
- **period**: Days (30, 90, 180) OR date range (2023-01-20 2023-07-20)
- **cash**: Starting capital (default: $10,000)
- **--plot**: Show performance charts
- **--save**: Save results to JSON file

#### Usage Examples
```bash
# Basic momentum test (30 days)
python backtester/run_backtest.py TSLA m 30

# Value strategy with custom capital
python backtester/run_backtest.py AAPL v 180 25000

# Historical date range with charts and save
python backtester/run_backtest.py NVDA m 2023-01-20 2023-07-20 --plot --save

# High capital test with full analysis
python backtester/run_backtest.py MSFT m 90 50000 --plot --save

# Quick comparison (momentum vs value)
python backtester/run_backtest.py AAPL m 30 && python backtester/run_backtest.py AAPL v 30

# Comprehensive testing across multiple scenarios
python backtester/batch_backtest.py

# View all usage examples and tips
python backtester/examples.py
```

#### Pro Tips
- **Use Strategy Defaults**: Always start with strategy defaults (momentum=5yr, value=10yr) unless testing specific hypotheses
- **Minimum Periods**: Momentum needs 3+ years, Value needs 5+ years for statistical validity
- **Historical Dates**: Use dates 15+ days old to avoid API rate limits
- **Validation**: Test multiple tickers to validate strategy robustness across sectors
- **Market Conditions**: Longer periods automatically capture different market regimes (bull/bear/sideways)
- **Capital Testing**: Try different starting amounts to test scalability
- **Batch Testing**: Use `--optimal-periods` flag for production validation, default mode for diversity testing

### Performance Optimization
- **Efficient Data Processing**: All sentiment analysis done once at startup with parallel API calls
- **Pre-calculated Scores**: B(t) and Final_Buy_Score calculated in advance using vectorized operations
- **Fast Simulation**: Lookup-based trading decisions for maximum speed (O(1) complexity)
- **Memory Management**: Optimized data structures (float32, categorical dtypes) and garbage collection
- **Parallel Backtesting**: ThreadPoolExecutor for concurrent test execution (configurable workers)
- **Vectorized Indicators**: NumPy/Pandas operations eliminate Python loops for 10-100x speedup
- **Cache Optimization**: LRU caches with thread-safe locks for sub-millisecond retrieval

## Disclaimer

**This tool is for educational and research purposes only. Do not rely solely on SigmaStocks for investment decisions. Always conduct your own research and consider consulting with financial professionals before making investment choices.**

## Requirements

- **Python 3.8 to 3.12** (⚠️ **Do not use Python 3.13+** as PyTorch does not yet have official CUDA/MPS acceleration binaries for newer versions).
- Internet connection for real-time data
- Alpaca API keys for news and historical data
- **Optional (but recommended):** GPU Hardware Acceleration (NVIDIA CUDA or Apple Silicon MPS) is fully supported natively for high-speed sentiment analysis and CNN inference.
- See `requirements.txt` for complete dependency list

## Dependencies

### Core Analysis & Performance
- `pandas>=2.0.0` - Data manipulation with vectorized operations
- `numpy>=1.24.0` - Numerical computing and array operations
- `yfinance>=0.2.0` - Financial data retrieval
- `concurrent.futures` (stdlib) - Parallel data ingestion and backtesting

### Hybrid Sentiment Analysis
- `vaderSentiment>=3.3.2` - Nuanced sentiment scoring with financial lexicon
- `transformers>=4.30.0` - DistilRoBERTa model for classification
- `torch>=2.0.0` - PyTorch for neural networks with GPU acceleration
- `tokenizers>=0.13.0` - Text tokenization for transformers

### News & Visualization
- `alpaca-py>=0.7.0` - Professional news data API
- `matplotlib>=3.7.0` - Chart visualization
- `tqdm>=4.65.0` - Progress bars for batch processing

### Backtesting & Optimization
- `pytz>=2023.3` - Timezone handling for historical data
- `functools.lru_cache` (stdlib) - Function-level memoization
- `threading.Lock` (stdlib) - Thread-safe cache synchronization

## Testing

The project includes comprehensive test coverage for the Convolutional Order Book feature:

```bash
# Install test dependencies
pip install pytest hypothesis

# Run all tests
pytest tests/

# Run specific test categories
pytest tests/ -m integration  # Integration tests only
pytest tests/test_cnn_output_range.py  # Property-based tests
pytest tests/test_analyze_liquidity.py  # Unit tests

# Run COB impact verification tests
python3 tests/test_cob_scenarios.py  # Scenario testing (recommended)
python3 tests/test_cob_impact.py     # Real-world testing

# Run with verbose output
pytest tests/ -v
```

**Test Coverage:**
- 27 test files covering CNN architecture, heatmap generation, and microstructure analysis
- Property-based tests using Hypothesis for invariant validation
- Integration tests for end-to-end flow verification
- Edge case and error handling tests
- **COB impact verification tests** demonstrating real-world effectiveness

**COB Impact Tests:**
- `test_cob_scenarios.py` - Simulates 5 market conditions to verify adjustments
- `test_cob_impact.py` - Tests with live market data
- See `COB_IMPACT_RESULTS.md` for detailed verification results

## License

This project is licensed under the GNU GPLv3 License - see the LICENSE file for details.

---

*Built with Python, Hybrid AI (VADER + DistilRoBERTa), Advanced Risk Management, Comprehensive Backtesting, and High-Performance Parallel Architecture*
