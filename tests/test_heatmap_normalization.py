"""
Property-based tests for heatmap normalization invariant.

Feature: convolutional-order-book
Task: 5.3 Write property test for heatmap normalization
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, assume, settings, HealthCheck

from src.microstructure import df_to_heatmap


# Custom strategy for generating OHLCV DataFrames
@st.composite
def ohlcv_dataframe(draw, min_rows=1, max_rows=500):
    """
    Generate random OHLCV DataFrame with varying lengths.
    
    Args:
        draw: Hypothesis draw function
        min_rows: Minimum number of rows
        max_rows: Maximum number of rows
    
    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume]
    """
    # Generate random number of rows
    num_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    
    # Generate base price (avoid extreme values that might cause numerical issues)
    base_price = draw(st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False))
    
    # Generate price volatility (how much prices vary)
    volatility = draw(st.floats(min_value=0.0, max_value=base_price * 0.5, allow_nan=False, allow_infinity=False))
    
    # Generate base volume
    base_volume = draw(st.integers(min_value=0, max_value=10000000))
    
    # Create timestamps
    start_time = datetime.now() - timedelta(minutes=num_rows)
    dates = pd.date_range(start=start_time, periods=num_rows, freq='1min')
    
    # Generate OHLCV data
    data = {
        'Open': [],
        'High': [],
        'Low': [],
        'Close': [],
        'Volume': []
    }
    
    for i in range(num_rows):
        # Generate random price movement
        price_change = draw(st.floats(min_value=-volatility, max_value=volatility, allow_nan=False, allow_infinity=False))
        open_price = base_price + price_change
        
        # Generate high/low around open price
        high_offset = draw(st.floats(min_value=0.0, max_value=volatility * 0.5, allow_nan=False, allow_infinity=False))
        low_offset = draw(st.floats(min_value=0.0, max_value=volatility * 0.5, allow_nan=False, allow_infinity=False))
        
        high_price = open_price + high_offset
        low_price = max(0.01, open_price - low_offset)  # Ensure positive price
        
        # Close price between low and high
        close_price = draw(st.floats(min_value=low_price, max_value=high_price, allow_nan=False, allow_infinity=False))
        
        # Generate volume
        volume_change = draw(st.integers(min_value=-base_volume // 2, max_value=base_volume))
        volume = max(0, base_volume + volume_change)
        
        data['Open'].append(open_price)
        data['High'].append(high_price)
        data['Low'].append(low_price)
        data['Close'].append(close_price)
        data['Volume'].append(volume)
    
    return pd.DataFrame(data, index=dates)


@given(df=ohlcv_dataframe(min_rows=1, max_rows=500))
def test_heatmap_normalization_range(df):
    """
    Feature: convolutional-order-book, Property 5: Normalization Invariant
    **Validates: Requirements 2.5, 2.7**
    
    Test that for any generated heatmap, all pixel intensity values are in the
    range [0.0, 1.0]. This ensures the CNN receives properly normalized inputs
    regardless of the volume distribution in the input data.
    
    This property must hold for:
    - Any DataFrame length from 1 to 500 rows
    - Any price range (including zero price range)
    - Any volume values (including zero volume)
    - Various volume distributions
    """
    heatmap = df_to_heatmap(df)
    
    if heatmap is not None:
        # Verify all values are in [0.0, 1.0]
        assert np.all(heatmap >= 0.0), \
            f"Heatmap contains values below 0.0: min={heatmap.min()}"
        
        assert np.all(heatmap <= 1.0), \
            f"Heatmap contains values above 1.0: max={heatmap.max()}"


@given(df=ohlcv_dataframe(min_rows=64, max_rows=500))
@settings(
    suppress_health_check=[HealthCheck.data_too_large, HealthCheck.too_slow],
    deadline=None,
)
def test_heatmap_normalization_bounds(df):
    """
    Feature: convolutional-order-book, Property 5: Normalization Invariant
    **Validates: Requirements 2.5, 2.7**
    
    Test that for any generated heatmap with non-uniform data, the minimum value
    is 0.0 and the maximum value is 1.0. This ensures proper normalization where
    the darkest pixel is 0 and the brightest pixel is 1.
    
    For uniform data (all prices identical), the heatmap should have consistent
    average intensity.
    """
    heatmap = df_to_heatmap(df)
    
    if heatmap is not None:
        # Check if data is non-uniform (has varying prices or volumes)
        price_range = df['High'].max() - df['Low'].min()
        volume_range = df['Volume'].max() - df['Volume'].min()
        
        # For non-uniform data, verify min is 0.0 and max is 1.0
        if price_range > 0 and volume_range > 0:
            # Allow small floating point tolerance
            tolerance = 1e-6
            
            # Check if heatmap has any variation
            heatmap_range = heatmap.max() - heatmap.min()
            
            if heatmap_range > tolerance:
                # Non-uniform heatmap should have min=0 and max=1
                assert heatmap.min() < tolerance, \
                    f"Non-uniform heatmap should have min=0.0, but got {heatmap.min()}"
                
                assert abs(heatmap.max() - 1.0) < tolerance, \
                    f"Non-uniform heatmap should have max=1.0, but got {heatmap.max()}"


@given(
    num_rows=st.integers(min_value=10, max_value=100),
    base_price=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    volume_multiplier=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
)
def test_heatmap_normalization_with_varying_volumes(num_rows, base_price, volume_multiplier):
    """
    Feature: convolutional-order-book, Property 5: Normalization Invariant
    **Validates: Requirements 2.5**
    
    Test that normalization works correctly with widely varying volume values.
    Even with extreme volume differences, all pixel values should be in [0, 1].
    """
    # Create DataFrame with exponentially varying volumes
    dates = pd.date_range(start=datetime.now(), periods=num_rows, freq='1min')
    
    data = {
        'Open': [base_price + i * 0.1 for i in range(num_rows)],
        'High': [base_price + i * 0.1 + 1.0 for i in range(num_rows)],
        'Low': [base_price + i * 0.1 - 1.0 for i in range(num_rows)],
        'Close': [base_price + i * 0.1 + 0.5 for i in range(num_rows)],
        'Volume': [int(1000 * (volume_multiplier ** i)) for i in range(num_rows)]
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    if heatmap is not None:
        # Verify all values are in [0.0, 1.0]
        assert np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0), \
            f"Heatmap values out of range [0, 1]: min={heatmap.min()}, max={heatmap.max()}"


@given(
    num_rows=st.integers(min_value=10, max_value=100),
    price=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    volume=st.integers(min_value=0, max_value=10000000)
)
def test_heatmap_normalization_uniform_data(num_rows, price, volume):
    """
    Feature: convolutional-order-book, Property 5: Normalization Invariant
    **Validates: Requirements 2.7**
    
    Test that for uniform data (all prices identical), the heatmap has consistent
    average intensity. The function should return a uniform heatmap rather than
    failing or producing invalid values.
    """
    # Create DataFrame with zero price range (all prices identical)
    dates = pd.date_range(start=datetime.now(), periods=num_rows, freq='1min')
    
    data = {
        'Open': [price] * num_rows,
        'High': [price] * num_rows,
        'Low': [price] * num_rows,
        'Close': [price] * num_rows,
        'Volume': [volume] * num_rows
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, \
        "Heatmap should not be None for uniform data"
    
    # Verify all values are in [0.0, 1.0]
    assert np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0), \
        f"Uniform heatmap values out of range [0, 1]: min={heatmap.min()}, max={heatmap.max()}"
    
    # For uniform data, all values should be the same (or very close)
    # Allow some tolerance for floating point arithmetic
    tolerance = 1e-5
    unique_values = np.unique(heatmap)
    
    # Should have very few unique values (ideally 1, but allow for floating point variations)
    assert len(unique_values) <= 3, \
        f"Uniform data should produce uniform heatmap, but got {len(unique_values)} unique values"


@given(
    num_rows=st.integers(min_value=64, max_value=200),
    base_price=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    price_volatility=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False)
)
def test_heatmap_normalization_with_outliers(num_rows, base_price, price_volatility):
    """
    Feature: convolutional-order-book, Property 5: Normalization Invariant
    **Validates: Requirements 2.5**
    
    Test that normalization handles outliers correctly. Even with extreme volume
    spikes, the heatmap should be properly normalized to [0, 1].
    """
    # Create DataFrame with one extreme volume outlier
    dates = pd.date_range(start=datetime.now(), periods=num_rows, freq='1min')
    
    volumes = [1000000] * num_rows
    # Add an extreme outlier in the middle
    outlier_idx = num_rows // 2
    volumes[outlier_idx] = 100000000  # 100x larger
    
    data = {
        'Open': [base_price + np.random.uniform(-price_volatility, price_volatility) for _ in range(num_rows)],
        'High': [base_price + np.random.uniform(0, price_volatility) for _ in range(num_rows)],
        'Low': [base_price - np.random.uniform(0, price_volatility) for _ in range(num_rows)],
        'Close': [base_price + np.random.uniform(-price_volatility, price_volatility) for _ in range(num_rows)],
        'Volume': volumes
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    if heatmap is not None:
        # Verify all values are in [0.0, 1.0]
        assert np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0), \
            f"Heatmap with outliers out of range [0, 1]: min={heatmap.min()}, max={heatmap.max()}"
        
        # Verify the outlier creates a bright spot (value close to 1.0)
        # The maximum value should be 1.0 (or very close)
        tolerance = 1e-5
        assert heatmap.max() > 1.0 - tolerance, \
            f"Outlier should create max value ~1.0, but got {heatmap.max()}"


def test_heatmap_normalization_zero_volume():
    """
    Unit test: Verify normalization with all zero volumes.
    **Validates: Requirements 2.5, 2.7**
    """
    num_rows = 64
    dates = pd.date_range(start=datetime.now(), periods=num_rows, freq='1min')
    
    data = {
        'Open': [100.0 + i * 0.1 for i in range(num_rows)],
        'High': [101.0 + i * 0.1 for i in range(num_rows)],
        'Low': [99.0 + i * 0.1 for i in range(num_rows)],
        'Close': [100.5 + i * 0.1 for i in range(num_rows)],
        'Volume': [0] * num_rows  # All zero volumes
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, \
        "Heatmap should not be None with zero volumes"
    
    # Verify all values are in [0.0, 1.0]
    assert np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0), \
        f"Heatmap with zero volumes out of range [0, 1]: min={heatmap.min()}, max={heatmap.max()}"
    
    # With zero volumes, heatmap should be all zeros (after normalization)
    assert np.all(heatmap == 0.0), \
        f"Heatmap with zero volumes should be all zeros, but got values: {np.unique(heatmap)}"
