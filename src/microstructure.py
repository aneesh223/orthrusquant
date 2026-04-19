"""
Convolutional Order Book - Vision Engine Module

This module implements CNN-based market microstructure analysis using liquidity heatmaps.
It converts 1-minute OHLCV data into 2D representations and uses a Convolutional Neural
Network to detect patterns such as liquidity voids, accumulation zones, and microstructure
volatility.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Union

logger = logging.getLogger(__name__)

import os
MODEL_PATH = "models/orderbook_cnn.pt"
_model_trained = False


# Global model instance for lazy initialization
_model = None
_device = None


def _get_model():
    """
    Lazy initialization of CNN model with device detection.
    
    Returns:
        Tuple of (model, device) where model is OrderBookCNN instance
        and device is torch.device (cuda or cpu)
    """
    global _model, _device
    if _model is None:
        # Detect device: use CUDA if available, otherwise CPU or MPS
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            _device = torch.device("mps")
        else:
            _device = torch.device("cpu")
            
        logger.info(f"Initializing OrderBookCNN on device: {_device}")
        
        # Initialize model and move to device
        _model = OrderBookCNN().to(_device)
        
        # Set to evaluation mode (no training)
        _model.eval()
        
        global _model_trained
        if os.path.exists(MODEL_PATH):
            _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
            _model_trained = True
        else:
            logger.warning("model weights not found")
            _model_trained = False
        
    return _model, _device


class OrderBookCNN(nn.Module):
    """
    Convolutional Neural Network for analyzing liquidity heatmaps.
    
    Architecture:
        Input: (batch, 1, 64, 64) grayscale heatmap
        Conv2d(1→16) → BatchNorm → ReLU → MaxPool(2x2) → Dropout(0.3)
        Conv2d(16→32) → BatchNorm → ReLU → MaxPool(2x2) → Dropout(0.3)
        Flatten → Linear(32*16*16 → 1) → Sigmoid
        Output: (batch, 1) anomaly score in [0, 1]
    """
    
    def __init__(self, dropout_rate: float = 0.3):
        """
        Initialize CNN layers.
        
        Args:
            dropout_rate: Dropout probability for regularization (default: 0.3)
        """
        super(OrderBookCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Fully connected layer
        # After two 2x2 max pools: 64 -> 32 -> 16
        # So output is 32 channels * 16 * 16 = 8192
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 16 * 16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 64, 64)
        
        Returns:
            Output tensor of shape (batch_size, 1) with values in [0, 1]
        """
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)  # 64x64 -> 32x32
        x = self.dropout1(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)  # 32x32 -> 16x16
        x = self.dropout2(x)
        
        # Fully connected layer
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x


def df_to_heatmap(df: pd.DataFrame, window_size: int = 64) -> Optional[np.ndarray]:
    """
    Convert OHLCV DataFrame to 64x64 liquidity heatmap.
    
    Args:
        df: DataFrame with columns [Open, High, Low, Close, Volume]
        window_size: Size of the output heatmap (default: 64)
    
    Returns:
        numpy array of shape (64, 64) with values in [0, 1]
        Returns None if conversion fails
        
    Heatmap Structure:
        - Y-axis: Price levels (low to high)
        - X-axis: Time steps (chronological)
        - Pixel intensity: log(1 + volume), normalized to [0, 1]
    """
    try:
        # Validate input
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to df_to_heatmap")
            return None
        
        # Check required columns
        required_cols = ['High', 'Low', 'Volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"DataFrame missing required columns. Has: {df.columns.tolist()}")
            return None
        
        # Extract last window_size minutes of data
        df_window = df.tail(window_size).copy()
        
        # Initialize heatmap with zeros
        heatmap = np.zeros((window_size, window_size), dtype=np.float32)
        
        # Handle insufficient data by padding with zeros (already initialized)
        num_rows = len(df_window)
        if num_rows < window_size:
            logger.warning(f"Insufficient data: {num_rows} rows, padding to {window_size}")
        
        # Calculate price range from Low and High columns
        price_min = df_window['Low'].min()
        price_max = df_window['High'].max()
        price_range = price_max - price_min
        
        # Handle zero price range by returning uniform heatmap
        if price_range == 0 or np.isnan(price_range):
            logger.warning("Zero or NaN price range detected, returning uniform heatmap")
            # Return uniform heatmap with average intensity
            avg_volume = df_window['Volume'].mean() if 'Volume' in df_window.columns else 0
            # Handle NaN average volume
            if pd.isna(avg_volume):
                avg_volume = 0
            avg_intensity = np.log1p(avg_volume)
            heatmap.fill(avg_intensity)
            # Normalize to [0, 1]
            if avg_intensity > 0:
                heatmap = (heatmap / avg_intensity).astype(np.float32)
            return heatmap
        
        # Map each (time, price, volume) to heatmap coordinates
        for time_idx, (idx, row) in enumerate(df_window.iterrows()):
            # Calculate x coordinate (time axis)
            # If we have fewer than window_size rows, offset to the right
            x = time_idx + (window_size - num_rows)
            
            # Get price levels (use midpoint of High and Low)
            low_price = float(row['Low'])
            high_price = float(row['High'])
            volume = float(row['Volume'])
            
            # Handle NaN values
            # Use pd.isna() which works for both floats and integers
            if pd.isna(low_price) or pd.isna(high_price) or pd.isna(volume):
                continue
            
            # Apply log transformation to volume
            intensity = np.log1p(volume)  # log(1 + volume)
            
            # Map price range to Y-coordinates
            # Lower prices -> lower Y (row 0), higher prices -> higher Y (row 63)
            y_low = int((low_price - price_min) / price_range * (window_size - 1))
            y_high = int((high_price - price_min) / price_range * (window_size - 1))
            
            # Clamp to valid range
            y_low = max(0, min(window_size - 1, y_low))
            y_high = max(0, min(window_size - 1, y_high))
            
            # Fill the price range for this time step
            for y in range(y_low, y_high + 1):
                heatmap[y, x] = max(heatmap[y, x], intensity)
        
        # Normalize heatmap to [0, 1] range
        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()
        
        if heatmap_max > heatmap_min:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            # All values are the same, set to 0
            heatmap.fill(0.0)
        
        return heatmap
        
    except Exception as e:
        logger.error(f"Failed to generate heatmap: {e}", exc_info=True)
        return None


def analyze_liquidity(ticker: str) -> Dict[str, Union[float, str]]:
    """
    Analyze market microstructure using CNN on liquidity heatmaps.
    
    Args:
        ticker: Stock symbol
    
    Returns:
        Dictionary with keys:
            - 'anomaly_score': float in [0.0, 1.0]
            - 'status': str describing market condition
            - 'confidence': str indicating data quality
            
    Status Messages:
        - score > 0.85: "Critical liquidity void detected"
        - 0.70 < score <= 0.85: "High microstructure volatility"
        - 0.20 <= score <= 0.70: "Normal market conditions"
        - score < 0.20: "Strong accumulation zone detected"
        - Insufficient data: "Insufficient data for microstructure analysis"
    """
    try:
        if not _model_trained:
            return {'anomaly_score': 0.5, 'status': 'Model not trained', 'confidence': 'low'}
        # Import get_intraday_data from src.market
        from src.market import get_intraday_data
        
        # Fetch 1-minute data for ticker
        df = get_intraday_data(ticker)
        
        # Check if data has at least 10 minutes
        if df is None or df.empty or len(df) < 10:
            logger.warning(f"Insufficient data for {ticker}: {len(df) if df is not None and not df.empty else 0} minutes")
            return {
                'anomaly_score': 0.5,
                'status': 'Insufficient data for microstructure analysis',
                'confidence': 'low'
            }
        
        # Convert data to heatmap using df_to_heatmap
        heatmap = df_to_heatmap(df)
        
        # If heatmap is None, return neutral score (0.5) with error status
        if heatmap is None:
            logger.error(f"Failed to generate heatmap for {ticker}")
            return {
                'anomaly_score': 0.5,
                'status': 'Insufficient data for microstructure analysis',
                'confidence': 'low'
            }
        
        # Convert heatmap to PyTorch tensor: shape (1, 1, 64, 64)
        tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()
        
        # Get model and device using _get_model()
        model, device = _get_model()
        
        # Move tensor to device
        tensor = tensor.to(device)
        
        # Run inference with torch.no_grad()
        with torch.no_grad():
            output = model(tensor)
        
        # Extract scalar score from output tensor
        score = output.item()
        
        # Classify score into status message based on thresholds
        if score > 0.85:
            status = "Critical liquidity void detected"
            confidence = "high"
        elif score > 0.70:
            status = "High microstructure volatility"
            confidence = "high"
        elif score >= 0.20:
            status = "Normal market conditions"
            confidence = "high"
        else:  # score < 0.20
            status = "Strong accumulation zone detected"
            confidence = "high"
        
        # Log the analysis result
        logger.info(f"Microstructure analysis for {ticker}: score={score:.3f}, status={status}")
        
        # Return dictionary with anomaly_score, status, confidence
        return {
            'anomaly_score': score,
            'status': status,
            'confidence': confidence
        }
        
    except Exception as e:
        # Wrap entire function in try-except to catch all errors
        logger.error(f"Microstructure analysis failed for {ticker}: {e}", exc_info=True)
        return {
            'anomaly_score': 0.5,
            'status': 'Insufficient data for microstructure analysis',
            'confidence': 'low'
        }
