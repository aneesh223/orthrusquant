"""
Unit tests for device detection and model initialization.

Feature: convolutional-order-book
Task: 3.2 Add device detection and model initialization
"""

import pytest
import torch
from unittest.mock import patch

# Import the function and class to test
from src.microstructure import _get_model, OrderBookCNN


def test_get_model_returns_model_and_device():
    """
    Test that _get_model returns a model instance and device.
    **Validates: Requirements 7.3, 7.4, 7.5**
    """
    # Reset the global state
    import src.microstructure
    src.microstructure._model = None
    src.microstructure._device = None
    
    # Call the function
    model, device = _get_model()
    
    # Verify model is an instance of OrderBookCNN
    assert isinstance(model, OrderBookCNN), \
        "Model should be an instance of OrderBookCNN"
    
    # Verify device is a torch.device
    assert isinstance(device, torch.device), \
        "Device should be a torch.device instance"
    
    # Verify device is either cuda, mps, or cpu
    assert device.type in ['cuda', 'mps', 'cpu'], \
        "Device type should be 'cuda', 'mps', or 'cpu'"
    
    # Verify model is in eval mode
    assert not model.training, \
        "Model should be in evaluation mode (not training)"


def test_get_model_uses_cuda_when_available():
    """
    Test that _get_model uses CUDA when available.
    **Validates: Requirements 7.4**
    """
    # Reset the global state
    import src.microstructure
    src.microstructure._model = None
    src.microstructure._device = None
    
    # Mock torch.cuda.is_available to return True
    # Also mock the model's .to() method to avoid actual CUDA operations
    with patch('torch.cuda.is_available', return_value=True):
        with patch.object(OrderBookCNN, 'to', return_value=OrderBookCNN()) as mock_to:
            model, device = _get_model()
            
            # Verify device is cuda
            assert device.type == 'cuda', \
                "Device should be 'cuda' when CUDA is available"
            
            # Verify .to() was called with a cuda device
            mock_to.assert_called_once()
            call_args = mock_to.call_args[0][0]
            assert call_args.type == 'cuda', \
                "Model should be moved to CUDA device"


def test_get_model_uses_cpu_when_cuda_unavailable():
    """
    Test that _get_model uses CPU when CUDA is unavailable.
    **Validates: Requirements 7.5**
    """
    # Reset the global state
    import src.microstructure
    src.microstructure._model = None
    src.microstructure._device = None
    
    # Mock torch.cuda.is_available to return False
    with patch('torch.cuda.is_available', return_value=False):
        model, device = _get_model()
        
        # Verify device is cpu
        assert device.type == 'cpu', \
            "Device should be 'cpu' when CUDA is unavailable"


def test_get_model_returns_singleton():
    """
    Test that _get_model returns the same model instance on subsequent calls.
    **Validates: Requirements 7.3**
    """
    # Reset the global state
    import src.microstructure
    src.microstructure._model = None
    src.microstructure._device = None
    
    # Call the function twice
    model1, device1 = _get_model()
    model2, device2 = _get_model()
    
    # Verify both calls return the same model instance
    assert model1 is model2, \
        "Subsequent calls to _get_model should return the same model instance"
    
    # Verify both calls return the same device instance
    assert device1 is device2, \
        "Subsequent calls to _get_model should return the same device instance"


def test_model_is_moved_to_device():
    """
    Test that the model is moved to the detected device.
    **Validates: Requirements 7.4, 7.5**
    """
    # Reset the global state
    import src.microstructure
    src.microstructure._model = None
    src.microstructure._device = None
    
    # Call the function
    model, device = _get_model()
    
    # Get the device of the first parameter
    param_device = next(model.parameters()).device
    
    # Verify the model's parameters are on the correct device
    assert param_device.type == device.type, \
        f"Model parameters should be on {device.type}, but are on {param_device.type}"


def test_model_is_in_eval_mode():
    """
    Test that the model is set to evaluation mode.
    **Validates: Requirements 7.3**
    """
    # Reset the global state
    import src.microstructure
    src.microstructure._model = None
    src.microstructure._device = None
    
    # Call the function
    model, device = _get_model()
    
    # Verify model is in eval mode (training=False)
    assert not model.training, \
        "Model should be in evaluation mode (training=False)"
    
    # Verify dropout layers are in eval mode
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            assert not module.training, \
                "Dropout layers should be in evaluation mode"
