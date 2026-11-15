import os

# Dummy credentials so imports and mocked clients never need real Alpaca keys.
os.environ.setdefault("ALPACA_API_KEY", "dummy")
os.environ.setdefault("ALPACA_SECRET_KEY", "dummy")

import pytest
from unittest.mock import patch

@pytest.fixture(autouse=True)
def mock_model_trained_global():
    with patch('src.microstructure._model_trained', True):
        yield
