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


@pytest.fixture(autouse=True)
def _block_http_clients(monkeypatch):
    """Stop HTTP libraries that do not go through socket.socket.

    pytest-socket blocks socket.socket. yfinance uses curl_cffi, which talks
    to the network from native code and would otherwise skip that block.
    """
    def _blocked(*args, **kwargs):
        raise RuntimeError("HTTP is disabled in tests")

    try:
        import curl_cffi.requests as curl_requests
        monkeypatch.setattr(curl_requests.Session, "request", _blocked)
    except ImportError:
        pass
    try:
        import requests
        monkeypatch.setattr(requests.sessions.Session, "request", _blocked)
    except ImportError:
        pass
