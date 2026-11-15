"""Unit tests for the scraper. Network clients are mocked."""

import inspect
import ssl
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src import config, scraper


def test_scraper_does_not_disable_tls_verification():
    assert ssl._create_default_https_context is not ssl._create_unverified_context
    source = inspect.getsource(scraper)
    assert "_create_unverified_context" not in source
    assert "ssl._create_default_https_context" not in source


def test_alpaca_news_sends_start_and_end(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def get(self, endpoint, parameters):
            captured["endpoint"] = endpoint
            captured["parameters"] = parameters
            return {"news": [], "next_page_token": None}

    monkeypatch.setattr(config, "API_KEY", "dummy-key")
    monkeypatch.setattr(config, "API_SECRET", "dummy-secret")
    monkeypatch.setattr(scraper, "get_profile_data", lambda ticker: (ticker, "N/A", "N/A"))
    monkeypatch.setattr("alpaca.common.rest.RESTClient", FakeClient)

    end = datetime(2022, 6, 15, tzinfo=timezone.utc)
    profile, articles = scraper.scrape_alpaca_news("AAPL", days=10, end_date=end)

    assert profile[0] == "AAPL"
    assert articles == []
    assert captured["endpoint"] == "/news"
    assert captured["parameters"]["symbols"] == "AAPL"
    assert captured["parameters"]["end"].startswith("2022-06-15")
    assert captured["parameters"]["start"].startswith("2022-06-05")
    assert captured["init"]["api_key"] == "dummy-key"


def test_alpaca_news_skips_bad_timestamps(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def get(self, endpoint, parameters):
            return {
                "news": [
                    {"created_at": "not-a-date", "headline": "Ignored", "source": "benzinga"},
                    {
                        "created_at": "2022-06-01T15:00:00Z",
                        "headline": "Kept",
                        "source": "benzinga",
                    },
                ],
                "next_page_token": None,
            }

    monkeypatch.setattr(config, "API_KEY", "dummy-key")
    monkeypatch.setattr(config, "API_SECRET", "dummy-secret")
    monkeypatch.setattr(scraper, "get_profile_data", lambda ticker: (ticker, "N/A", "N/A"))
    monkeypatch.setattr("alpaca.common.rest.RESTClient", FakeClient)

    _, articles = scraper.scrape_alpaca_news(
        "AAPL", days=30, end_date=datetime(2022, 6, 15, tzinfo=timezone.utc)
    )
    assert len(articles) == 1
    assert articles[0][1] == "Kept"


def test_config_import_does_not_require_keys():
    """Importing config with an empty environment must not raise."""
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env.pop("ALPACA_API_KEY", None)
    env.pop("ALPACA_SECRET_KEY", None)
    env["PYTHONPATH"] = os.getcwd()
    # Ignore a developer .env if one is present in this checkout.
    env["DOTENV_PATH"] = os.devnull
    completed = subprocess.run(
        [sys.executable, "-c", "import src.config; from src.config import require_api_keys"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_require_api_keys_raises_when_missing(monkeypatch):
    monkeypatch.setattr(config, "API_KEY", None)
    monkeypatch.setattr(config, "API_SECRET", "")
    with pytest.raises(ValueError, match="ALPACA_API_KEY"):
        config.require_api_keys()
