"""Backtester helpers that do not touch the network."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytz

from backtester.backtester import (
    AlpacaBacktester,
    news_window_days,
    sentiment_coverage,
)
from src import analyzer, scraper


def test_backtester_constructs_without_opening_a_client():
    engine = AlpacaBacktester("AAPL", strategy="momentum", initial_cash=1000)
    assert engine._alpaca_client is None
    assert engine.ticker == "AAPL"


def test_filter_news_drops_articles_after_the_cutoff():
    engine = AlpacaBacktester("AAPL")
    early = datetime(2021, 1, 4, tzinfo=timezone.utc)
    late = datetime(2021, 2, 1, tzinfo=timezone.utc)
    engine.full_news_data = [
        [late, "future headline", "benzinga"],
        [early, "past headline", "benzinga"],
        ["not-a-timestamp", "dropped", "benzinga"],
    ]
    cutoff = datetime(2021, 1, 15, tzinfo=timezone.utc)
    kept = engine.filter_news_by_date(cutoff)
    assert [item[1] for item in kept] == ["past headline"]


def test_news_window_is_not_capped_at_90_days():
    assert news_window_days(400, None) == 410
    assert news_window_days(5, None) == 15
    # The cap is opt-in. Callers who want the old limit pass it explicitly.
    assert news_window_days(400, 90) == 90


def test_sentiment_coverage_counts_only_days_with_history():
    days = pd.bdate_range("2021-01-04", periods=10, tz="UTC")
    # Observations start on the 6th session, so the first five fall back.
    observations = [days[5].date(), days[8].date()]
    stats = sentiment_coverage(days, observations)
    assert stats["sentiment_observation_days"] == 2
    assert stats["neutral_fallback_days"] == 5
    assert stats["sentiment_covered_days"] == 5
    assert stats["sentiment_coverage_pct"] == 50.0


def test_fetch_news_uses_historical_alpaca_source(monkeypatch):
    calls = {}

    def _historical(ticker, days=30, end_date=None):
        calls["ticker"] = ticker
        calls["days"] = days
        calls["end"] = end_date
        return (ticker, "N/A", "N/A"), []

    monkeypatch.setattr(scraper, "scrape_historical_news", _historical)
    monkeypatch.setattr(
        scraper,
        "scrape_hybrid",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("hybrid")),
    )
    monkeypatch.setattr(
        scraper,
        "scrape_gnews",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("gnews")),
    )
    engine = AlpacaBacktester("AAPL", news_lookback_days=None)
    end = datetime(2021, 6, 1, tzinfo=timezone.utc)
    engine.fetch_news_data(days=120, end_date=end)
    assert calls == {"ticker": "AAPL", "days": 120, "end": end}


def test_offline_run_reports_partial_sentiment_coverage(monkeypatch):
    analyzer.get_ai_analyzer.cache_clear()
    monkeypatch.setattr(analyzer, "_ensure_model", lambda: False)

    index = pd.bdate_range("2021-01-04", periods=40, tz="UTC")
    rng = np.random.default_rng(3)
    close = 50 + np.cumsum(rng.normal(0, 0.4, len(index)))
    close = np.maximum(close, 1.0)
    prices = pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": 1000,
        },
        index=index,
    )
    # Articles only on the last five sessions. Earlier sessions have no
    # observation on or before them, so they count as neutral fallback.
    news = []
    for ts in index[-5:]:
        news.append([
            ts.to_pydatetime(),
            "Company reports strong profit and a bullish upgrade",
            "benzinga",
        ])
    engine = AlpacaBacktester("TEST", strategy="value", initial_cash=10000)
    engine.full_price_data = prices
    engine.full_news_data = news
    start = index[20].to_pydatetime()
    end = index[-1].to_pydatetime()
    results = engine.run(start, end, lookback_days=30)
    assert results["news_source"] == "alpaca"
    assert results["news_days_requested"] == (end - start).days + 10
    assert 0 < results["sentiment_coverage_pct"] < 100
    assert results["neutral_fallback_days"] > 0
    assert results["sentiment_observation_days"] > 0


def test_filter_news_accepts_naive_cutoff():
    engine = AlpacaBacktester("MSFT")
    engine.full_news_data = [
        [pytz.utc.localize(datetime(2021, 5, 1, 12, 0)), "kept", "benzinga"]
    ]
    kept = engine.filter_news_by_date(datetime(2021, 5, 2))
    assert len(kept) == 1
