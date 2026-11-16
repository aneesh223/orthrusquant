"""Sentiment is fresh or neutral. Technicals are recomputed every session.

A headline only covers the calendar days [published, published + N). Days
outside that window must not reuse the old score, and coverage counts only
days that actually have a row inside the window.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from backtester.backtester import AlpacaBacktester
from src import analyzer


@pytest.fixture(autouse=True)
def _vader_only(monkeypatch):
    analyzer.get_ai_analyzer.cache_clear()
    monkeypatch.setattr(analyzer, "_ensure_model", lambda: False)
    yield
    analyzer.get_ai_analyzer.cache_clear()


def _prices(n=320, start="2020-01-02", seed=11):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=n, tz="UTC")
    rets = rng.normal(0.0004, 0.012, n)
    close = 50.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=index,
    )


def _run(prices, news, start, end, freshness=5, strategy="momentum"):
    engine = AlpacaBacktester(
        "TEST",
        strategy=strategy,
        initial_cash=10_000.0,
        sentiment_freshness_days=freshness,
    )
    engine.full_price_data = prices
    engine.full_news_data = news
    return engine.run(start, end, lookback_days=30)


def _scores(results):
    return [row["Final_Buy_Score"] for row in results["portfolio_history"]]


def test_stale_headlines_do_not_freeze_technicals_or_count_as_coverage():
    prices = _prices()
    start, end = prices.index[60], prices.index[-1]
    # Both headlines are well before the simulated window.
    old = (start - pd.Timedelta(days=40)).to_pydatetime()
    news = [
        (old, "Company beats earnings estimates and raises full-year guidance", "Reuters"),
        (old + timedelta(days=1), "Shares rally on record revenue and bullish outlook", "Reuters"),
    ]
    results = _run(prices, news, start.to_pydatetime(), end.to_pydatetime())
    assert results["sentiment_coverage_pct"] == 0.0
    assert results["sentiment_freshness_days"] == 5
    assert max(_scores(results)) - min(_scores(results)) > 1e-6


def test_headline_covers_only_its_freshness_window():
    prices = _prices(n=80, start="2021-01-04", seed=4)
    # Wednesday 2021-01-06. N=5 covers through Sunday 2021-01-10.
    # The next session, Monday 2021-01-11, is day k+N and must be neutral.
    k = datetime(2021, 1, 6, tzinfo=timezone.utc)
    news = [(k, "Company beats earnings estimates and raises full-year guidance", "Reuters")]
    start, end = prices.index[0], prices.index[-1]
    results = _run(prices, news, start.to_pydatetime(), end.to_pydatetime(), freshness=5)
    by_day = {row["Date"].date(): row for row in results["portfolio_history"]}
    for offset in range(5):
        day = (k + timedelta(days=offset)).date()
        if day not in by_day:
            continue
        assert by_day[day]["Sentiment_Covered"] is True
    neutral_day = (k + timedelta(days=5)).date()
    assert neutral_day in by_day
    assert by_day[neutral_day]["Sentiment_Covered"] is False
    assert by_day[neutral_day]["Sentiment_Compound"] == pytest.approx(0.0)
    assert by_day[k.date()]["Sentiment_Compound"] != pytest.approx(0.0)


def test_freshness_window_changes_coverage():
    prices = _prices(n=40, start="2021-01-04", seed=8)
    k = prices.index[10]
    news = [(k.to_pydatetime(), "Shares rally on record revenue and bullish outlook", "Reuters")]
    start, end = prices.index[0].to_pydatetime(), prices.index[-1].to_pydatetime()
    narrow = _run(prices, news, start, end, freshness=1)
    wide = _run(prices, news, start, end, freshness=5)
    assert narrow["sentiment_freshness_days"] == 1
    assert wide["sentiment_freshness_days"] == 5
    assert wide["sentiment_coverage_pct"] > narrow["sentiment_coverage_pct"]
    assert narrow["sentiment_covered_days"] == 1
