"""Regression tests for the two look-ahead leaks in the scoring path.

Perturbing only future prices, or only future headlines, must not change any
earlier day's score. Sentiment keys are bare ``datetime.date`` objects, which
is the shape ``get_sentiment`` produces for multi-day frames.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src import analyzer, market


def _business_prices(n=260, start="2021-01-04", seed=0):
    index = pd.bdate_range(start, periods=n, tz="UTC")
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.0, 1.0, n))
    close = np.maximum(close, 1.0)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(n, 1_000_000),
        },
        index=index,
    )


def _scores_by_date(verdict):
    out = {}
    for ts, score in verdict["Final_Buy_Scores_Over_Time"]:
        out[ts.date()] = float(score)
    return out


def _disable_transformer(monkeypatch):
    """Keep sentiment on the local VADER lexicon. Do not download weights."""
    analyzer.get_ai_analyzer.cache_clear()
    monkeypatch.setattr(analyzer, "_ensure_model", lambda: False)


def test_future_prices_do_not_change_earlier_scores():
    prices = _business_prices()
    # Bare dates, the type that used to raise against a UTC DatetimeIndex.
    sentiment_dates = [ts.date() for ts in prices.index[210:240]]
    sentiment = pd.DataFrame(
        {"Compound_Score": np.linspace(-0.4, 0.4, len(sentiment_dates))},
        index=sentiment_dates,
    )
    cutoff = sentiment_dates[15]

    base = market.calculate_verdict(
        "TEST",
        sentiment,
        strategy="momentum",
        lookback_days=10000,
        price_data=prices,
    )
    perturbed = prices.copy()
    future = np.array([ts.date() > cutoff for ts in perturbed.index])
    perturbed.loc[future, ["Open", "High", "Low", "Close"]] *= 3.0

    alt = market.calculate_verdict(
        "TEST",
        sentiment,
        strategy="momentum",
        lookback_days=10000,
        price_data=perturbed,
    )

    base_scores = _scores_by_date(base)
    alt_scores = _scores_by_date(alt)
    earlier = [day for day in base_scores if day <= cutoff]
    assert earlier, "expected scores on or before the cutoff"
    for day in earlier:
        assert alt_scores[day] == pytest.approx(base_scores[day], abs=1e-6)


def test_date_index_does_not_fall_back_to_the_full_series():
    prices = _business_prices(n=80, seed=1)
    day = prices.index[40].date()
    sentiment = pd.DataFrame({"Compound_Score": [0.2]}, index=[day])

    verdict = market.calculate_verdict(
        "TEST",
        sentiment,
        strategy="momentum",
        lookback_days=10000,
        price_data=prices,
    )
    scored_day = verdict["Final_Buy_Scores_Over_Time"][0][0]
    assert scored_day.tzinfo is not None
    # The slice used for this date must stop at this session. A 3x shock
    # after it changes the full-series MACD but must not change this score.
    shocked = prices.copy()
    shocked.loc[shocked.index.date > day, ["Open", "High", "Low", "Close"]] *= 3.0
    shocked_verdict = market.calculate_verdict(
        "TEST",
        sentiment,
        strategy="momentum",
        lookback_days=10000,
        price_data=shocked,
    )
    assert shocked_verdict["Final_Buy_Score"] == pytest.approx(
        verdict["Final_Buy_Score"], abs=1e-6
    )


def test_unaligned_price_index_raises_instead_of_using_every_bar():
    sentiment = pd.DataFrame({"Compound_Score": [0.1]}, index=[datetime(2021, 1, 4).date()])
    prices = pd.DataFrame({"Close": [10.0, 11.0, 12.0]})
    with pytest.raises(TypeError, match="DatetimeIndex"):
        market.calculate_verdict(
            "TEST",
            sentiment,
            strategy="value",
            lookback_days=10000,
            price_data=prices,
        )


def test_prices_through_reraises_on_failed_comparison():
    prices = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    with pytest.raises(TypeError, match="full series"):
        market._prices_through(prices, pd.Timestamp("2021-01-04", tz="UTC"))


def test_future_news_does_not_change_earlier_scores(monkeypatch):
    _disable_transformer(monkeypatch)
    prices = _business_prices(n=40, seed=2)
    days = list(prices.index[-12:])

    def headlines(future_text):
        rows = []
        # Newest-first, matching scraper.scrape_hybrid.
        for i, ts in enumerate(reversed(days)):
            if i < 4:
                text = future_text
            else:
                text = "Company reports strong profit and a bullish upgrade"
            rows.append([ts.to_pydatetime(), text, "Reuters"])
        return rows

    calm = analyzer.get_sentiment(headlines("Company reports strong profit and a bullish upgrade"), "TEST", 30)
    shocked = analyzer.get_sentiment(headlines("Shares crash plunge and downgrade after a loss"), "TEST", 30)

    assert list(calm.index) == sorted(calm.index)
    assert list(shocked.index) == sorted(shocked.index)

    calm_verdict = market.calculate_verdict(
        "TEST", calm, strategy="momentum", lookback_days=10000, price_data=prices
    )
    shocked_verdict = market.calculate_verdict(
        "TEST", shocked, strategy="momentum", lookback_days=10000, price_data=prices
    )
    calm_scores = _scores_by_date(calm_verdict)
    shocked_scores = _scores_by_date(shocked_verdict)

    # The four newest sessions are the ones whose headlines changed.
    protected = sorted(calm_scores)[:-4]
    assert len(protected) >= 6
    for day in protected:
        assert shocked_scores[day] == pytest.approx(calm_scores[day], abs=1e-6)
    # The shock has to be visible somewhere, or the test would pass by doing nothing.
    assert shocked_scores[sorted(shocked_scores)[-1]] != pytest.approx(
        calm_scores[sorted(calm_scores)[-1]], abs=1e-3
    )


def test_get_sentiment_index_is_chronological_even_if_input_is_newest_first(monkeypatch):
    _disable_transformer(monkeypatch)
    start = datetime(2021, 6, 1, tzinfo=timezone.utc)
    rows = []
    for offset in range(6, -1, -1):
        rows.append([
            start + timedelta(days=offset),
            "Shares rally after a strong profit beat",
            "Bloomberg",
        ])
    frame = analyzer.get_sentiment(rows, "AAPL", timeframe_days=30)
    assert list(frame.index) == sorted(frame.index)
    assert frame["Compound_Score"].notna().all()
