"""Backtester helpers that do not touch the network."""

from datetime import datetime, timezone

import pytz

from backtester.backtester import AlpacaBacktester


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


def test_filter_news_accepts_naive_cutoff():
    engine = AlpacaBacktester("MSFT")
    engine.full_news_data = [
        [pytz.utc.localize(datetime(2021, 5, 1, 12, 0)), "kept", "benzinga"]
    ]
    kept = engine.filter_news_by_date(datetime(2021, 5, 2))
    assert len(kept) == 1
