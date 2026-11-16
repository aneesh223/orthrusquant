"""Trailing stop is per position, and the price warm-up covers 200 trading bars."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from backtester.backtester import AlpacaBacktester


def _flat_prices(n=300, start="2023-01-02", price=100.0):
    index = pd.bdate_range(start, periods=n, tz="UTC") + pd.Timedelta(hours=5)
    close = np.full(n, price)
    return pd.DataFrame(
        {"Open": close, "High": close + 0.5, "Low": close - 0.5, "Close": close, "Volume": 1_000_000},
        index=index,
    )


def _engine_after_drawdown():
    """A book that once peaked at $12,000 and now holds $10,000 in cash."""
    engine = AlpacaBacktester("TEST", strategy="momentum", initial_cash=10_000,
                              commission_bps=0.0, slippage_bps=0.0)
    engine.full_price_data = _flat_prices()
    engine.max_portfolio_value = 12_000.0
    return engine


def test_new_entry_is_not_stopped_out_by_an_old_account_peak():
    engine = _engine_after_drawdown()
    day = engine.full_price_data.index[-1].to_pydatetime()
    engine.execute_adaptive_trade("BUY", 100.0, day, 6.0, "SIDEWAYS")
    assert engine.shares > 0

    # Price has not moved since the fill, so no exit rule should fire.
    value = engine.cash + engine.shares * 100.0
    should_sell, reason = engine.check_adaptive_risk_management(100.0, value, "SIDEWAYS", day)
    assert not should_sell, reason


def test_trailing_stop_still_fires_on_a_drop_from_the_position_peak():
    engine = _engine_after_drawdown()
    day = engine.full_price_data.index[-1].to_pydatetime()
    engine.execute_adaptive_trade("BUY", 100.0, day, 6.0, "SIDEWAYS")
    shares, cash = engine.shares, engine.cash

    # Run the position up 20%, then let it fall back 15% from that high.
    up = cash + shares * 120.0
    engine.check_adaptive_risk_management(120.0, up, "SIDEWAYS", day)
    down_price = 102.0
    down = cash + shares * down_price
    risk = engine.get_adaptive_risk_params("SIDEWAYS", day)
    expected = (down - up) / up <= risk["trailing_stop_pct"]
    should_sell, _ = engine.check_adaptive_risk_management(down_price, down, "SIDEWAYS", day)
    assert expected, "fixture should be a real trailing-stop breach"
    assert should_sell
    assert engine.max_portfolio_value == up


def test_price_fetch_covers_200_trading_bars_before_start(monkeypatch):
    engine = AlpacaBacktester("TEST", strategy="momentum", initial_cash=10_000)
    calls = {}

    def _fake_fetch(start, end, symbol=None):
        calls["start"], calls["end"] = start, end
        return _flat_prices(n=600, start="2022-01-03")

    monkeypatch.setattr(engine, "fetch_historical_data", _fake_fetch)
    engine.full_news_data = []
    start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    engine.prepare_simulation_data(start, start + timedelta(days=30), news_days=40)

    bars = pd.bdate_range(calls["start"].date(), start.date() - timedelta(days=1))
    # bdate_range ignores exchange holidays (~9 a year), so leave headroom.
    assert len(bars) >= 209
