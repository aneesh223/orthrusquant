"""Commission and slippage on fills."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from backtester.backtester import AlpacaBacktester
from backtester.metrics import affordable_shares, buy_fill, sell_fill


def test_five_bps_each_side_on_a_round_lot():
    fill, notional, commission, outlay = buy_fill(100.0, 10, commission_bps=5, slippage_bps=5)
    assert fill == pytest.approx(100.05)
    assert notional == pytest.approx(1000.5)
    assert commission == pytest.approx(1000.5 * 0.0005)
    assert outlay == pytest.approx(notional + commission)

    sell_px, sell_notional, sell_commission, proceeds = sell_fill(
        100.0, 10, commission_bps=5, slippage_bps=5
    )
    assert sell_px == pytest.approx(99.95)
    assert sell_commission == pytest.approx(sell_notional * 0.0005)
    assert proceeds == pytest.approx(sell_notional - sell_commission)
    assert proceeds < 1000


def test_affordable_shares_leave_room_for_costs():
    shares = affordable_shares(1000, 100, commission_bps=5, slippage_bps=5)
    _, _, _, outlay = buy_fill(100, shares, 5, 5)
    assert shares == 9
    assert outlay <= 1000
    _, _, _, too_many = buy_fill(100, shares + 1, 5, 5)
    assert too_many > 1000


def test_zero_costs_match_the_quoted_price():
    engine = AlpacaBacktester(
        "TEST", initial_cash=5000, commission_bps=0, slippage_bps=0
    )
    when = datetime(2021, 1, 4, tzinfo=timezone.utc)
    engine.execute_trade("BUY", 50.0, when, 8.0)
    trade = engine.trade_log[0]
    assert trade["Shares"] == 100
    assert trade["Fill_Price"] == pytest.approx(50.0)
    assert trade["Commission"] == pytest.approx(0.0)
    assert engine.cash == pytest.approx(0.0)

    engine.execute_trade("SELL", 55.0, when, 2.0)
    sell = engine.trade_log[1]
    assert sell["Fill_Price"] == pytest.approx(55.0)
    assert engine.cash == pytest.approx(5500.0)


def test_adaptive_buy_pays_more_than_the_quoted_notional():
    engine = AlpacaBacktester(
        "TEST", strategy="value", initial_cash=10000, commission_bps=5, slippage_bps=5
    )
    index = pd.bdate_range("2021-01-04", periods=40, tz="UTC")
    engine.full_price_data = pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1000},
        index=index,
    )
    when = index[-1].to_pydatetime()
    engine.execute_adaptive_trade("BUY", 100.0, when, 5.0, "SIDEWAYS")
    assert engine.trade_log, "expected a buy"
    trade = engine.trade_log[0]
    quoted = trade["Shares"] * trade["Price"]
    assert trade["Cost"] > quoted
    assert trade["Fill_Price"] == pytest.approx(100.05)
    assert trade["Commission"] > 0
    assert engine.cash == pytest.approx(10000 - trade["Cost"])
