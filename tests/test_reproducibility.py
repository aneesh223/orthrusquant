"""Seed 0, the fixed ticker universe, optimal windows, and the date split."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from backtester.batch_backtest import (
    TICKER_POOLS,
    VALUE_OPTIMAL_DAYS,
    MOMENTUM_OPTIMAL_DAYS,
    apply_seed,
    get_random_date_range,
    resolve_split_date,
    select_random_parameters,
)
from backtester.backtester import AlpacaBacktester
from src import analyzer
from src.constants import SECTOR_THRESHOLDS


def test_seed_zero_is_applied_and_reproducible():
    assert apply_seed(0) is True
    first = [select_random_parameters()[0] for _ in range(6)]
    assert apply_seed(0) is True
    second = [select_random_parameters()[0] for _ in range(6)]
    assert first == second
    assert apply_seed(None) is False


def test_ticker_universe_is_fixed_and_uses_zm():
    names = [ticker for pool in TICKER_POOLS.values() for ticker in pool]
    assert "ZOOM" not in names
    assert "ZM" in TICKER_POOLS["volatile_growth"]
    assert SECTOR_THRESHOLDS["ZM"] == -0.20
    assert "ZOOM" not in SECTOR_THRESHOLDS
    # Drawing twice from a fixed seed stays inside the published universe.
    apply_seed(1)
    picked = {select_random_parameters()[0] for _ in range(20)}
    assert picked <= set(names)


def test_optimal_periods_are_the_long_windows():
    apply_seed(2)
    start, end = get_random_date_range("m", optimal_periods=True)
    span = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days
    assert span in MOMENTUM_OPTIMAL_DAYS or span >= 1095
    apply_seed(3)
    start, end = get_random_date_range("v", optimal_periods=True)
    span = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days
    assert span in VALUE_OPTIMAL_DAYS or span >= 1825


def test_train_ratio_derives_a_date_inside_the_window():
    split = resolve_split_date("2021-01-01", "2021-12-31", train_ratio=0.7)
    assert datetime(2021, 1, 1) < split < datetime(2021, 12, 31)
    explicit = resolve_split_date("2021-01-01", "2021-12-31", split_date="2021-06-01", train_ratio=0.1)
    assert explicit == datetime(2021, 6, 1)
    with pytest.raises(ValueError):
        resolve_split_date("2021-01-01", "2021-12-31", train_ratio=1.5)


def test_split_reports_only_the_held_out_book(monkeypatch):
    analyzer.get_ai_analyzer.cache_clear()
    monkeypatch.setattr(analyzer, "_ensure_model", lambda: False)

    index = pd.bdate_range("2021-01-04", periods=40, tz="UTC")
    rng = np.random.default_rng(4)
    close = 40 + np.cumsum(rng.normal(0, 0.3, len(index)))
    close = np.maximum(close, 1.0)
    prices = pd.DataFrame(
        {"Open": close, "High": close + 0.4, "Low": close - 0.4, "Close": close, "Volume": 1000},
        index=index,
    )
    engine = AlpacaBacktester("TEST", strategy="value", initial_cash=10000, commission_bps=0, slippage_bps=0)
    engine.full_price_data = prices
    engine.full_news_data = []
    split = index[20].to_pydatetime()
    results = engine.run(index[0].to_pydatetime(), index[-1].to_pydatetime(), split_date=split)
    assert results["split_applied"] is True
    assert results["frozen_buy_threshold"] is not None
    assert results["frozen_sell_threshold"] is not None
    assert results["portfolio_history"]
    assert all(row["Date"] >= split for row in results["portfolio_history"])
    assert results["eval_start"] >= split
    # The reported trade count is the held-out book, not the training trades.
    assert results["num_trades"] == len(results["trade_log"])
    assert results["train_num_trades"] == len(results["train_trade_log"])
