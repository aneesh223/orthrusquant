"""Buy-and-hold entry convention, performance stats, and batch aggregation."""

import json
from datetime import datetime

import pandas as pd
import pytest

from backtester.batch_backtest import (
    aggregate_percent_alpha,
    summarize_result,
    write_batch_results,
)
from backtester.metrics import buy_and_hold_result, performance_metrics


def test_buy_and_hold_enters_at_the_next_open():
    index = pd.bdate_range("2021-01-04", periods=4, tz="UTC")
    prices = pd.DataFrame(
        {
            "Open": [100.0, 80.0, 80.0, 80.0],
            "High": [101.0, 81.0, 81.0, 91.0],
            "Low": [99.0, 79.0, 79.0, 79.0],
            "Close": [100.0, 80.0, 80.0, 90.0],
            "Volume": 1000,
        },
        index=index,
    )
    result = buy_and_hold_result(prices, index, 8000, commission_bps=0, slippage_bps=0)
    # Day-1 close is 100. The strategy cannot fill there; the next open is 80.
    assert result["quoted_entry"] == 80.0
    assert result["entry_date"] == index[1]
    assert result["shares"] == 100
    assert result["final_value"] == pytest.approx(9000.0)
    assert result["return_pct"] == pytest.approx(12.5)
    # Entering at the first close would have been (90-100)/100 = -10%.
    assert result["return_pct"] != pytest.approx(-10.0)


def test_buy_and_hold_needs_a_second_session():
    index = pd.bdate_range("2021-01-04", periods=1, tz="UTC")
    prices = pd.DataFrame({"Open": [10.0], "Close": [12.0]}, index=index)
    result = buy_and_hold_result(prices, index, 1000, commission_bps=0, slippage_bps=0)
    assert result["return_pct"] == 0.0
    assert result["entry_price"] is None


def test_performance_metrics_cover_return_cagr_sharpe_and_drawdown():
    history = [
        {"Date": datetime(2021, 1, 4), "Portfolio_Value": 100.0},
        {"Date": datetime(2021, 1, 5), "Portfolio_Value": 110.0},
        {"Date": datetime(2021, 1, 6), "Portfolio_Value": 90.0},
        {"Date": datetime(2021, 1, 7), "Portfolio_Value": 100.0},
    ]
    stats = performance_metrics(history, 100.0, num_trades=2)
    assert stats["total_return_pct"] == pytest.approx(0.0)
    assert stats["num_trades"] == 2
    assert stats["cagr_pct"] == pytest.approx(0.0)
    assert isinstance(stats["sharpe"], float)
    assert stats["max_drawdown_pct"] == pytest.approx((90.0 - 110.0) / 110.0 * 100.0)


def test_spy_benchmark_uses_the_same_entry(monkeypatch):
    from backtester.backtester import AlpacaBacktester
    from src import analyzer

    analyzer.get_ai_analyzer.cache_clear()
    monkeypatch.setattr(analyzer, "_ensure_model", lambda: False)

    index = pd.bdate_range("2021-01-04", periods=30, tz="UTC")
    close = pd.Series(range(100, 130), index=index, dtype=float)
    prices = pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1000},
        index=index,
    )
    spy_close = close * 0.5
    spy = pd.DataFrame(
        {"Open": spy_close, "High": spy_close + 1, "Low": spy_close - 1, "Close": spy_close, "Volume": 1000},
        index=index,
    )
    engine = AlpacaBacktester(
        "TEST", strategy="value", initial_cash=10000,
        commission_bps=0, slippage_bps=0, benchmark="SPY",
    )
    engine.full_price_data = prices
    engine.full_news_data = []
    engine.benchmark_prices = spy
    # Short window skips the transformer and still produces a benchmark figure.
    start = index[0].to_pydatetime()
    end = index[5].to_pydatetime()
    results = engine.run(start, end)
    assert results["benchmark_ticker"] == "SPY"
    assert results["benchmark_return_pct"] is not None
    assert "cagr_pct" in results
    assert "sharpe" in results
    assert "max_drawdown_pct" in results
    assert results["num_trades"] == len(results["trade_log"])
    # Entry is the second session's open, not the first close.
    assert results["buy_hold_entry_date"] == index[1]


def test_batch_percent_alpha_is_computed_from_fields_and_written(tmp_path):
    rows = [
        {"success": True, "ticker": "AAPL", "strategy": "MOMENTUM",
         "total_return_pct": 10.0, "buy_hold_return_pct": 4.0, "num_trades": 3},
        {"success": True, "ticker": "MSFT", "strategy": "VALUE",
         "total_return_pct": 1.0, "buy_hold_return_pct": 5.0, "num_trades": 1},
        {"success": False, "ticker": "BAD", "strategy": "VALUE", "error": "no data"},
    ]
    summarized = [summarize_result(row) for row in rows]
    assert summarized[0]["percent_alpha"] == pytest.approx(6.0)
    assert summarized[1]["percent_alpha"] == pytest.approx(-4.0)
    summary = aggregate_percent_alpha(summarized)
    assert summary["mean_percent_alpha"] == pytest.approx(1.0)
    assert summary["count"] == 2

    json_path, csv_path = write_batch_results(
        tmp_path / "batch_results",
        rows,
        seed=0,
        params={"num_tests": 3, "benchmark": "SPY"},
    )
    payload = json.loads(open(json_path, encoding="utf-8").read())
    assert payload["seed"] == 0
    assert payload["params"]["benchmark"] == "SPY"
    assert payload["summary"]["mean_percent_alpha"] == pytest.approx(1.0)
    text = open(csv_path, encoding="utf-8").read()
    assert "percent_alpha" in text
    assert "AAPL" in text
