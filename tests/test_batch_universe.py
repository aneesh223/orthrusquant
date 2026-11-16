"""Ticker pools are not double-counted, and the batch summary says so."""

import pytest

from backtester.batch_backtest import TICKER_POOLS, aggregate_percent_alpha


def test_each_ticker_appears_in_exactly_one_pool():
    seen = {}
    for pool, names in TICKER_POOLS.items():
        for name in names:
            assert name not in seen, f"{name} is in both {seen.get(name)} and {pool}"
            seen[name] = pool


def test_aggregate_reports_failures_spread_and_median():
    rows = [
        {"success": True, "percent_alpha": 6.0, "sentiment_coverage_pct": 50.0},
        {"success": True, "percent_alpha": -4.0, "sentiment_coverage_pct": 10.0},
        {"success": False, "percent_alpha": None, "error": "no data"},
        {"success": False, "percent_alpha": None, "error": "timeout"},
    ]
    summary = aggregate_percent_alpha(rows)
    assert summary["failed_count"] == 2
    assert summary["failure_reasons"] == ["no data", "timeout"]
    assert summary["mean_percent_alpha"] == pytest.approx(1.0)
    assert summary["median_percent_alpha"] == pytest.approx(1.0)
    # Sample standard deviation of +6 and -4.
    assert summary["std_percent_alpha"] == pytest.approx(7.0710678118654755)


def test_zero_coverage_runs_are_left_out_of_the_headline_mean():
    rows = [
        {"success": True, "percent_alpha": 4.0, "sentiment_coverage_pct": 25.0},
        {"success": True, "percent_alpha": 100.0, "sentiment_coverage_pct": 0.0},
    ]
    summary = aggregate_percent_alpha(rows)
    assert summary["mean_percent_alpha"] == pytest.approx(4.0)
    assert summary["count"] == 1
    assert summary["zero_coverage_count"] == 1
