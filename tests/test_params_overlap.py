"""The split does not hold out parameters that were chosen on this history."""

import numpy as np
import pandas as pd

from backtester.backtester import AlpacaBacktester


def _run(start):
    index = pd.bdate_range(start, periods=6, tz="UTC")
    close = np.linspace(20.0, 23.0, len(index))
    prices = pd.DataFrame(
        {"Open": close, "High": close + 0.2, "Low": close - 0.2, "Close": close, "Volume": 1000},
        index=index,
    )
    engine = AlpacaBacktester("TEST", initial_cash=10_000.0)
    engine.full_price_data = prices
    engine.full_news_data = []
    return engine.run(index[0].to_pydatetime(), index[-1].to_pydatetime())


def test_eval_window_starting_in_2025_overlaps_tuned_parameters():
    results = _run("2025-06-02")
    assert results["params_tuned_on_eval_window"] is True


def test_eval_window_after_the_parameter_edit_is_not_flagged():
    results = _run("2026-04-01")
    assert results["params_tuned_on_eval_window"] is False
