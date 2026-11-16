"""Sector offsets must reach the backtester, not only the shared function."""

import pytest

from backtester.backtester import AlpacaBacktester
from src.logic import calculate_adaptive_thresholds


def test_zm_bull_buy_threshold_includes_the_sector_offset():
    # Short history skips the ±10% volatility nudge, so the sector offset is
    # the only difference. 4.25 - 0.20 = 4.05, inside the [4.0, 6.0] clamp.
    without = calculate_adaptive_thresholds([], "BULL", strategy="momentum")
    engine = AlpacaBacktester("ZM", strategy="momentum")
    buy, _sell = engine.calculate_adaptive_thresholds([], "BULL")
    assert buy == pytest.approx(without[0] - 0.20)
    assert 4.0 < buy < 6.0
