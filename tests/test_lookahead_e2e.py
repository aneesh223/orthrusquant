"""End-to-end look-ahead tests for AlpacaBacktester.run().

These drive the real simulation loop with injected prices and headlines and
check that nothing dated on or before a cutoff session T depends on:

(a) what happens after T (future prices, future headline polarity, extra
    future-only headlines),
(b) whether the data even extends past T (truncation, both windows > 180 d),
(c) the same, with the truncated window under 180 days,
(d) the wall-clock date the backtest is run on,

and (e) documents the one same-day dependency that is allowed: a headline
published after the close on T changes T's score, which is safe only because
orders queued on T fill at the T+1 open.

Sentiment uses VADER only. No network and no model weights.
"""

import contextlib
import io
import inspect
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from backtester.backtester import AlpacaBacktester
from src import analyzer, market

N_BARS = 520
SIM_START_BAR = 260
T_BAR = 420
SPLIT_BAR = 330
SHORT_T_BAR = 360

POSITIVE = [
    "Company beats earnings estimates and raises full-year guidance",
    "Analysts upgrade the stock after strong profit growth",
    "Shares rally on record revenue and bullish outlook",
    "Company wins major contract, stock surges",
    "Strong demand lifts sales to an all-time high",
]
NEGATIVE = [
    "Company misses earnings and cuts guidance, shares plunge",
    "Analysts downgrade the stock after weak profit and losses",
    "Shares crash on fraud probe and lawsuit",
    "Company warns of layoffs as sales collapse",
    "Regulators fine the company, stock tumbles to a loss",
]
FLIP = {**{p: n for p, n in zip(POSITIVE, NEGATIVE)}, **{n: p for p, n in zip(POSITIVE, NEGATIVE)}}
SLOTS = [(14, 0), (18, 0), (22, 30)]

PAST_COLUMNS = ["Date", "Cash", "Shares", "Portfolio_Value", "Final_Buy_Score", "Market_Regime"]


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def make_prices(seed, start="2018-01-01", n=N_BARS):
    """Business-day bars at 05:00 UTC. Drift flips sign every 60 bars."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=n) + pd.Timedelta(hours=5)
    index = index.tz_localize("UTC")
    drift = np.where((np.arange(n) // 60) % 2 == 0, 0.0015, -0.0015)
    rets = drift + rng.normal(0.0, 0.015, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0.0, 0.005, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


def make_news(seed, prices):
    """Up to three headlines per session, newest-first like the scraper."""
    rng = np.random.default_rng(seed + 1000)
    rows = []
    for ts in prices.index:
        day = ts.normalize()
        for hour, minute in SLOTS:
            if rng.random() < 0.6:
                pool = POSITIVE if rng.random() < 0.5 else NEGATIVE
                text = pool[int(rng.integers(len(pool)))]
                when = (day + pd.Timedelta(hours=hour, minutes=minute)).to_pydatetime()
                rows.append((when, text, "Reuters"))
    rows.sort(key=lambda r: r[0], reverse=True)
    return rows


def end_of_utc_day(ts):
    return ts.normalize() + pd.Timedelta(days=1)


def perturb_future(seed, prices, news, t_ts):
    """Change only what is unknowable on T: later bars and later headlines."""
    rng = np.random.default_rng(seed + 2000)
    p = prices.copy()
    after = p.index > t_ts
    p.loc[after, ["Open", "High", "Low", "Close"]] *= 1.7

    cutoff = end_of_utc_day(t_ts).to_pydatetime()
    n_news = [(ts, FLIP[text] if ts >= cutoff else text, src) for ts, text, src in news]

    future_days = p.index[after]
    picks = rng.choice(len(future_days), size=29, replace=False)
    for k in picks:
        when = (future_days[k].normalize() + pd.Timedelta(hours=16, minutes=7)).to_pydatetime()
        n_news.append((when, NEGATIVE[int(k) % len(NEGATIVE)], "Bloomberg"))
    n_news.sort(key=lambda r: r[0], reverse=True)
    return p, n_news


@pytest.fixture(autouse=True)
def _vader_only(monkeypatch):
    analyzer.get_ai_analyzer.cache_clear()
    monkeypatch.setattr(analyzer, "_ensure_model", lambda: False)
    yield
    analyzer.get_ai_analyzer.cache_clear()


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------

def run_backtest(prices, news, strategy, start, end, split=None):
    engine = AlpacaBacktester("TEST", strategy=strategy, initial_cash=10_000.0)
    engine.full_price_data = prices.copy()
    engine.full_news_data = list(news)
    kwargs = {"lookback_days": 30}
    if split is not None:
        if "split_date" not in inspect.signature(engine.run).parameters:
            pytest.skip("run() has no split_date on this revision")
        kwargs["split_date"] = split
    with contextlib.redirect_stdout(io.StringIO()):
        return engine.run(start.to_pydatetime(), end.to_pydatetime(), **kwargs)


def history_through(results, t_ts):
    df = pd.DataFrame(results["portfolio_history"])
    assert not df.empty, "simulation produced no history"
    df = df[df["Date"] <= t_ts][PAST_COLUMNS].reset_index(drop=True)
    return df


def trades_through(results, t_ts):
    rows = list(results.get("train_trade_log", [])) + list(results["trade_log"])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[df["Date"] <= t_ts].reset_index(drop=True)


def assert_past_identical(base, alt, t_ts):
    hb, ha = history_through(base, t_ts), history_through(alt, t_ts)
    assert len(hb) > 0
    pd.testing.assert_frame_equal(hb, ha, check_exact=True)
    tb, ta = trades_through(base, t_ts), trades_through(alt, t_ts)
    pd.testing.assert_frame_equal(tb, ta, check_exact=True)


CASES = [(0, "momentum"), (1, "value"), (2, "momentum")]


# --------------------------------------------------------------------------
# (a) Perturb only the future
# --------------------------------------------------------------------------

@pytest.mark.parametrize("use_split", [False, True], ids=["nosplit", "split"])
@pytest.mark.parametrize("strategy", ["momentum", "value"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_future_perturbation_does_not_change_the_past(seed, strategy, use_split):
    prices = make_prices(seed)
    news = make_news(seed, prices)
    start, end, t_ts = prices.index[SIM_START_BAR], prices.index[-1], prices.index[T_BAR]
    split = prices.index[SPLIT_BAR].to_pydatetime() if use_split else None

    alt_prices, alt_news = perturb_future(seed, prices, news, t_ts)
    # Sanity: the perturbation leaves everything up to the end of T's UTC day alone.
    pd.testing.assert_frame_equal(prices[prices.index <= t_ts], alt_prices[alt_prices.index <= t_ts])
    cutoff = end_of_utc_day(t_ts).to_pydatetime()
    assert [r for r in news if r[0] < cutoff] == [r for r in alt_news if r[0] < cutoff]

    base = run_backtest(prices, news, strategy, start, end, split)
    alt = run_backtest(alt_prices, alt_news, strategy, start, end, split)

    assert_past_identical(base, alt, t_ts)

    # The future must actually differ, or this test passes by doing nothing.
    fb = pd.DataFrame(base["portfolio_history"])
    fa = pd.DataFrame(alt["portfolio_history"])
    fb, fa = fb[fb["Date"] > t_ts], fa[fa["Date"] > t_ts]
    assert len(fb) == len(fa) > 0
    assert not np.allclose(fb["Final_Buy_Score"].values, fa["Final_Buy_Score"].values)
    assert not np.allclose(fb["Price"].values, fa["Price"].values)


# --------------------------------------------------------------------------
# (b) Truncate at T, both windows > 180 calendar days
# --------------------------------------------------------------------------

def _truncate(prices, news, t_ts):
    cutoff = end_of_utc_day(t_ts).to_pydatetime()
    return prices[prices.index <= t_ts].copy(), [r for r in news if r[0] < cutoff]


@pytest.mark.parametrize("seed,strategy", CASES)
def test_b_truncating_at_T_does_not_change_the_past(seed, strategy):
    prices = make_prices(seed)
    news = make_news(seed, prices)
    start, end, t_ts = prices.index[SIM_START_BAR], prices.index[-1], prices.index[T_BAR]
    assert (t_ts - start).days > 180 and (end - start).days > 180

    full = run_backtest(prices, news, strategy, start, end)
    tp, tn = _truncate(prices, news, t_ts)
    short = run_backtest(tp, tn, strategy, start, t_ts)
    assert_past_identical(full, short, t_ts)


# --------------------------------------------------------------------------
# (c) Truncate at bar 360, window < 180 calendar days
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed,strategy", CASES)
@pytest.mark.xfail(strict=True, reason="window-length regime cadence, fixed in item 2")
def test_c_short_window_truncation_does_not_change_the_past(seed, strategy):
    prices = make_prices(seed)
    news = make_news(seed, prices)
    start, end, t_ts = prices.index[SIM_START_BAR], prices.index[-1], prices.index[SHORT_T_BAR]
    assert (t_ts - start).days < 180 < (end - start).days

    full = run_backtest(prices, news, strategy, start, end)
    tp, tn = _truncate(prices, news, t_ts)
    short = run_backtest(tp, tn, strategy, start, t_ts)
    assert_past_identical(full, short, t_ts)


# --------------------------------------------------------------------------
# (d) Wall-clock independence
# --------------------------------------------------------------------------

def _frozen_datetime(fixed):
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = fixed if tz is None else fixed.replace(tzinfo=timezone.utc).astimezone(tz)
            return cls(*value.timetuple()[:6], value.microsecond, tzinfo=value.tzinfo)

        @classmethod
        def today(cls):
            return cls.now()

    return FrozenDatetime


@pytest.mark.parametrize("strategy", ["momentum", "value"])
def test_d_results_do_not_depend_on_the_wall_clock(monkeypatch, strategy):
    prices = make_prices(3, start="2024-11-01")
    assert prices.index[-1] < pd.Timestamp("2026-11-01", tz="UTC")
    news = make_news(3, prices)
    start, end = prices.index[SIM_START_BAR], prices.index[-1]

    monkeypatch.setattr(market, "datetime", _frozen_datetime(datetime(2026, 11, 15, 12, 0)))
    today = run_backtest(prices, news, strategy, start, end)
    monkeypatch.setattr(market, "datetime", _frozen_datetime(datetime(2027, 5, 15, 12, 0)))
    later = run_backtest(prices, news, strategy, start, end)

    assert_past_identical(today, later, end)
    assert today["final_portfolio_value"] == later["final_portfolio_value"]


# --------------------------------------------------------------------------
# (e) Documentation: an after-close headline on T moves T's score
# --------------------------------------------------------------------------

def test_e_after_close_headline_changes_T_score_but_fills_wait_for_T_plus_1():
    prices = make_prices(0)
    news = make_news(0, prices)
    start, end, t_ts = prices.index[SIM_START_BAR], prices.index[-1], prices.index[T_BAR]
    late = (t_ts.normalize() + pd.Timedelta(hours=22, minutes=30)).to_pydatetime()

    # Pin T's own headlines: one positive at 14:00 in both runs, and a negative
    # one at 22:30 UTC (after the US close) in the second run only.
    t_day, t_next = t_ts.normalize().to_pydatetime(), end_of_utc_day(t_ts).to_pydatetime()
    morning = (t_ts.normalize() + pd.Timedelta(hours=14)).to_pydatetime()
    without = [r for r in news if not (t_day <= r[0] < t_next)] + [(morning, POSITIVE[0], "Reuters")]
    without.sort(key=lambda r: r[0], reverse=True)
    with_late = sorted(without + [(late, NEGATIVE[2], "Reuters")], key=lambda r: r[0], reverse=True)

    a = run_backtest(prices, without, "momentum", start, end)
    b = run_backtest(prices, with_late, "momentum", start, end)

    prev = prices.index[T_BAR - 1]
    assert_past_identical(a, b, prev)

    ha, hb = pd.DataFrame(a["portfolio_history"]), pd.DataFrame(b["portfolio_history"])
    score_a = ha.loc[ha["Date"] == t_ts, "Final_Buy_Score"].iloc[0]
    score_b = hb.loc[hb["Date"] == t_ts, "Final_Buy_Score"].iloc[0]
    # The 22:30 UTC headline is after the 20:00/21:00 UTC close, yet it is part
    # of T's score. That is allowed only because nothing fills at T's prices.
    assert score_a != score_b

    # No fill dated T can depend on it: trades dated <= T are identical.
    pd.testing.assert_frame_equal(trades_through(a, t_ts), trades_through(b, t_ts), check_exact=True)
    # And every fill is at the session's open, never at a close.
    for res in (a, b):
        for trade in res["trade_log"]:
            assert trade["Price"] == prices.loc[trade["Date"], "Open"]
