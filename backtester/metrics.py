"""Fill costs and performance statistics for backtest orders.

Commission and slippage are both expressed in basis points per side.
A buy pays a higher fill and a commission on that notional. A sell receives
a lower fill and pays a commission on the proceeds before the commission.
"""

from datetime import datetime

import numpy as np


def _rates(commission_bps, slippage_bps):
    return commission_bps / 10_000.0, slippage_bps / 10_000.0


def buy_fill(price, shares, commission_bps, slippage_bps):
    """Return ``(fill_price, notional, commission, cash_outlay)``."""
    commission_rate, slippage = _rates(commission_bps, slippage_bps)
    fill_price = float(price) * (1.0 + slippage)
    notional = shares * fill_price
    commission = notional * commission_rate
    return fill_price, notional, commission, notional + commission


def sell_fill(price, shares, commission_bps, slippage_bps):
    """Return ``(fill_price, notional, commission, cash_proceeds)``."""
    commission_rate, slippage = _rates(commission_bps, slippage_bps)
    fill_price = float(price) * (1.0 - slippage)
    notional = shares * fill_price
    commission = notional * commission_rate
    return fill_price, notional, commission, notional - commission


def affordable_shares(cash, price, commission_bps, slippage_bps):
    """Whole shares whose all-in cost does not exceed ``cash``."""
    if price <= 0 or cash <= 0:
        return 0
    _, _, _, one_share = buy_fill(price, 1, commission_bps, slippage_bps)
    if one_share <= 0:
        return 0
    return int(cash / one_share)


def _bar_value(price_data, when, column):
    """Look up one OHLC field, matching the timestamp or its calendar date."""
    if when in price_data.index:
        value = price_data.loc[when, column]
    else:
        target = when.date() if isinstance(when, datetime) else when
        match = None
        for ts in price_data.index:
            ts_date = ts.date() if isinstance(ts, datetime) else ts
            if ts_date == target:
                match = ts
                break
        if match is None:
            raise KeyError(when)
        value = price_data.loc[match, column]
    if hasattr(value, "iloc"):
        value = value.iloc[0]
    return float(value)


def buy_and_hold_result(price_data, trading_days, initial_cash, commission_bps=5.0, slippage_bps=5.0):
    """Buy-and-hold using the strategy's entry convention.

    Signals are formed on a bar and filled at the next open, so the earliest
    comparable entry is the open of the second session, not the first close.
    The position is marked at the final close, same as an open strategy book,
    after paying the entry-side commission and slippage.
    """
    days = list(trading_days)
    empty = {
        "return_pct": 0.0,
        "final_value": float(initial_cash),
        "entry_price": None,
        "quoted_entry": None,
        "entry_date": None,
        "shares": 0,
    }
    if len(days) < 2 or initial_cash <= 0 or price_data is None or len(price_data) == 0:
        return empty
    try:
        quoted = _bar_value(price_data, days[1], "Open")
        final_price = _bar_value(price_data, days[-1], "Close")
    except (KeyError, TypeError, ValueError):
        return empty
    if quoted <= 0 or final_price <= 0:
        return empty
    shares = affordable_shares(initial_cash, quoted, commission_bps, slippage_bps)
    _, _, _, spent = buy_fill(quoted, shares, commission_bps, slippage_bps)
    final_value = (initial_cash - spent) + shares * final_price
    return {
        "return_pct": (final_value - initial_cash) / initial_cash * 100.0,
        "final_value": float(final_value),
        "entry_price": float(quoted * (1.0 + slippage_bps / 10_000.0)),
        "quoted_entry": float(quoted),
        "entry_date": days[1],
        "shares": int(shares),
    }


def performance_metrics(portfolio_history, initial_cash, num_trades):
    """Percent return, CAGR, Sharpe, max drawdown, and trade count.

    Sharpe uses daily portfolio returns and a 252-day year. CAGR uses the
    calendar span of the recorded history. Drawdown is negative when the
    book falls below its peak.
    """
    blank = {
        "total_return_pct": 0.0,
        "cagr_pct": None,
        "sharpe": None,
        "max_drawdown_pct": 0.0,
        "num_trades": int(num_trades),
    }
    if not portfolio_history or initial_cash <= 0:
        return blank
    values = np.array([float(row["Portfolio_Value"]) for row in portfolio_history], dtype=float)
    dates = [row["Date"] for row in portfolio_history]
    final = float(values[-1])
    total_return_pct = (final - initial_cash) / initial_cash * 100.0
    cagr_pct = None
    if len(dates) > 1 and final > 0:
        span_days = (dates[-1] - dates[0]).days
        if span_days > 0:
            cagr_pct = ((final / initial_cash) ** (365.25 / span_days) - 1.0) * 100.0
    sharpe = None
    if len(values) >= 3:
        returns = np.diff(values) / values[:-1]
        returns = returns[np.isfinite(returns)]
        if len(returns) >= 2:
            std = float(np.std(returns, ddof=1))
            if std > 0:
                sharpe = float(np.mean(returns) / std * np.sqrt(252.0))
    peak = np.maximum.accumulate(values)
    drawdowns = np.where(peak > 0, (values - peak) / peak, 0.0)
    return {
        "total_return_pct": float(total_return_pct),
        "cagr_pct": None if cagr_pct is None else float(cagr_pct),
        "sharpe": sharpe,
        "max_drawdown_pct": float(np.min(drawdowns) * 100.0),
        "num_trades": int(num_trades),
    }
