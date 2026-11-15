"""Fill costs for backtest orders.

Commission and slippage are both expressed in basis points per side.
A buy pays a higher fill and a commission on that notional. A sell receives
a lower fill and pays a commission on the proceeds before the commission.
"""


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
