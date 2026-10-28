"""Transaction-cost accounting shared by the CryptoFactors backtests.

Convention: ``cost_bps`` is charged **per side**, i.e. on every unit of notional
bought or sold. A position change is therefore charged on the traded notional
``sum_i |w_new_i - w_old_i|``, not on a count of names.

The quintile long-short books hold $1 long (equal weight) and $1 short (equal
weight) per $1 of capital, so they carry gross exposure 2. Rotating every name in
both legs trades 4 units of notional (sell 1 + buy 1 on each leg).

Before 2026-10-26 the long-short backtests charged ``to * cost`` with
``to = (|dL| + |dS|) / (|L| + |S|)``, the *average* of the two legs' traded
notional, which is half of what the book actually trades. Published tables
computed with that formula therefore charged roughly 2.5 bp per side, not the
stated 5 bp. This module is pure Python so that it can be tested without pandas.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence


def leg_traded_notional(prev: Iterable[str], cur: Iterable[str]) -> float:
    """Traded notional of one equal-weight leg with gross exposure 1.0.

    Each leg holds ``1 / len(leg)`` per name. The result is
    ``sum_i |w_cur_i - w_prev_i|`` over the union of both name sets, so a full
    replacement trades 2.0 and a change in the number of names also charges the
    re-weighting of names that stay in the leg.
    """
    prev_set, cur_set = set(prev), set(cur)
    w_prev = 1.0 / len(prev_set) if prev_set else 0.0
    w_cur = 1.0 / len(cur_set) if cur_set else 0.0
    traded = 0.0
    for name in prev_set | cur_set:
        traded += abs((w_cur if name in cur_set else 0.0) - (w_prev if name in prev_set else 0.0))
    return traded


def long_short_traded_notional(
    prev_long: Iterable[str],
    cur_long: Iterable[str],
    prev_short: Iterable[str],
    cur_short: Iterable[str],
) -> float:
    """Traded notional of a $1-long / $1-short equal-weight book (sum over both legs)."""
    return leg_traded_notional(prev_long, cur_long) + leg_traded_notional(prev_short, cur_short)


def cost_from_traded(traded_notional: float, cost_bps_per_side: float) -> float:
    """Cost as a fraction of capital for a given traded notional (per unit of capital)."""
    return float(traded_notional) * float(cost_bps_per_side) / 1e4


def overlay_rebalance_costs(
    exposures: Sequence[Mapping[str, float]],
    cost_bps_per_side: float,
    gross_per_unit: float = 2.0,
) -> list[float]:
    """Daily cost of changing sleeve allocations in a composite of long-short sleeves.

    ``exposures[t][sleeve]`` is the fraction of capital allocated to a sleeve on
    day ``t`` (overlay weight times any leverage multiplier). Each unit allocated to
    a dollar-neutral sleeve carries ``gross_per_unit`` of gross notional (2.0 for
    $1 long + $1 short), so moving an allocation by ``d`` trades
    ``|d| * gross_per_unit`` of notional.

    Day 0 is not charged: the sleeve return series already include the cost of
    building their own positions. Re-balancing for intraday drift and netting
    against the sleeves' own trades or across sleeves are not modelled.
    """
    costs: list[float] = []
    prev: Mapping[str, float] | None = None
    for cur in exposures:
        if prev is None:
            costs.append(0.0)
        else:
            names = set(prev) | set(cur)
            delta = sum(abs(float(cur.get(k, 0.0)) - float(prev.get(k, 0.0))) for k in names)
            costs.append(cost_from_traded(delta * gross_per_unit, cost_bps_per_side))
        prev = cur
    return costs
