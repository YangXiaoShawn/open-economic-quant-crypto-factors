"""Transaction-cost accounting: long-short books are charged on traded notional of both legs,
and composite overlays are charged when they change sleeve allocations."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from crypto_factors.costs import (  # noqa: E402
    cost_from_traded,
    leg_traded_notional,
    long_short_traded_notional,
    overlay_rebalance_costs,
)

HAS_PANDAS = importlib.util.find_spec("pandas") is not None


class TestTradedNotional(unittest.TestCase):
    def test_one_swap_per_leg_trades_two_units(self) -> None:
        # Two names per leg; one name swapped in each leg: sell 0.5 + buy 0.5 per leg.
        traded = long_short_traded_notional({"A", "B"}, {"A", "C"}, {"X", "Y"}, {"X", "Z"})
        self.assertAlmostEqual(traded, 2.0)
        # 5 bp per side on 2 units of notional = 10 bp of capital.
        self.assertAlmostEqual(cost_from_traded(traded, 5.0), 0.0010)
        # The pre-2026-10-26 formula charged the average of the two legs: half of this.
        old_to = (len({"A", "B"} ^ {"A", "C"}) + len({"X", "Y"} ^ {"X", "Z"})) / 4
        self.assertAlmostEqual(cost_from_traded(traded, 5.0), 2.0 * old_to * 5.0 / 1e4)

    def test_initial_build_and_full_rotation(self) -> None:
        self.assertAlmostEqual(leg_traded_notional(set(), {"A", "B"}), 1.0)
        self.assertAlmostEqual(leg_traded_notional({"A", "B"}, {"C", "D"}), 2.0)
        self.assertAlmostEqual(leg_traded_notional({"A", "B"}, {"A", "B"}), 0.0)

    def test_leg_size_change_reweights_names_that_stay(self) -> None:
        # 1/2 -> 1/3 on A and B (2 x 1/6) plus buying 1/3 of C.
        self.assertAlmostEqual(leg_traded_notional({"A", "B"}, {"A", "B", "C"}), 2.0 / 3.0)


class TestOverlayRebalanceCosts(unittest.TestCase):
    def test_weight_change_charges_gross_notional(self) -> None:
        exposures = [{"a": 0.5, "b": 0.5}, {"a": 0.75, "b": 0.25}, {"a": 0.75, "b": 0.25}]
        costs = overlay_rebalance_costs(exposures, cost_bps_per_side=5.0)
        # sum |dw| = 0.5 of capital in $1-long/$1-short sleeves -> 1.0 notional -> 5 bp.
        self.assertEqual(len(costs), 3)
        self.assertAlmostEqual(costs[0], 0.0)
        self.assertAlmostEqual(costs[1], 0.0005)
        self.assertAlmostEqual(costs[2], 0.0)

    def test_leverage_change_is_charged(self) -> None:
        costs = overlay_rebalance_costs([{"c": 1.0}, {"c": 1.5}, {"c": 1.2}], cost_bps_per_side=5.0)
        self.assertAlmostEqual(costs[1], 0.5 * 2.0 * 5.0 / 1e4)
        self.assertAlmostEqual(costs[2], 0.3 * 2.0 * 5.0 / 1e4)
        self.assertEqual(overlay_rebalance_costs([{"c": 1.0}, {"c": 1.5}], cost_bps_per_side=0.0), [0.0, 0.0])


@unittest.skipUnless(HAS_PANDAS, "pandas is required for the backtest toy example")
class TestBacktestFactorToyPanel(unittest.TestCase):
    def test_long_short_cost_is_charged_on_both_legs(self) -> None:
        import pandas as pd

        from crypto_factors.backtest import COST_BPS, backtest_factor

        syms = [f"S{i}" for i in range(10)]
        # Quintiles of 10 names hold 2 names per leg. All forward returns are zero,
        # so each day's long-short return is minus the cost charged that day.
        day_signals = {
            "2026-01-01": list(range(10)),                  # long S8,S9  short S0,S1
            "2026-01-02": [0, 2, 1, 3, 4, 5, 6, 9, 8, 7],   # long S7,S8  short S0,S2
            "2026-01-03": [0, 2, 1, 3, 4, 5, 6, 9, 8, 7],   # unchanged
        }
        rows = [
            {"day": pd.Timestamp(day), "sym": sym, "sig_col": float(sig[i]), "fwd_ret": 0.0}
            for day, sig in day_signals.items()
            for i, sym in enumerate(syms)
        ]
        bt = backtest_factor(pd.DataFrame(rows), "sig_col", +1)
        unit = COST_BPS / 1e4
        # Day 1 builds both legs (1 + 1 units), day 2 swaps one name per leg (1 + 1 units).
        self.assertEqual(list(bt["traded"].round(12)), [2.0, 2.0, 0.0])
        self.assertEqual(list(bt["turnover"].round(12)), [1.0, 1.0, 0.0])
        for got, want in zip(bt["ls"], [-2.0 * unit, -2.0 * unit, 0.0]):
            self.assertAlmostEqual(got, want)
        # Long-only line: one leg of gross 1.
        for got, want in zip(bt["long"], [-1.0 * unit, -1.0 * unit, 0.0]):
            self.assertAlmostEqual(got, want)


if __name__ == "__main__":
    unittest.main()
