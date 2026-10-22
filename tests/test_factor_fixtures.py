"""Verify sample perpetual futures fixture schema and values."""
from __future__ import annotations

import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestCryptoFactorsFixtures(unittest.TestCase):
    def test_sample_perp_panel_fixture(self) -> None:
        path = ROOT / "data" / "fixtures" / "sample_perp_panel.csv"
        with path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 4)
        symbols = [r["symbol"] for r in rows]
        self.assertEqual(symbols, ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
        for row in rows:
            self.assertGreater(float(row["close"]), 0.0)
            self.assertGreater(float(row["retail_ls_ratio"]), 0.0)


if __name__ == "__main__":
    unittest.main()
