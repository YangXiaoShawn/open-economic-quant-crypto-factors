"""Verify published factor transfer and post-publication OOS tables."""
from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "crypto_factors"))

from crypto_factors.cli import main


class TestCryptoFactorsPublishedResults(unittest.TestCase):
    def test_factor_transfer_summary(self) -> None:
        path = ROOT / "reports" / "factor_transfer_summary.csv"
        with path.open("r", encoding="utf-8") as fh:
            rows = {r["factor"]: r for r in csv.DictReader(fh)}
        self.assertEqual(len(rows), 7)
        self.assertAlmostEqual(float(rows["Funding rate"]["sharpe_730d"]), 1.42, places=2)
        self.assertAlmostEqual(float(rows["Retail long/short ratio"]["sharpe_730d"]), 1.39, places=2)
        self.assertAlmostEqual(float(rows["Low-price range"]["sharpe_730d"]), 1.63, places=2)
        self.assertAlmostEqual(float(rows["Cross-sectional momentum"]["sharpe_730d"]), -0.38, places=2)

    def test_postpub_composite_consistency_with_raw_csv(self) -> None:
        raw_path = ROOT / "data" / "results" / "report_postpub_monthly.csv"
        with raw_path.open("r", encoding="utf-8") as fh:
            raw_rows = list(csv.DictReader(fh))
        prod_b = [r for r in raw_rows if "产品B·大币top8·对冲beta" in r["strategy"]][0]
        self.assertAlmostEqual(float(prod_b["2026-06"]) * 100, 4.57, places=2)
        self.assertAlmostEqual(float(prod_b["2026-07"]) * 100, -6.93, places=2)
        self.assertAlmostEqual(float(prod_b["2026-08"]) * 100, -3.97, places=2)

    def test_cli_summary_runs(self) -> None:
        self.assertEqual(main(["summary"]), 0)


if __name__ == "__main__":
    unittest.main()
