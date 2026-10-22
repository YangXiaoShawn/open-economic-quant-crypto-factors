"""Verify post-publication cumulative window invariants."""
from __future__ import annotations

import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestCryptoFactorsPostPubWindows(unittest.TestCase):
    def test_postpub_summary_rows(self) -> None:
        path = ROOT / "reports" / "postpub_composite_summary.csv"
        with path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 5)
        self.assertEqual(int(rows[0]["days"]), 515)
        self.assertEqual(int(rows[-1]["days"]), 92)
        self.assertAlmostEqual(float(rows[-1]["market_neutral_return_pct"]), -6.55, places=2)


if __name__ == "__main__":
    unittest.main()
