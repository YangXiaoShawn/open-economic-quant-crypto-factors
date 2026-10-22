"""Verify published 730-day factor transfer, 92-day post-publication OOS tables, and oeq-core sample pipeline."""
from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from crypto_factors import config, run_sample_pipeline
from crypto_factors.cli import main


class TestCryptoPublishedResults(unittest.TestCase):
    def test_factor_transfer_summary(self) -> None:
        path = ROOT / "reports" / "factor_transfer_summary.csv"
        with path.open("r", encoding="utf-8") as fh:
            rows = {r["factor"]: r for r in csv.DictReader(fh)}
        self.assertEqual(len(rows), 7)
        self.assertAlmostEqual(float(rows["Funding rate"]["sharpe_730d"]), 1.42, places=2)
        self.assertAlmostEqual(float(rows["Retail long/short ratio"]["sharpe_730d"]), 1.39, places=2)
        self.assertAlmostEqual(float(rows["Low-price range"]["sharpe_730d"]), 1.63, places=2)
        self.assertAlmostEqual(float(rows["Cross-sectional momentum"]["sharpe_730d"]), -0.38, places=2)

    def test_postpub_composite_windows(self) -> None:
        path = ROOT / "reports" / "postpub_composite_summary.csv"
        with path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 5)
        self.assertAlmostEqual(float(rows[1]["market_neutral_return_pct"]), 4.57, places=2)
        self.assertAlmostEqual(float(rows[2]["market_neutral_return_pct"]), -6.93, places=2)
        self.assertAlmostEqual(float(rows[3]["market_neutral_return_pct"]), -3.97, places=2)
        self.assertAlmostEqual(float(rows[4]["market_neutral_return_pct"]), -6.55, places=2)

    def test_sample_pipeline_and_oeq_core_fdr(self) -> None:
        self.assertTrue(config.FIXTURES_DIR.exists())
        result = run_sample_pipeline(root=ROOT)
        self.assertEqual(result["claim_boundary_grade"], "BACKTEST_ASSOCIATION")
        self.assertEqual(result["sample_fixture_dates"], 10)
        self.assertEqual(result["regime_variants_evaluated"], 3)
        self.assertEqual(result["verified_files"], 4)
        self.assertGreater(result["regime_conditioned_postpub_92d_pct"], result["postpub_92d_return_pct"])
        fdr = result["sample_factor_fdr"]
        self.assertGreater(fdr["funding_contrarian"]["mean_ic"], 0.0)
        self.assertGreater(fdr["low_price_amp"]["mean_ic"], 0.0)
        self.assertGreater(result["sample_static_composite_mean_net"], 0.0)
        self.assertGreater(result["sample_vol_targeted_mean_net"], 0.0)
        self.assertGreater(result["sample_regime_conditioned_mean_net"], 0.0)

    def test_cli_summary_runs(self) -> None:
        self.assertEqual(main(["reproduce-sample"]), 0)


if __name__ == "__main__":
    unittest.main()
