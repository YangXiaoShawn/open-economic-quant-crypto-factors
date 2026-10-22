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
        self.assertEqual(result["real_perp_days"], 785)
        self.assertAlmostEqual(result["real_785d_static_sharpe"], 1.87, places=2)
        self.assertAlmostEqual(result["real_785d_inv_vol_gated_sharpe"], 2.19, places=2)
        self.assertAlmostEqual(result["real_785d_inv_vol_vt_sharpe"], 2.12, places=2)
        self.assertAlmostEqual(result["real_785d_inv_vol_vt_max_dd_pct"], -8.55, places=2)
        self.assertAlmostEqual(result["real_785d_inv_vol_vt_calmar"], 4.33, places=2)

    def test_real_785d_perp_validation_table(self) -> None:
        path = ROOT / "reports" / "real_785d_perp_validation.csv"
        with path.open("r", encoding="utf-8") as fh:
            rows = {r["strategy"]: r for r in csv.DictReader(fh)}
        self.assertEqual(len(rows), 4)
        self.assertAlmostEqual(float(rows["static_equal_weight"]["full_785d_sharpe"]), 1.87, places=2)
        self.assertAlmostEqual(float(rows["inv_vol_funding_squeeze_gated"]["full_785d_sharpe"]), 2.19, places=2)
        self.assertAlmostEqual(float(rows["inv_vol_funding_gated_voltarget_15pct"]["full_785d_max_dd_pct"]), -8.55, places=2)
        self.assertAlmostEqual(float(rows["inv_vol_funding_gated_voltarget_15pct"]["full_785d_calmar"]), 4.33, places=2)

    def test_latest_2026_10_backtest_update(self) -> None:
        comp_path = ROOT / "reports" / "latest_oos_2026_10_composite.csv"
        with comp_path.open("r", encoding="utf-8") as fh:
            comp = {r["strategy"]: r for r in csv.DictReader(fh)}
        self.assertAlmostEqual(float(comp["inv_vol_funding_gated_voltarget_15pct"]["full_835d_sharpe"]), 1.69, places=2)
        self.assertAlmostEqual(float(comp["inv_vol_funding_gated_voltarget_15pct"]["since_sep28_22d_pct"]), 1.59, places=2)
        self.assertAlmostEqual(float(comp["static_equal_weight"]["since_sep28_22d_pct"]), 0.51, places=2)

        postpub_path = ROOT / "data" / "results" / "report_postpub_2026-10.csv"
        with postpub_path.open("r", encoding="utf-8") as fh:
            rows = {(r["group"], r["strategy"]): r for r in csv.DictReader(fh)}
        self.assertAlmostEqual(float(rows[("② 单因子多空 (理想口径, 单边5bps)", "散户多空比(大)")]["上次9月复核后(9/29起).cum"]), 0.06662, places=4)
        self.assertAlmostEqual(float(rows[("② 单因子多空 (理想口径, 单边5bps)", "资金费率(大)")]["九月.cum"]), 0.05098, places=4)
        self.assertAlmostEqual(float(rows[("④ 可交付产品 (周调仓·15bps)", "产品B·大币top8·对冲beta")]["九月.cum"]), 0.01474, places=4)

        stress_path = ROOT / "reports" / "live_readiness_2026_10" / "cost_and_delay_stress.csv"
        with stress_path.open("r", encoding="utf-8") as fh:
            stress = {(r["factor_or_strategy"], r["delay_mode"], float(r["cost_bps"])): r for r in csv.DictReader(fh)}
        self.assertAlmostEqual(float(stress[("散户多空比(大)", "1m_delay_00:01_24h", 5.0)]["full_sharpe"]), 1.43, places=2)
        self.assertGreater(float(stress[("散户多空比(大)", "0m_ideal_close", 5.0)]["breakeven_bps_full"]), 95.0)
        self.assertLess(float(stress[("持仓量变化(大)", "0m_ideal_close", 5.0)]["breakeven_bps_full"]), 10.0)

        hedge_path = ROOT / "reports" / "live_readiness_2026_10" / "product_b_realistic_hedges.csv"
        with hedge_path.open("r", encoding="utf-8") as fh:
            hedges = {(r["hedge_specification"], r["execution_timing"]): r for r in csv.DictReader(fh)}
        self.assertAlmostEqual(
            float(hedges[("real_ew24_perp_with_funding", "1m_00:01_delay")]["full_537d_sharpe"]),
            0.81,
            places=2,
        )

    def test_cli_summary_runs(self) -> None:
        self.assertEqual(main(["reproduce-sample"]), 0)


if __name__ == "__main__":
    unittest.main()
