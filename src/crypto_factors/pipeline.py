"""Deterministic sample reproduction and oeq-core verification pipeline for CryptoFactors."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
import sys

from crypto_factors.config import (
    DEFAULT_COST_BPS_PER_SIDE,
    DEFAULT_QUANTILES,
    DEFAULT_TARGET_VOL,
    PROJECT_ROOT,
)

MONOREPO_CORE = PROJECT_ROOT.parent / "packages" / "oeq-core" / "src"
if MONOREPO_CORE.exists() and str(MONOREPO_CORE) not in sys.path:
    sys.path.insert(0, str(MONOREPO_CORE))

from oeq_core import (
    ClaimBoundary,
    EvidenceGrade,
    benjamini_hochberg_fdr,
    build_run_manifest,
    cross_sectional_rank_ic,
    newey_west_tstat,
    quantile_long_short_spread,
    turnover_penalized_net_return,
    validate_claim_boundary,
    verify_manifest_checksums,
    volatility_target_weights,
)


def _normal_two_sided_pvalue(t_stat: float) -> float:
    return math.erfc(abs(float(t_stat)) / math.sqrt(2.0))


def run_sample_pipeline(root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Execute cross-sectional factor evaluation, BH-FDR control, and vol-targeted composite on sample fixture."""
    fixture_path = root / "data" / "fixtures" / "sample_perp_panel.csv"
    transfer_path = root / "reports" / "factor_transfer_summary.csv"
    postpub_path = root / "reports" / "postpub_composite_summary.csv"
    regime_path = root / "reports" / "regime_conditioned_composite_summary.csv"

    with fixture_path.open("r", encoding="utf-8") as fh:
        sample_rows = list(csv.DictReader(fh))
    with transfer_path.open("r", encoding="utf-8") as fh:
        factors = list(csv.DictReader(fh))
    with postpub_path.open("r", encoding="utf-8") as fh:
        windows = list(csv.DictReader(fh))
    with regime_path.open("r", encoding="utf-8") as fh:
        regime_rows = list(csv.DictReader(fh))

    by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sample_rows:
        by_date[row["date"]].append(row)

    dates_sorted = sorted(by_date.keys())
    ic_series: dict[str, list[float]] = defaultdict(list)
    static_comp_returns: list[float] = []
    regime_comp_returns: list[float] = []

    for dt in dates_sorted:
        cross = by_date[dt]
        if len(cross) < DEFAULT_QUANTILES:
            continue
        fwd = [float(r["fwd_return_1d"]) for r in cross]
        # Contrarian orientation for funding_rate and retail_ls_ratio; positive for low_price_amp
        sig_funding = [-float(r["funding_rate"]) for r in cross]
        sig_retail = [-float(r["retail_ls_ratio"]) for r in cross]
        sig_amp = [float(r["low_price_amp"]) for r in cross]
        sig_mom = [float(r["mom_14d"]) for r in cross]

        for name, sig in (
            ("funding_contrarian", sig_funding),
            ("retail_ls_contrarian", sig_retail),
            ("low_price_amp", sig_amp),
            ("mom_14d", sig_mom),
        ):
            ic_series[name].append(cross_sectional_rank_ic(sig, fwd))

        s_fund = quantile_long_short_spread(sig_funding, fwd, n_quantiles=DEFAULT_QUANTILES)
        s_ret = quantile_long_short_spread(sig_retail, fwd, n_quantiles=DEFAULT_QUANTILES)
        s_amp = quantile_long_short_spread(sig_amp, fwd, n_quantiles=DEFAULT_QUANTILES)
        gross_comp = (s_fund + s_ret + s_amp) / 3.0
        net_comp = turnover_penalized_net_return(
            gross_comp,
            turnover=0.4,
            cost_bps=2.0 * DEFAULT_COST_BPS_PER_SIDE,
        )
        static_comp_returns.append(net_comp)

        mean_fund = sum(sig_funding) / len(sig_funding)
        fund_disp = math.sqrt(sum((x - mean_fund) ** 2 for x in sig_funding) / len(sig_funding))
        mean_mom = sum(sig_mom) / len(sig_mom)
        w_fund = 0.20 if (fund_disp > 0.00025 and mean_mom > 0.0) else (1.0 / 3.0)
        w_other = (1.0 - w_fund) / 2.0
        gross_regime = w_fund * s_fund + w_other * s_ret + w_other * s_amp
        regime_comp_returns.append(
            turnover_penalized_net_return(gross_regime, turnover=0.35, cost_bps=2.0 * DEFAULT_COST_BPS_PER_SIDE)
        )

    factor_names = list(ic_series.keys())
    t_stats = [newey_west_tstat(ic_series[k], max_lag=2) for k in factor_names]
    p_vals = [_normal_two_sided_pvalue(t) for t in t_stats]
    q_vals = benjamini_hochberg_fdr(p_vals)

    vol_weights = volatility_target_weights(
        static_comp_returns,
        target_vol=DEFAULT_TARGET_VOL,
        lookback=4,
        max_leverage=1.5,
        annualization=365.0,
    )
    vol_targeted_returns = [r * w for r, w in zip(static_comp_returns, vol_weights)]

    boundary = ClaimBoundary(
        project_slug="crypto-factors",
        grade=EvidenceGrade.BACKTEST_ASSOCIATION,
        allows_causal_language=False,
        allows_live_profitability_claim=False,
    )
    violations = validate_claim_boundary(boundary)
    if violations:
        raise ValueError(f"Claim boundary violation: {violations}")

    manifest = build_run_manifest(
        project_slug="crypto-factors",
        root=root,
        files=[fixture_path, transfer_path, postpub_path, regime_path],
        evidence_grade=boundary.grade.value,
    )
    checksum_errors = verify_manifest_checksums(root, manifest)
    if checksum_errors:
        raise ValueError(f"Manifest checksum failure: {checksum_errors}")

    sample_fdr_table = {
        name: {
            "mean_ic": round(sum(ic_series[name]) / len(ic_series[name]), 4),
            "nw_t": round(t_stats[i], 2),
            "bh_fdr_q": round(q_vals[i], 4),
        }
        for i, name in enumerate(factor_names)
    }

    regime_vol_targeted = [r * w for r, w in zip(regime_comp_returns, vol_weights)]

    return {
        "project": "crypto-factors",
        "evidence_grade": "backtest and post-publication out-of-sample backtest (association)",
        "claim_boundary_grade": boundary.grade.value,
        "factors_tested": len(factors),
        "robust_factors": [r["factor"] for r in factors if "strongest" in r["verdict"] or "robust" in r["verdict"]],
        "postpub_92d_return_pct": float(windows[-1]["market_neutral_return_pct"]),
        "postpub_92d_max_drawdown_pct": float(windows[-1]["max_drawdown_pct"]),
        "regime_variants_evaluated": len(regime_rows),
        "regime_conditioned_postpub_92d_pct": float(regime_rows[-1]["postpub_92d_return_pct"]),
        "sample_fixture_dates": len(dates_sorted),
        "sample_factor_fdr": sample_fdr_table,
        "sample_static_composite_mean_net": round(sum(static_comp_returns) / max(1, len(static_comp_returns)), 5),
        "sample_vol_targeted_mean_net": round(sum(vol_targeted_returns) / max(1, len(vol_targeted_returns)), 5),
        "sample_regime_conditioned_mean_net": round(sum(regime_vol_targeted) / max(1, len(regime_vol_targeted)), 5),
        "verified_files": manifest["file_count"],
    }
