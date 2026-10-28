"""Deterministic sample reproduction and real 785-day perpetual futures validation pipeline for CryptoFactors."""
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
from crypto_factors.costs import overlay_rebalance_costs

MONOREPO_CORE = PROJECT_ROOT.parent / "packages" / "oeq-core" / "src"
if MONOREPO_CORE.exists() and str(MONOREPO_CORE) not in sys.path:
    sys.path.insert(0, str(MONOREPO_CORE))

from oeq_core import (
    ClaimBoundary,
    EvidenceGrade,
    benjamini_hochberg_fdr,
    build_run_manifest,
    cross_sectional_rank_ic,
    inverse_volatility_weights,
    newey_west_tstat,
    quantile_long_short_spread,
    turnover_penalized_net_return,
    validate_claim_boundary,
    verify_manifest_checksums,
    volatility_target_weights,
)


def _normal_two_sided_pvalue(t_stat: float) -> float:
    return math.erfc(abs(float(t_stat)) / math.sqrt(2.0))


def _series_metrics(returns: list[float]) -> dict[str, float]:
    eq, peak, mdd = 1.0, 1.0, 0.0
    for r in returns:
        eq *= 1.0 + r
        if eq > peak:
            peak = eq
        dd = eq / peak - 1.0
        if dd < mdd:
            mdd = dd
    mu = sum(returns) / len(returns)
    sd = math.sqrt(sum((x - mu) ** 2 for x in returns) / max(1, len(returns) - 1))
    sharpe = (mu / sd) * math.sqrt(365.0)
    ann_ret = eq ** (365.0 / len(returns)) - 1.0
    calmar = ann_ret / abs(mdd) if mdd < 0 else 0.0
    return {
        "cum_ret_pct": round((eq - 1.0) * 100.0, 2),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(mdd * 100.0, 2),
        "calmar": round(calmar, 2),
        "nw_t": round(newey_west_tstat(returns, max_lag=5), 2),
    }


def evaluate_real_785d_perp_nav(
    root: Path = PROJECT_ROOT,
    rebalance_cost_bps_per_side: float = DEFAULT_COST_BPS_PER_SIDE,
) -> dict[str, object]:
    """Evaluate 785-day real Binance perpetual futures factor NAV series (2024-06-03 to 2026-07-27).

    The three sleeve NAVs in ``equity_perp.csv`` are taken as published. The overlays
    (inverse-vol weights, funding-squeeze gate, vol target up to 1.5x) are charged
    ``rebalance_cost_bps_per_side`` on the gross notional they trade when sleeve
    allocations change (see ``costs.overlay_rebalance_costs``). Before 2026-10-26 the
    overlays were charged nothing; ``rebalance_cost_bps_per_side=0`` reproduces the
    published table. Caveat: the sleeve NAVs themselves were produced by the
    pre-2026-10-26 long-short cost formula, which charged about half of the stated
    5 bp per side, so every figure here is still an upper bound.
    """
    nav_path = root / "data" / "results" / "equity_perp.csv"
    with nav_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    core3 = ("低价位振幅", "资金费率", "散户多空比")
    rets = {
        c: [float(rows[i][c]) / float(rows[i - 1][c]) - 1.0 for i in range(1, len(rows))]
        for c in core3
    }
    n = len(rets[core3[0]])
    ew_daily = [sum(rets[c][t] for c in core3) / 3.0 for t in range(n)]
    vw_ew = volatility_target_weights(ew_daily, target_vol=0.15, lookback=20, max_leverage=1.5, annualization=365.0)
    ew_vt_cost = overlay_rebalance_costs(
        [{c: vw_ew[t] / 3.0 for c in core3} for t in range(n)], rebalance_cost_bps_per_side
    )
    ew_vt = [ew_daily[t] * vw_ew[t] - ew_vt_cost[t] for t in range(n)]

    inv_gross: list[float] = []
    inv_weights: list[dict[str, float]] = []
    for t in range(n):
        if t < 20:
            w = {c: 1.0 / 3.0 for c in core3}
        else:
            hist = {c: rets[c][t - 20 : t] for c in core3}
            w = inverse_volatility_weights(hist, min_periods=20)
            if sum(rets["资金费率"][t - 5 : t]) < -0.015:
                w["资金费率"] *= 0.25
                s = sum(w.values())
                w = {c: w[c] / s for c in core3}
        inv_weights.append(w)
        inv_gross.append(sum(w[c] * rets[c][t] for c in core3))

    # The vol target is sized on the gross overlay series, exactly as published.
    vw_ig = volatility_target_weights(inv_gross, target_vol=0.15, lookback=20, max_leverage=1.5, annualization=365.0)
    inv_cost = overlay_rebalance_costs(inv_weights, rebalance_cost_bps_per_side)
    inv_vt_cost = overlay_rebalance_costs(
        [{c: vw_ig[t] * inv_weights[t][c] for c in core3} for t in range(n)], rebalance_cost_bps_per_side
    )
    inv_gated = [inv_gross[t] - inv_cost[t] for t in range(n)]
    inv_gated_vt = [inv_gross[t] * vw_ig[t] - inv_vt_cost[t] for t in range(n)]

    st_ew = _series_metrics(ew_daily)
    st_vt = _series_metrics(ew_vt)
    st_ig = _series_metrics(inv_gated)
    st_ig_vt = _series_metrics(inv_gated_vt)

    return {
        "real_perp_days": n,
        "real_785d_overlay_cost_bps_per_side": float(rebalance_cost_bps_per_side),
        "real_785d_static_sharpe": st_ew["sharpe"],
        "real_785d_static_max_dd_pct": st_ew["max_dd_pct"],
        "real_785d_voltarget_sharpe": st_vt["sharpe"],
        "real_785d_voltarget_cum_ret_pct": st_vt["cum_ret_pct"],
        "real_785d_voltarget_max_dd_pct": st_vt["max_dd_pct"],
        "real_785d_voltarget_calmar": st_vt["calmar"],
        "real_785d_inv_vol_gated_sharpe": st_ig["sharpe"],
        "real_785d_inv_vol_gated_cum_ret_pct": st_ig["cum_ret_pct"],
        "real_785d_inv_vol_gated_max_dd_pct": st_ig["max_dd_pct"],
        "real_785d_inv_vol_gated_calmar": st_ig["calmar"],
        "real_785d_inv_vol_vt_sharpe": st_ig_vt["sharpe"],
        "real_785d_inv_vol_vt_cum_ret_pct": st_ig_vt["cum_ret_pct"],
        "real_785d_inv_vol_vt_max_dd_pct": st_ig_vt["max_dd_pct"],
        "real_785d_inv_vol_vt_calmar": st_ig_vt["calmar"],
        "real_785d_inv_vol_vt_nw_t": st_ig_vt["nw_t"],
        "real_785d_overlay_cost_ann_pct": {
            "volatility_targeted_15pct": round(sum(ew_vt_cost) / n * 365.0 * 100.0, 2),
            "inv_vol_funding_squeeze_gated": round(sum(inv_cost) / n * 365.0 * 100.0, 2),
            "inv_vol_funding_gated_voltarget_15pct": round(sum(inv_vt_cost) / n * 365.0 * 100.0, 2),
        },
    }


def run_sample_pipeline(root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Execute cross-sectional factor evaluation, BH-FDR control, and real 785d perp NAV evaluation."""
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
    # Leverage changes of the vol target trade the whole long-short composite (gross 2 per unit).
    leverage_costs = overlay_rebalance_costs(
        [{"composite": w} for w in vol_weights], DEFAULT_COST_BPS_PER_SIDE
    )
    vol_targeted_returns = [r * w - c for r, w, c in zip(static_comp_returns, vol_weights, leverage_costs)]

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

    regime_vol_targeted = [r * w - c for r, w, c in zip(regime_comp_returns, vol_weights, leverage_costs)]
    real_perp_summary = evaluate_real_785d_perp_nav(root=root)

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
        **real_perp_summary,
    }
