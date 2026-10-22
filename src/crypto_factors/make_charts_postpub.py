# -*- coding: utf-8 -*-
"""Post-publication chart: cumulative return since 2026-06-01 + monthly returns, June-September 2026.

Same strategy definitions as report_postpub.py (nothing re-fitted). English labels, for README.md.

Usage: python make_charts_postpub.py  ->  equity_postpub.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest_perp import build_panel
from report_postpub import all_series, MONTHS

OUT = os.path.dirname(os.path.abspath(__file__))
START = pd.Timestamp("2026-06-01", tz="UTC")

SURF = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e3e2de"

# (group, strategy) -> (English label, colour)
PICK = [
    ("④ 可交付产品 (周调仓·15bps)", "产品B·大币top8·对冲beta", "Composite B, market-neutral", "#0b0b0b"),
    ("② 单因子多空 (理想口径, 单边5bps)", "资金费率(大)", "Funding rate (L/S)", "#eb6834"),
    ("② 单因子多空 (理想口径, 单边5bps)", "散户多空比(大)", "Retail long/short (L/S)", "#1baf7a"),
    ("② 单因子多空 (理想口径, 单边5bps)", "低价位振幅(宽)", "Low-price range (L/S)", "#2a78d6"),
    ("② 单因子多空 (理想口径, 单边5bps)", "持仓量变化(大)", "Open-interest change (L/S)", "#eda100"),
]


def main():
    panel = build_panel()
    groups = all_series(panel)
    series = []
    for g, s, lbl, c in PICK:
        x = groups[g][s].dropna().sort_index()
        series.append((lbl, c, x[x.index >= START]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.4), facecolor=SURF,
                                   gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.22))
    for ax in (ax1, ax2):
        ax.set_facecolor(SURF)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)

    for lbl, c, x in series:
        eq = (1 + x).cumprod() - 1
        lw = 2.4 if lbl.startswith("Composite") else 1.5
        ax1.plot(eq.index, eq.values * 100, color=c, lw=lw, label=lbl, zorder=3)
        ax1.annotate(f"{eq.iloc[-1] * 100:+.1f}%", (eq.index[-1], eq.iloc[-1] * 100),
                     xytext=(4, 0), textcoords="offset points", color=c, fontsize=8, va="center")
    ax1.axhline(0, color=INK3, lw=0.8)
    ax1.set_title("Cumulative return since 1 June 2026 (%)", color=INK, fontsize=11, loc="left")
    ax1.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    ax1.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))

    w = 0.8 / len(series)
    xs = np.arange(len(MONTHS))
    for i, (lbl, c, x) in enumerate(series):
        vals = []
        for m in MONTHS:
            sm = x[x.index.strftime("%Y-%m") == m]
            vals.append(np.prod(1 + sm.values) - 1 if len(sm) >= 5 else np.nan)
        ax2.bar(xs - 0.4 + w * (i + 0.5), np.array(vals) * 100, w * 0.92, color=c, zorder=3)
    ax2.axhline(0, color=INK3, lw=0.8)
    ax2.set_xticks(xs, [pd.Timestamp(m + "-01").strftime("%b %Y") for m in MONTHS])
    ax2.set_title("Monthly return (%)", color=INK, fontsize=11, loc="left")

    fund_end = panel.dropna(subset=["FUND_f"])["day"].max().date()
    fig.text(0.01, 0.005,
             f"Long-short legs: daily quintile spreads, 5 bp per side. Composite B: top 8 large coins, weekly "
             f"rebalance, 15 bp, spot basket minus large-coin index. Funding data end {fund_end}, so funding "
             f"and the composite stop there.", color=INK3, fontsize=7.5)
    fn = os.path.join(OUT, "equity_postpub.png")
    fig.savefig(fn, dpi=150, bbox_inches="tight", facecolor=SURF)
    print("saved", fn)


if __name__ == "__main__":
    main()
