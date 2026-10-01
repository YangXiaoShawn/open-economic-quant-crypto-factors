# -*- coding: utf-8 -*-
"""交付产品 B 三线图: 池基准(亏) vs 纯现货多头(少亏) vs 多头+perp对冲beta(纯alpha,赚)。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from backtest import metrics
from backtest_perp import build_panel, BIG_POOL
from deploy_spot import zc, backtest_basket, bench

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.dirname(__file__)


def dd(eq):
    return eq / eq.cummax() - 1.0


def main():
    panel = build_panel()
    df = panel[panel["sym"].isin(BIG_POOL)].copy()
    df["SCORE"] = (zc(df, "AMP_lo_f", -1) + zc(df, "FUND_f", -1) + zc(df, "GLS_f", -1))
    df = df.dropna(subset=["SCORE"])
    bm = bench(df)
    prod, _ = backtest_basket(df, "SCORE", top_k=8, rebal=7, cost_bps=15)
    hedged = (prod - bm.reindex(prod.index)).dropna()

    curves = {
        "池基准·买入持有 (大币等权)": (1+bm.reindex(prod.index).fillna(0)).cumprod(),
        "现货多头产品 (top8·周调)":   (1+prod).cumprod(),
        "多头+perp对冲beta (纯alpha)": (1+hedged).cumprod(),
    }
    sty = {"池基准·买入持有 (大币等权)": dict(color="#888", lw=1.4, ls="--"),
           "现货多头产品 (top8·周调)":   dict(color="#1f77b4", lw=1.8),
           "多头+perp对冲beta (纯alpha)": dict(color="#2ca02c", lw=2.6)}

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8.3),
                                 gridspec_kw={"height_ratios": [2.3, 1]}, sharex=True)
    for n, e in curves.items():
        a1.plot(e.index, e.values, label=n, **sty[n])
    a1.set_yscale("log"); a1.set_ylabel("净值(对数轴,起点=1)")
    a1.set_title("可交付产品: 现货多头 vs 对冲beta纯alpha  (大币池24, top8等权, 周调, 15bps, 510天)",
                 fontsize=12.5)
    a1.legend(loc="lower left", fontsize=11); a1.grid(True, which="both", alpha=0.25)

    for n in ["现货多头产品 (top8·周调)", "多头+perp对冲beta (纯alpha)"]:
        d = dd(curves[n])
        a2.fill_between(d.index, d.values*100, 0, alpha=0.18, color=sty[n]["color"])
        a2.plot(d.index, d.values*100, color=sty[n]["color"], lw=1.2, label=n)
    a2.set_ylabel("回撤(%)"); a2.set_xlabel("日期"); a2.grid(True, alpha=0.25)
    a2.legend(loc="lower left", fontsize=9)

    mh = metrics(hedged.values)
    fig.text(0.5, 0.005, f"对冲beta纯alpha: 净值{1+mh['tot']:.2f}  夏普{mh['sharpe']:.2f}  "
             f"回撤{mh['mdd']*100:.0f}%  日胜率{mh['win']*100:.0f}%  周换手~5%",
             ha="center", fontsize=10, color="#333")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    p = os.path.join(OUT, "equity_deploy.png")
    fig.savefig(p, dpi=130); print("已保存:", p)


if __name__ == "__main__":
    main()
