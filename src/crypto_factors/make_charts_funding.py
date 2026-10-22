# -*- coding: utf-8 -*-
"""funding 时序择时图: 全市场 funding 作 risk-off 开关, 在 alt 熊市里削减亏损。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from funding_deep import build_panel, funding_timing

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.dirname(__file__)


def main():
    df = funding_timing(build_panel())
    df = df.dropna(subset=["agg"]).copy()
    eq_bh = (1 + df["bh"]).cumprod()
    eq_tm = (1 + df["timed"]).cumprod()

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8),
                                 gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True)
    a1.plot(eq_bh.index, eq_bh.values, color="#888", lw=1.6, ls="--", label="买入持有(全币等权)")
    a1.plot(eq_tm.index, eq_tm.values, color="#d62728", lw=2.2, label="funding 择时(多空)")
    a1.set_yscale("log"); a1.set_ylabel("净值(对数轴)")
    a1.set_title("全市场平均 funding 作 risk-off 开关 — alt 熊市里削减亏损 (大币池24, 滚动z-score防前视)",
                 fontsize=12.5)
    a1.legend(loc="lower left", fontsize=11); a1.grid(True, which="both", alpha=0.25)

    # 底panel: 全市场 funding (拥挤度) 本身
    a2.fill_between(df.index, df["agg"].values * 1e4, 0, color="#1f77b4", alpha=0.3)
    a2.axhline(0, color="#333", lw=0.8)
    a2.set_ylabel("全市场平均\nfunding (bp/日)"); a2.set_xlabel("日期"); a2.grid(True, alpha=0.25)

    fig.tight_layout()
    p = os.path.join(OUT, "funding_timing.png")
    fig.savefig(p, dpi=130)
    print("已保存:", p)


if __name__ == "__main__":
    main()
