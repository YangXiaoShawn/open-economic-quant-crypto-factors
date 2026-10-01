# -*- coding: utf-8 -*-
"""三窗口对比图: 年初至今 / 七月初至今 / 近15天 净值 + 区间收益条形。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest import backtest_factor, metrics
from backtest_perp import build_panel, pool_panel, ls_series, BIG_POOL, FACTORS
from deploy_spot import zc, backtest_basket, bench

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.dirname(os.path.abspath(__file__))
YTD = pd.Timestamp("2026-01-01", tz="UTC")
JUL = pd.Timestamp("2026-07-01", tz="UTC")

SURF = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GRID = "#e3e2de"
WCOLOR = ["#2a78d6", "#eb6834", "#1baf7a"]


def utc(s):
    i = pd.to_datetime(pd.Index(s.index))
    i = i.tz_localize("UTC") if i.tz is None else i.tz_convert("UTC")
    return pd.Series(s.values, index=i).sort_index()


def win_slice(s, last):
    cut15 = last - pd.Timedelta(days=14)
    return {
        "年初至今": s[s.index >= YTD],
        "七月初至今": s[s.index >= JUL],
        "近15天": s[s.index >= cut15],
    }


def main():
    panel = build_panel()
    last = pd.Timestamp(panel["day"].max())
    last = last.tz_localize("UTC") if last.tzinfo is None else last.tz_convert("UTC")

    rets = {}
    for (col, d, pool), nm, c in [
            (("AMP_lo_f", -1, "wide"), "低价位振幅", SERIES[0]),
            (("FUND_f", -1, "big"), "资金费率", SERIES[1]),
            (("GLS_f", -1, "big"), "散户多空比", SERIES[2])]:
        rets[nm] = (utc(ls_series(panel, col, d, pool)[0].dropna()), c)

    df = pool_panel(panel, "big").copy()
    df["SCORE"] = sum(zc(df, c, sg) for c, sg in
                      [("AMP_lo_f", -1), ("FUND_f", -1), ("GLS_f", -1)])
    df = df.dropna(subset=["SCORE"])
    r = utc(backtest_basket(df, "SCORE", 8, rebal=7, cost_bps=15)[0])
    bmk = utc(bench(df))
    rets["产品B 对冲beta"] = ((r - bmk.reindex(r.index)).dropna(), SERIES[3])
    mkt = utc(panel.dropna(subset=["fwd_ret"]).groupby("day")["fwd_ret"].mean())

    wnames = ["年初至今", "七月初至今", "近15天"]
    fig = plt.figure(figsize=(13.6, 11.2), facecolor=SURF)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.05], hspace=0.38, wspace=0.18)

    for i, wname in enumerate(wnames):
        ax = fig.add_subplot(gs[0, i], facecolor=SURF)
        for nm, (s, c) in rets.items():
            ws = win_slice(s, last)[wname]
            if len(ws) < 3:
                continue
            eq = (1 + ws).cumprod()
            ax.plot(eq.index, eq.values, color=c, lw=1.8, zorder=3, solid_capstyle="round")
            ax.annotate(f"{eq.iloc[-1]:.2f}x", (eq.index[-1], eq.iloc[-1]),
                        color=INK, fontsize=8, va="center")
        wm = win_slice(mkt, last)[wname]
        if len(wm) >= 3:
            eqm = (1 + wm).cumprod()
            ax.plot(eqm.index, eqm.values, color=INK3, lw=1.2, ls="--", zorder=2)
        ax.axhline(1.0, color=INK3, lw=0.9, zorder=1)
        ax.set_title(wname, color=INK, fontsize=12, loc="left", pad=8)
        ax.grid(True, color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}x"))

    axb = fig.add_subplot(gs[1, :], facecolor=SURF)
    names = list(rets.keys()) + ["市场等权"]
    series_all = {**{k: v[0] for k, v in rets.items()}, "市场等权": mkt}
    x = np.arange(len(names))
    width = 0.24
    for j, wname in enumerate(wnames):
        vals = []
        for nm in names:
            ws = win_slice(series_all[nm], last)[wname]
            m = metrics(ws.values) if len(ws) >= 5 else None
            vals.append(m["tot"] * 100 if m else np.nan)
        bars = axb.bar(x + (j - 1) * width, vals, width, color=WCOLOR[j],
                       label=wname, zorder=3, edgecolor=SURF, linewidth=0.6)
        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            axb.text(b.get_x() + b.get_width() / 2, b.get_height() + (0.4 if v >= 0 else -1.2),
                     f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top",
                     fontsize=8, color=INK)
    axb.axhline(0, color=INK3, lw=1.1, zorder=2)
    axb.set_xticks(x)
    axb.set_xticklabels(names, fontsize=10, color=INK)
    axb.set_ylabel("区间累计收益 (%)", color=INK2, fontsize=10)
    axb.set_title("区间收益对比  (短窗年化勿外推; funding 末日可能早于价量)",
                  color=INK, fontsize=12, loc="left", pad=10)
    axb.grid(True, axis="y", color=GRID, lw=0.7)
    axb.set_axisbelow(True)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    axb.spines["left"].set_color(GRID)
    axb.spines["bottom"].set_color(GRID)
    axb.tick_params(colors=INK2, labelsize=9)
    lg = axb.legend(loc="upper right", frameon=True, fontsize=10, facecolor=SURF, edgecolor=GRID)
    for t in lg.get_texts():
        t.set_color(INK)

    fig.suptitle(f"crypto 微观结构因子 · 三窗口对比  (数据至 {last.date()})",
                 color=INK, fontsize=15, x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.006,
             "口径: 横截面五分位多空, 日调仓, 单边5bps; 产品B=大币top8周调15bps − 指数空头。"
             "策略未重新拟合, 全样本日收益再切片。",
             color=INK2, fontsize=9, ha="left")
    fig.tight_layout(rect=[0, 0.025, 1, 0.97])
    p = os.path.join(OUT, "equity_windows.png")
    fig.savefig(p, dpi=130, facecolor=SURF)
    print("已保存:", p)


if __name__ == "__main__":
    main()
