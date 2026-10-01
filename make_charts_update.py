# -*- coding: utf-8 -*-
"""
本次更新专用图: 长期净值 (标出新增窗口) + 各因子夏普 旧窗→新窗 变化。

配色用 dataviz 参考调色板的固定槽位顺序 (blue/orange/aqua/yellow), 折线属"相邻配对"清单,
该顺序已在参考文件中通过 CVD/对比度校验; 另加末端直接标注作二次编码 (黄/青在浅色底
对比度 <3:1, 按 relief 规则必须有可见标签)。
"""
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
SPLIT = pd.Timestamp("2026-06-01", tz="UTC")

SURF = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]     # 槽 1-4, 固定顺序
GRID = "#e3e2de"


def utc(s):
    """ls_series/backtest_basket 返回的 index 有的带时区有的不带, 统一成 UTC 再比较。"""
    i = pd.to_datetime(pd.Index(s.index))
    i = i.tz_localize("UTC") if i.tz is None else i.tz_convert("UTC")
    return pd.Series(s.values, index=i).sort_index()


def main():
    panel = build_panel()

    # ---------- 面板 A 用的四条策略 + 市场基准 ----------
    curves, rets = {}, {}
    for (col, d, pool), nm, c in [
            (("AMP_lo_f", -1, "wide"), "低价位振幅 多空", SERIES[0]),
            (("FUND_f", -1, "big"), "资金费率 多空", SERIES[1]),
            (("GLS_f", -1, "big"), "散户多空比 多空", SERIES[2])]:
        rets[nm] = (utc(ls_series(panel, col, d, pool)[0].dropna()), c)

    df = pool_panel(panel, "big").copy()
    df["SCORE"] = sum(zc(df, c, sg) for c, sg in
                      [("AMP_lo_f", -1), ("FUND_f", -1), ("GLS_f", -1)])
    df = df.dropna(subset=["SCORE"])
    r = utc(backtest_basket(df, "SCORE", 8, rebal=7, cost_bps=15)[0])
    bmk = utc(bench(df))
    rets["产品B 对冲beta (纯alpha)"] = ((r - bmk.reindex(r.index)).dropna(), SERIES[3])

    mkt = utc(panel.dropna(subset=["fwd_ret"]).groupby("day")["fwd_ret"].mean())

    fig = plt.figure(figsize=(13.5, 10.5), facecolor=SURF)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 1], hspace=0.28)

    # ================= 面板 A: 长期净值 =================
    a1 = fig.add_subplot(gs[0], facecolor=SURF)
    a1.axvspan(SPLIT, max(s.index.max() for s, _ in rets.values()),
               color="#2a78d6", alpha=0.07, zorder=0)
    for nm, (s, c) in rets.items():
        eq = (1 + s).cumprod()
        a1.plot(eq.index, eq.values, color=c, lw=2.0, zorder=3, solid_capstyle="round")
        a1.annotate(f" {nm}  {eq.iloc[-1]:.2f}x", (eq.index[-1], eq.iloc[-1]),
                    color=INK, fontsize=9.5, va="center", zorder=5)
    eqm = (1 + mkt).cumprod()
    a1.plot(eqm.index, eqm.values, color=INK3, lw=1.4, ls="--", zorder=2)
    a1.annotate(f" 市场等权  {eqm.iloc[-1]:.2f}x", (eqm.index[-1], eqm.iloc[-1]),
                color=INK2, fontsize=9.5, va="center")
    a1.axhline(1.0, color=INK3, lw=1.1, zorder=1)
    a1.set_yscale("log")
    ticks = [0.15, 0.25, 0.4, 0.6, 1.0, 1.5, 2.0, 3.0]
    a1.set_yticks(ticks)
    a1.set_yticklabels([f"{t:g}x" for t in ticks])
    a1.minorticks_off()
    a1.set_ylabel("净值 (对数轴, 起点 = 1)", color=INK2, fontsize=10)
    a1.set_title("① 长期净值曲线 —— 阴影为本次新增窗口 (2026-06-01 起)",
                 color=INK, fontsize=13, loc="left", pad=12)
    a1.grid(True, which="major", color=GRID, lw=0.8, alpha=0.9)
    a1.set_axisbelow(True)
    for sp in ("top", "right"):
        a1.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a1.spines[sp].set_color(GRID)
    a1.tick_params(colors=INK2, labelsize=9)
    a1.set_xlim(eqm.index.min(), eqm.index.max() + pd.Timedelta(days=155))

    # ================= 面板 B: 夏普 旧窗 -> 新窗 =================
    rows = []
    for col, name, d, pool, typ in FACTORS:
        if col not in panel.columns:
            continue
        bt = backtest_factor(pool_panel(panel, pool), col, d)
        if len(bt) < 30:
            continue
        s = utc(pd.Series(bt["ls"].values, index=bt["day"].values))
        o, n = s[s.index < SPLIT], s[s.index >= SPLIT]
        mo, mn = metrics(o.values), metrics(n.values)
        if not mo or not mn:
            continue
        rows.append((name, mo["sharpe"], mn["sharpe"], len(n)))
    rows.sort(key=lambda x: x[1])

    a2 = fig.add_subplot(gs[1], facecolor=SURF)
    y = np.arange(len(rows))
    for i, (nm, so, sn, nd) in enumerate(rows):
        a2.plot([so, sn], [i, i], color=GRID, lw=2.4, zorder=1, solid_capstyle="round")
    a2.scatter([r[1] for r in rows], y, s=95, color=SERIES[0], zorder=3,
               edgecolor=SURF, linewidth=2, label="旧窗 (~2026-05-31)")
    a2.scatter([r[2] for r in rows], y, s=95, color=SERIES[1], zorder=3,
               edgecolor=SURF, linewidth=2, label="新增窗口 (2026-06 起)")
    a2.axvline(0, color=INK3, lw=1.2, zorder=2)
    a2.set_yticks(y)
    a2.set_yticklabels([f"{r[0]}  ({r[3]}天)" for r in rows], fontsize=9.5, color=INK)
    a2.set_xlabel("夏普比率 (横截面五分位多空)", color=INK2, fontsize=10)
    a2.set_title("② 每个因子的夏普: 旧窗 → 新增窗口  (新窗仅 1~2 个月, 噪声大, 只看方向)",
                 color=INK, fontsize=13, loc="left", pad=12)
    a2.grid(True, axis="x", color=GRID, lw=0.8)
    a2.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        a2.spines[sp].set_visible(False)
    a2.spines["bottom"].set_color(GRID)
    a2.tick_params(colors=INK2, labelsize=9)
    lg = a2.legend(loc="lower right", frameon=True, fontsize=10, facecolor=SURF,
                   edgecolor=GRID)
    for t in lg.get_texts():
        t.set_color(INK)

    fig.suptitle("crypto 微观结构因子 · 数据更新至 2026-07-28 后的分期表现",
                 color=INK, fontsize=15, x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.005,
             "口径: 横截面五分位多空, 日调仓, 单边5bps, 年化按365天。策略未重新拟合 —— "
             "全样本算一遍再按日期切片, 故新增窗口是真样本外。"
             "funding 类因子止于 2026-06-30 (dump 只有月档)。",
             color=INK2, fontsize=9, ha="left")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    p = os.path.join(OUT, "equity_update.png")
    fig.savefig(p, dpi=130, facecolor=SURF)
    print("已保存:", p)
    for nm, so, sn, nd in rows:
        print(f"  {nm:10s} 旧 {so:+.2f} -> 新 {sn:+.2f}  ({nd}天)")


if __name__ == "__main__":
    main()
