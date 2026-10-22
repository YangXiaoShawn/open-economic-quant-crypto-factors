# -*- coding: utf-8 -*-
"""
生成最优单因子(理想反转)与组合的净值图 + 回撤图。
读 data_binance (60币×730天), 复用 backtest 的多空逻辑。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from backtest import backtest_factor, metrics
from backtest_all32 import build_panel

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.dirname(__file__)


def ls_series(panel, col, d):
    bt = backtest_factor(panel, col, d)
    return pd.Series(bt["ls"].values, index=pd.to_datetime(bt["day"].values))


def dd(eq):
    return eq / eq.cummax() - 1.0


def main():
    panel = build_panel()
    nc, nd = panel["sym"].nunique(), panel["day"].nunique()

    # 各因子多空收益流
    rev = ls_series(panel, "REV_hi_f", -1)          # 最优单因子
    big = ls_series(panel, "OFIbig_f", +1)          # 大单资金流
    ofi = ls_series(panel, "OFI_f",   +1)
    amp = ls_series(panel, "AMP_lo_f", -1)
    mom = ls_series(panel, "MOM_f",   +1)
    df = pd.concat({"rev": rev, "big": big, "ofi": ofi, "amp": amp, "mom": mom}, axis=1).dropna()

    combo2 = df[["rev", "big"]].mean(axis=1)        # 反转+资金流
    combo5 = df[["rev", "big", "ofi", "amp", "mom"]].mean(axis=1)  # 5因子等权

    # 市场基准: 全币等权买入持有 (次日收益均值)
    mkt = panel.groupby("day")["fwd_ret"].mean()
    mkt.index = pd.to_datetime(mkt.index)
    mkt = mkt.reindex(df.index).fillna(0)

    curves = {
        "理想反转(最优单因子)": (1 + df["rev"]).cumprod(),
        "反转+资金流·等权":     (1 + combo2).cumprod(),
        "5因子·等权":          (1 + combo5).cumprod(),
        "市场等权买入持有":      (1 + mkt).cumprod(),
    }
    styles = {"理想反转(最优单因子)": dict(color="#1f77b4", lw=2),
              "反转+资金流·等权":     dict(color="#2ca02c", lw=2),
              "5因子·等权":          dict(color="#d62728", lw=2.4),
              "市场等权买入持有":      dict(color="#888888", lw=1.2, ls="--")}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8.5),
                                   gridspec_kw={"height_ratios": [2.4, 1]}, sharex=True)
    for name, eq in curves.items():
        ax1.plot(eq.index, eq.values, label=name, **styles[name])
    ax1.set_yscale("log")
    ax1.set_ylabel("净值 (对数轴, 起点=1)")
    ax1.set_title(f"开源微观结构因子 crypto 迁移 — 净值曲线  "
                  f"({nc}币 × {nd}天, 五分位多空, 日调仓, 单边5bps)", fontsize=13)
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(True, which="both", alpha=0.25)

    # 回撤图 (策略, 不含市场)
    for name in ["理想反转(最优单因子)", "反转+资金流·等权", "5因子·等权"]:
        d = dd(curves[name])
        ax2.fill_between(d.index, d.values * 100, 0, alpha=0.18, color=styles[name]["color"])
        ax2.plot(d.index, d.values * 100, color=styles[name]["color"], lw=1.2, label=name)
    ax2.set_ylabel("回撤 (%)")
    ax2.set_xlabel("日期")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower left", fontsize=9, ncol=3)

    # 指标注脚
    lines = []
    for name, key, dseries in [("理想反转", "rev", df["rev"]),
                               ("反转+资金流", None, combo2),
                               ("5因子等权", None, combo5)]:
        m = metrics((df[key] if key else dseries).values)
        lines.append(f"{name}: 净值{1+m['tot']:.2f} 夏普{m['sharpe']:.2f} 回撤{m['mdd']*100:.0f}%")
    fig.text(0.5, 0.005, "   |   ".join(lines), ha="center", fontsize=10, color="#333")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    p1 = os.path.join(OUT, "equity_curves.png")
    fig.savefig(p1, dpi=130)
    print("已保存:", p1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
