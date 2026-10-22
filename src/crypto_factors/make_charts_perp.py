# -*- coding: utf-8 -*-
"""capstone 净值图: 新增的 perp 资金流维度 (funding/散户多空比) + 最优组合 vs 市场。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from backtest import metrics
from backtest_perp import build_panel, ls_series

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.dirname(__file__)


def dd(eq):
    return eq / eq.cummax() - 1.0


def main():
    panel = build_panel()
    nc, nd = panel["sym"].nunique(), panel["day"].nunique()

    fund = ls_series(panel, "FUND_f", -1, "big")[0]      # 资金费率(perp独有)
    gls  = ls_series(panel, "GLS_f", -1, "big")[0]       # 散户多空比(perp独有)
    rev  = ls_series(panel, "REV_hi_f", -1, "wide")[0]   # 价格反转
    big  = ls_series(panel, "OFIbig_f", +1, "big")[0]    # T2大单资金流
    amp  = ls_series(panel, "AMP_lo_f", -1, "wide")[0]   # 低价位振幅

    R = pd.concat({"fund":fund,"gls":gls,"rev":rev,"big":big,"amp":amp}, axis=1)
    comboA = R[["rev","big","fund"]].dropna().mean(axis=1)   # 价格×资金流×funding 等权

    mkt = panel.groupby("day")["fwd_ret"].mean()
    mkt.index = pd.to_datetime(mkt.index)

    curves = {
        "资金费率 funding★ (perp独有)": (1+fund.dropna()).cumprod(),
        "散户多空比★ (perp独有)":       (1+gls.dropna()).cumprod(),
        "价格×资金流×funding·等权":      (1+comboA).cumprod(),
        "市场等权买入持有":              (1+mkt.reindex(comboA.index).fillna(0)).cumprod(),
    }
    sty = {"资金费率 funding★ (perp独有)": dict(color="#d62728", lw=2.2),
           "散户多空比★ (perp独有)":       dict(color="#9467bd", lw=1.8),
           "价格×资金流×funding·等权":      dict(color="#2ca02c", lw=2.6),
           "市场等权买入持有":              dict(color="#888", lw=1.2, ls="--")}

    fig,(a1,a2)=plt.subplots(2,1,figsize=(12,8.5),
                             gridspec_kw={"height_ratios":[2.4,1]},sharex=True)
    for n,e in curves.items():
        a1.plot(e.index,e.values,label=n,**sty[n])
    a1.set_yscale("log"); a1.set_ylabel("净值(对数轴,起点=1)")
    a1.set_title(f"crypto perp 微观结构因子: funding/OI 新维度 + 约束组合层  "
                 f"(宽池{nc}币/大币池24 × {nd}天,五分位多空,5bps)",fontsize=12.5)
    a1.legend(loc="upper left",fontsize=11); a1.grid(True,which="both",alpha=0.25)

    for n in ["资金费率 funding★ (perp独有)","价格×资金流×funding·等权"]:
        d=dd(curves[n])
        a2.fill_between(d.index,d.values*100,0,alpha=0.18,color=sty[n]["color"])
        a2.plot(d.index,d.values*100,color=sty[n]["color"],lw=1.2,label=n)
    a2.set_ylabel("回撤(%)"); a2.set_xlabel("日期"); a2.grid(True,alpha=0.25)
    a2.legend(loc="lower left",fontsize=9)

    lines=[]
    for nm,s in [("资金费率",fund),("散户多空比",gls),("组合(价格×资金流×funding)",comboA)]:
        m=metrics(s.dropna().values)
        lines.append(f"{nm}: 净值{1+m['tot']:.2f} 夏普{m['sharpe']:.2f} 回撤{m['mdd']*100:.0f}%")
    fig.text(0.5,0.005,"   |   ".join(lines),ha="center",fontsize=10,color="#333")
    fig.tight_layout(rect=[0,0.03,1,1])
    p=os.path.join(OUT,"equity_perp.png")
    fig.savefig(p,dpi=130); print("已保存:",p); print("\n".join(lines))


if __name__=="__main__":
    main()
