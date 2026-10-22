# -*- coding: utf-8 -*-
"""
收口: 把"去水分后唯一能现货多头直接吃"的三个因子做成可交付产品 —— 纯现货、不做空、定期调仓。

存活的 long-only 超额 (见 backtest_real.py): 低价位振幅(+1.36) / 资金费率(+0.71) / 散户多空比(+0.86)。
合成一个横截面 z-score 复合分, 每期持最高分 top-k 等权篮子, 定期调仓; 再做工程化压测:
  - 冲击/手续费成本网格 (5/15/30 bps)
  - 调仓频率 (日/周/双周) 控换手
  - 持仓数下限 (top-k≥8, 避免 2~3 名集中)
两套: A 宽池105×730天(低价位振幅+资金费率, 两者宽池皆活); B 大币池24×510天(再叠散户多空比, OI只在大币)。
"""
import os
import numpy as np
import pandas as pd
from backtest import metrics
from backtest_perp import build_panel, BIG_POOL

ROOT = os.path.dirname(__file__)


def zc(panel, col, sign):
    return panel.groupby("day")[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9)) * sign


def backtest_basket(df, score, top_k, rebal, cost_bps):
    """纯多头 top-k 等权篮子, 每 rebal 天调仓一次, 换手计 cost。返回(日收益, 换手序列)。"""
    df = df.dropna(subset=["fwd_ret"]).copy()
    days = sorted(df["day"].unique())
    held, w = set(), {}
    rets, tos = [], []
    by_day = {d: g for d, g in df.groupby("day")}
    for i, day in enumerate(days):
        g = by_day[day]
        if i % rebal == 0:
            gs = g.dropna(subset=[score])
            if gs["sym"].nunique() >= top_k:
                target = set(gs.nlargest(top_k, score)["sym"])
                to = len(target ^ held) / max(len(target), 1)
                held = target
            else:
                to = 0.0
        else:
            to = 0.0
        sub = g[g["sym"].isin(held)]
        r = sub["fwd_ret"].mean() if len(sub) else 0.0
        rets.append((0.0 if pd.isna(r) else r) - to * cost_bps / 1e4)
        tos.append(to)
    return pd.Series(rets, index=pd.to_datetime(days)), np.array(tos)


def bench(df):
    m = df.dropna(subset=["fwd_ret"]).groupby("day")["fwd_ret"].mean()
    return pd.Series(m.values, index=pd.to_datetime(m.index))


def show(label, r, to=None):
    m = metrics(r.values if hasattr(r, "values") else r)
    extra = "" if to is None else f"  日均换手{np.mean(to)*100:3.0f}% 年调仓成本~{np.mean(to)*100* (365/7) if False else ''}"
    tturn = "" if to is None else f"{np.mean(to)*100:5.0f}%"
    print(f"{label:24s}{1+m['tot']:6.2f}{m['ann']*100:+7.0f}%{m['sharpe']:+6.2f}{m['mdd']*100:6.0f}%{m['win']*100:5.0f}%{tturn:>8s}")


def product(panel, name, pool, factors, top_k):
    df = panel if pool == "wide" else panel[panel["sym"].isin(BIG_POOL)].copy()
    # 复合多头分: 各因子横截面 z 按"利多方向"求和
    df = df.copy()
    df["SCORE"] = sum(zc(df, c, s) for c, s in factors)
    df = df.dropna(subset=["SCORE"])
    nd = df["day"].nunique(); nc = df["sym"].nunique()
    print(f"\n{'='*86}\n产品 {name}: {' + '.join(c for c,_ in factors)}  "
          f"({'宽池' if pool=='wide' else '大币池'}{nc}币 × {nd}天, top{top_k} 等权现货多头)\n{'='*86}")
    print(f"{'方案':24s}{'净值':>6s}{'年化':>8s}{'夏普':>6s}{'回撤':>6s}{'胜率':>5s}{'换手':>8s}")
    print("-" * 86)
    bm = bench(df)
    show("基准·池等权买入持有", bm)

    # 主方案: 周调仓 15bps
    r, to = backtest_basket(df, "SCORE", top_k, rebal=7, cost_bps=15)
    show("★产品·周调仓(15bps)", r, to)
    # 对冲版: 多篮子 − 池基准 (= 用 perp 空一个指数对冲掉 beta, 只留 alpha)
    ex = (r - bm.reindex(r.index)).dropna()
    show("★产品·对冲beta(纯alpha)", ex)

    print("--- 工程压测: 调仓频率 × 成本 ---")
    for rebal, rn in [(1, "日调"), (7, "周调"), (14, "双周")]:
        for cost in [5, 15, 30]:
            r, to = backtest_basket(df, "SCORE", top_k, rebal=rebal, cost_bps=cost)
            show(f"  {rn}·{cost}bps", r, to)
    # 持仓数下限敏感性 (周调15bps)
    print("--- 持仓数 top-k 敏感性 (周调·15bps) ---")
    for k in sorted(set([max(3, top_k//2), top_k, top_k*2])):
        if k <= nc:
            r, to = backtest_basket(df, "SCORE", k, rebal=7, cost_bps=15)
            show(f"  top{k}", r, to)
    return df


def main():
    panel = build_panel()
    product(panel, "A·核心(730天)", "wide",
            [("AMP_lo_f", -1), ("FUND_f", -1)], top_k=15)
    product(panel, "B·增强(510天,+OI)", "big",
            [("AMP_lo_f", -1), ("FUND_f", -1), ("GLS_f", -1)], top_k=8)
    print("\n" + "=" * 86)
    print("交付结论见 README『现货多头可交付产品』节。")


if __name__ == "__main__":
    main()
