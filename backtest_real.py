# -*- coding: utf-8 -*-
"""
把回测"做实"—— 去掉理想化假设, 看哪些 alpha 真能落地。

现有回测的水分: 所有夏普都假设"能免费做空个币"。crypto 现货不能做空, 唯一空腿是 **perp**,
于是空腿要 **计提资金费 (funding)** —— 而 funding 又恰是我们的 alpha, 两者必须联立。

本脚本给每个因子三个口径:
  ① 多空(纯价格)    : 现有理想口径 (long/short 个币, 不计 funding)            <- 有水分
  ② 多空(计funding)  : perp 双腿, 空腿收/付资金费 (按真实方向计提)             <- 可部署多空
  ③ 纯多头spot超额   : 只买最高分位 (现货可做, 不做空), 减市场等权 = 不靠做空的 alpha
并对"资金费率"因子做 价格 vs carry 分解 (它的钱到底来自反转还是吃 carry?)。

funding 计提口径: last_funding_rate>0 => 多头付、空头收。持有期(t→t+1)用次日 funding (fwd_fund)。
  多头 funding 损益 = −fwd_fund ; 空头 = +fwd_fund 。
"""
import os
import numpy as np
import pandas as pd
from backtest import metrics, QUANTILE, COST_BPS
from backtest_perp import build_panel, BIG_POOL, FACTORS, pool_panel


def backtest_real(panel, col, direction, pool):
    df = pool_panel(panel, pool).dropna(subset=[col, "fwd_ret"]).copy()
    df["sig"] = df[col] * direction
    has_fund = "fwd_fund" in df.columns
    rows = []
    prev_l, prev_s = set(), set()
    for day, g in df.groupby("day"):
        if g["sym"].nunique() < QUANTILE * 2:
            continue
        g = g.copy()
        g["q"] = pd.qcut(g["sig"].rank(method="first"), QUANTILE, labels=False)
        longs, shorts = g[g["q"] == QUANTILE - 1], g[g["q"] == 0]
        lr, sr = longs["fwd_ret"].mean(), shorts["fwd_ret"].mean()
        mkt = g["fwd_ret"].mean()
        cl, cs = set(longs["sym"]), set(shorts["sym"])
        to = (len(cl ^ prev_l) + len(cs ^ prev_s)) / max(len(cl) + len(cs), 1)
        to_l = len(cl ^ prev_l) / max(len(cl), 1)
        cost = to * COST_BPS / 1e4
        # funding 计提
        lf = longs["fwd_fund"].mean() if has_fund else 0.0
        sf = shorts["fwd_fund"].mean() if has_fund else 0.0
        lf = 0.0 if pd.isna(lf) else lf
        sf = 0.0 if pd.isna(sf) else sf
        fund_pnl = sf - lf                     # 空腿收 + 多腿付
        ls_price = (lr - sr) - cost
        ls_real = ls_price + fund_pnl
        long_only = (lr - mkt) - to_l * COST_BPS / 1e4   # 纯多头超额(不做空)
        rows.append(dict(day=day, ls_price=ls_price, ls_real=ls_real,
                         long_only=long_only, fund_pnl=fund_pnl, price=lr - sr))
        prev_l, prev_s = cl, cs
    return pd.DataFrame(rows)


def line(name, r, extra=""):
    m = metrics(r)
    if not m:
        print(f"{name:14s} 样本不足"); return
    print(f"{name:14s}{1+m['tot']:6.2f}{m['ann']*100:+7.0f}%{m['sharpe']:+6.2f}{m['mdd']*100:6.0f}%{m['win']*100:5.0f}%  {extra}")


def main():
    panel = build_panel()
    print(f"\n{'='*84}")
    print("把回测做实: 多空(纯价格) vs 多空(计funding) vs 纯多头spot超额")
    print(f"  现货不可做空 → 空腿走 perp 计提资金费; 纯多头=只买最高分位减市场 (无需做空)")
    print(f"{'='*84}")

    for tag, sub in [("【多空·纯价格 (理想,有水分)】", "ls_price"),
                     ("【多空·计funding (可部署)】", "ls_real"),
                     ("【纯多头 spot 超额 (无需做空)】", "long_only")]:
        print(f"\n{tag}")
        print(f"{'因子':14s}{'净值':>6s}{'年化':>8s}{'夏普':>6s}{'回撤':>6s}{'胜率':>5s}")
        print("-" * 64)
        for col, name, d, pool, typ in FACTORS:
            if col not in panel.columns:
                continue
            bt = backtest_real(panel, col, d, pool)
            if len(bt):
                line(name, bt[sub].values)

    # ---- 资金费率因子: 价格 vs carry 分解 ----
    print(f"\n{'='*84}\n[资金费率因子 收益分解] 它的钱来自'价格反转'还是'吃 carry'?\n{'='*84}")
    bt = backtest_real(panel, "FUND_f", -1, "big")
    mp, mf, mr = metrics(bt["price"].values), metrics(bt["fund_pnl"].values), metrics(bt["ls_real"].values)
    print(f"  价格反转分量 (空高funding币, 价格回落): 净值{1+mp['tot']:.2f} 夏普{mp['sharpe']:+.2f}")
    print(f"  carry 分量   (空高funding收资金费)     : 净值{1+mf['tot']:.2f} 夏普{mf['sharpe']:+.2f}  "
          f"(日均{bt['fund_pnl'].mean()*1e4:+.1f}bp, 年化{bt['fund_pnl'].mean()*365*100:+.0f}%)")
    print(f"  合计 (计funding 多空)                  : 净值{1+mr['tot']:.2f} 夏普{mr['sharpe']:+.2f}")
    share = bt["fund_pnl"].sum() / (bt["price"].sum() + bt["fund_pnl"].sum() + 1e-12)
    print(f"  => carry 占总收益约 {share*100:.0f}%  "
          f"({'主要吃carry' if share>0.6 else 'carry与反转皆贡献' if share>0.25 else '主要是价格反转'})")
    print("=" * 84)


if __name__ == "__main__":
    main()
