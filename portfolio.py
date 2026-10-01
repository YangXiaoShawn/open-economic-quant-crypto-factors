# -*- coding: utf-8 -*-
"""
组合层: 把"已迁移"的低相关因子合成一个组合, 验证能否降回撤、升夏普。
- 取独立稳健/边缘因子的每日多空收益流
- 看相关性 -> 选低相关的几个
- 等权 / 逆波动(风险平价) 两种合成, 对比单因子
"""
import os
import numpy as np
import pandas as pd
from backtest import backtest_factor, metrics
from backtest_all32 import build_panel

# 候选: 反转 / 大单资金流 / 主动买入 / 振幅切割 / 截面动量
CANDS = [
    ("REV_hi_f", -1, "理想反转"),
    ("OFIbig_f", +1, "大单资金流"),
    ("OFI_f",    +1, "主动买入"),
    ("AMP_lo_f", -1, "低价位振幅"),
    ("MOM_f",    +1, "截面动量"),
]


def factor_ls_series(panel, col, d):
    bt = backtest_factor(panel, col, d)
    return pd.Series(bt["ls"].values, index=pd.to_datetime(bt["day"].values))


def summary(r, label):
    m = metrics(r.values)
    print(f"{label:16s}净值{1+m['tot']:5.2f}  年化{m['ann']*100:+6.0f}%  夏普{m['sharpe']:+5.2f}"
          f"  回撤{m['mdd']*100:5.0f}%  日胜率{m['win']*100:4.0f}%")
    return m


def main():
    panel = build_panel()
    nc = panel["sym"].nunique()
    print(f"\n{'='*72}\n组合层回测  ({nc}币 × {panel['day'].nunique()}天)\n{'='*72}")

    # 各因子收益流
    series = {}
    print("\n[单因子多空]")
    for col, d, name in CANDS:
        s = factor_ls_series(panel, col, d)
        series[name] = s
        summary(s, name)

    df = pd.DataFrame(series).dropna()
    print(f"\n[因子收益流 相关矩阵]  (低相关才有分散价值)")
    print((df.corr()*100).round(0).astype(int).to_string())

    # 选低相关核心: 反转 + 大单资金流 (价量 vs 资金流, 天然低相关)
    core = ["理想反转", "大单资金流"]
    print(f"\n[组合: {' + '.join(core)}]  (反转×资金流, 跨类低相关)")
    ew = df[core].mean(axis=1)
    summary(ew, "  等权组合")
    # 逆波动加权(风险平价)
    iv = 1 / df[core].std()
    iv = iv / iv.sum()
    rp = (df[core] * iv).sum(axis=1)
    summary(rp, "  逆波动组合")

    # 全5因子等权
    print(f"\n[组合: 全5因子等权]")
    summary(df.mean(axis=1), "  5因子等权")

    # 与最佳单因子对比
    best = max(CANDS, key=lambda c: metrics(series[c[2]].values)["sharpe"])
    bm = metrics(series[best[2]].values)
    cm = metrics(ew.values)
    print(f"\n[结论] 最佳单因子 {best[2]}: 夏普{bm['sharpe']:.2f}/回撤{bm['mdd']*100:.0f}%")
    print(f"       反转+资金流等权:    夏普{cm['sharpe']:.2f}/回撤{cm['mdd']*100:.0f}%")
    dd_better = (cm['mdd'] - bm['mdd']) * 100
    print(f"       => 组合夏普 {cm['sharpe']-bm['sharpe']:+.2f}, 回撤 {dd_better:+.0f}pct "
          f"({'改善' if cm['mdd']>bm['mdd'] else '恶化'})")

    out = os.path.join(os.path.dirname(__file__), "equity_portfolio.csv")
    eq = pd.DataFrame({"reversal": (1+df['理想反转']).cumprod(),
                       "flow": (1+df['大单资金流']).cumprod(),
                       "combo_ew": (1+ew).cumprod(),
                       "combo_rp": (1+rp).cumprod()})
    eq.to_csv(out)
    print(f"\n净值曲线已存: {out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
