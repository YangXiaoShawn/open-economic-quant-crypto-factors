# -*- coding: utf-8 -*-
"""
样本外验证 (防过拟合) —— 在已有 Binance 面板上做三类内部 OOS 检验:
  1. 时间分割: 因子在 前半/后半 + 逐年子期 是否同号稳健 (防"全样本选因子"过拟合);
  2. 参数稳健: 变平滑窗 N、变分位数 q, 夏普是否随参数崩塌 (防参数挑选);
  3. 滚动稳定: 120 日滚动夏普, 正占比 / 中位 (防"靠某段行情")。
跨交易所 OOS (OKX) 见 oos_okx.py。
"""
import os
import numpy as np
import pandas as pd
import backtest
from backtest import backtest_factor, metrics
from backtest_perp import build_panel, BIG_POOL, pool_panel

# (因子列, 名称, 方向, 池, 原始列, 平滑方式)  原始列=None 则不做变N
FACS = [
    ("REV_hi_f", "理想反转",   -1, "wide", "REV_hi",  "sum"),
    ("AMP_lo_f", "低价位振幅", -1, "wide", "AMP_lo",  "mean"),
    ("MOM_f",    "截面动量",   +1, "wide", None,      None),
    ("OFIbig_f", "大额主动买", +1, "big",  "OFI_big", "mean"),
    ("FUND_f",   "资金费率",   -1, "big",  None,      None),
    ("GLS_f",    "散户多空比", -1, "big",  None,      None),
]


def ls_series(panel, col, d, pool):
    bt = backtest_factor(pool_panel(panel, pool), col, d)
    return pd.Series(bt["ls"].values, index=pd.to_datetime(bt["day"].values))


def sh(r):
    m = metrics(r.values if hasattr(r, "values") else r)
    return m["sharpe"] if m else np.nan


def part1_timesplit(panel):
    print(f"\n{'='*92}\n1. 时间分割 OOS: 同一因子在 前半/后半 + 逐年 的夏普 (同号且都为正=稳健, 防选因子过拟合)\n{'='*92}")
    days = pd.to_datetime(sorted(panel["day"].unique())).tz_localize(None)
    split = days[len(days)//2]
    print(f"  分割点={split.date()}  前半 {days[0].date()}~{split.date()}  后半 {split.date()}~{days[-1].date()}")
    print(f"\n{'因子':12s}{'全样本':>8s}{'前半H1':>8s}{'后半H2':>8s}{'24下半':>8s}{'2025':>8s}{'26上半':>8s}  判定")
    print("-" * 92)
    for col, name, d, pool, _, _ in FACS:
        s = ls_series(panel, col, d, pool)
        full = sh(s); h1 = sh(s[s.index < split]); h2 = sh(s[s.index >= split])
        y24 = sh(s[(s.index >= "2024-06") & (s.index < "2025-01")])
        y25 = sh(s[(s.index >= "2025-01") & (s.index < "2026-01")])
        y26 = sh(s[s.index >= "2026-01"])
        robust = (h1 > 0 and h2 > 0)
        verdict = "✅两半皆正" if robust else ("◐单边" if (h1 > 0) != (h2 > 0) else "✗两半皆负")
        print(f"{name:12s}{full:+8.2f}{h1:+8.2f}{h2:+8.2f}{y24:+8.2f}{y25:+8.2f}{y26:+8.2f}  {verdict}")
    print("-" * 92)
    print("注: 大币池每腿~5名, 短子期(逐年/26上半~5月)噪声大, 重点看 H1 vs H2 同号。")


def resmooth(panel, raw, N, how):
    p = panel.sort_values(["sym", "day"])
    g = p.groupby("sym")[raw]
    s = g.transform(lambda x: x.rolling(N, min_periods=2).sum()) if how == "sum" \
        else g.transform(lambda x: x.rolling(N, min_periods=2).mean())
    out = panel.copy(); out["_tmp"] = s.reindex(panel.index)
    return out


def part2_params(panel):
    print(f"\n{'='*92}\n2. 参数稳健: 变平滑窗 N / 变分位数 q —— 夏普应平滑变化, 不应只在某个参数为正\n{'='*92}")
    print(f"\n[变平滑窗 N]  (默认 N=5)\n{'因子':12s}" + "".join(f"{'N='+str(n):>8s}" for n in [2,3,5,8,13]))
    print("-" * 60)
    for col, name, d, pool, raw, how in FACS:
        if raw is None:
            continue
        row = f"{name:12s}"
        for N in [2, 3, 5, 8, 13]:
            pp = resmooth(panel, raw, N, how)
            row += f"{sh(ls_series(pp, '_tmp', d, pool)):+8.2f}"
        print(row)

    print(f"\n[变分位数 q]  (默认 q=5)\n{'因子':12s}" + "".join(f"{'q='+str(q):>8s}" for q in [3,4,5,8,10]))
    print("-" * 60)
    old = backtest.QUANTILE
    try:
        for col, name, d, pool, _, _ in FACS:
            row = f"{name:12s}"
            for q in [3, 4, 5, 8, 10]:
                backtest.QUANTILE = q
                row += f"{sh(ls_series(panel, col, d, pool)):+8.2f}"
            print(row)
    finally:
        backtest.QUANTILE = old


def part3_rolling(panel, win=120):
    print(f"\n{'='*92}\n3. 滚动稳定: {win}日滚动夏普 (年化) —— 正窗占比高=不靠单段行情\n{'='*92}")
    print(f"{'因子':12s}{'正窗占比':>9s}{'中位夏普':>9s}{'最差':>8s}{'最好':>8s}")
    print("-" * 56)
    for col, name, d, pool, _, _ in FACS:
        s = ls_series(panel, col, d, pool).dropna()
        roll = s.rolling(win).apply(lambda x: (x.mean()/(x.std()+1e-12))*np.sqrt(365), raw=True).dropna()
        if len(roll) < 10:
            print(f"{name:12s}  窗口不足"); continue
        print(f"{name:12s}{(roll>0).mean()*100:8.0f}%{roll.median():+9.2f}{roll.min():+8.2f}{roll.max():+8.2f}")


def main():
    panel = build_panel()
    print(f"OOS 内部检验  ({panel['sym'].nunique()}币 × {panel['day'].nunique()}天)")
    part1_timesplit(panel)
    part2_params(panel)
    part3_rolling(panel)
    print("=" * 92)


if __name__ == "__main__":
    main()
