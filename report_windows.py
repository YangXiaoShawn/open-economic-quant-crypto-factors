# -*- coding: utf-8 -*-
"""
三窗口业绩对比: 年初至今 / 七月初至今 / 近15天。

策略不重新拟合。全样本算出日收益后再按日期切片。
funding dump 只有月档, 末日可能早于价量, 表里逐行标天数。

用法: python report_windows.py
产物: report_windows.csv + 控制台表
"""
import os
import numpy as np
import pandas as pd

from backtest import backtest_factor, metrics
from backtest_perp import build_panel, BIG_POOL, FACTORS, pool_panel
from backtest_real import backtest_real
from deploy_spot import zc, backtest_basket, bench

ROOT = os.path.dirname(os.path.abspath(__file__))
YTD = pd.Timestamp("2026-01-01", tz="UTC")
JUL = pd.Timestamp("2026-07-01", tz="UTC")


def utc_series(values, days):
    return pd.Series(values, index=pd.to_datetime(days, utc=True)).sort_index().dropna()


def windows_of(s, last):
    s = s.dropna().sort_index()
    cut15 = last - pd.Timedelta(days=14)
    return {
        "年初至今": s[s.index >= YTD],
        "七月初至今": s[s.index >= JUL],
        "近15天": s[s.index >= cut15],
    }


def stat_row(s):
    if s is None or len(s) < 5:
        return None
    m = metrics(s.values)
    if not m:
        return None
    return dict(days=len(s), start=s.index[0].date(), end=s.index[-1].date(),
                cum=m["tot"], ann=m["ann"], sharpe=m["sharpe"], mdd=m["mdd"], win=m["win"])


def table(name2series, title, last, note=""):
    rows = []
    wnames = ["年初至今", "七月初至今", "近15天"]
    print(f"\n{'='*118}")
    print(title)
    if note:
        print(note)
    print("=" * 118)
    print(f"{'策略':22s}" + "".join(f"{w:>32s}" for w in wnames))
    print(f"{'':22s}" + "".join(f"{'区间收益':>10s}{'回撤':>8s}{'夏普':>7s}{'天':>5s}" for _ in wnames))
    print("-" * 118)
    for name, s in name2series.items():
        parts, rec = "", {"strategy": name}
        for wname, ws in windows_of(s, last).items():
            st = stat_row(ws)
            if st is None:
                parts += f"{'--':>10s}{'--':>8s}{'--':>7s}{'--':>5s}"
                rec[f"{wname}.days"] = 0
                continue
            parts += f"{st['cum']*100:+9.1f}%{st['mdd']*100:+7.1f}%{st['sharpe']:+7.2f}{st['days']:5d}"
            for k, v in st.items():
                rec[f"{wname}.{k}"] = v
        print(f"{name:22s}{parts}")
        rows.append(rec)
    print("-" * 118)
    return rows


def detail(name2series, title, last):
    print(f"\n{title}")
    print(f"{'策略':22s}{'窗口':>10s}{'区间收益':>10s}{'年化*':>8s}{'夏普*':>7s}{'回撤':>8s}{'胜率':>7s}{'天':>5s}{'起止':>24s}")
    print("-" * 110)
    for name, s in name2series.items():
        for wname, ws in windows_of(s, last).items():
            st = stat_row(ws)
            if st is None:
                print(f"{name:22s}{wname:>10s}  (样本不足)")
                continue
            print(f"{name:22s}{wname:>10s}{st['cum']*100:+9.1f}%{st['ann']*100:+7.0f}%"
                  f"{st['sharpe']:+7.2f}{st['mdd']*100:+7.1f}%{st['win']*100:6.1f}%"
                  f"{st['days']:5d}{str(st['start'])+'~'+str(st['end']):>24s}")
    print("-" * 110)
    print("* 短窗年化/夏普外推噪声大, 区间收益、回撤、天数才是硬事实。")


def main():
    panel = build_panel()
    last = pd.Timestamp(panel["day"].max())
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    else:
        last = last.tz_convert("UTC")
    print(f"\n面板: {panel['sym'].nunique()} 币 × {panel['day'].nunique()} 天  "
          f"({panel['day'].min().date()} ~ {last.date()})")
    fnd = panel.dropna(subset=["FUND_f"])
    print(f"  funding 覆盖至 {fnd['day'].max().date()};"
          f"  OI 覆盖至 {panel.dropna(subset=['GLS_f'])['day'].max().date()}")
    print(f"  窗口: 年初至今>={YTD.date()}  七月初至今>={JUL.date()}  "
          f"近15天>={(last - pd.Timedelta(days=14)).date()}")

    allrows = []

    bm = {}
    wide = panel.dropna(subset=["fwd_ret"])
    b = wide.groupby("day")["fwd_ret"].mean()
    bm["基准·宽池等权"] = utc_series(b.values, b.index)
    bg = wide[wide["sym"].isin(BIG_POOL)].groupby("day")["fwd_ret"].mean()
    bm["基准·大币池等权"] = utc_series(bg.values, bg.index)
    btc = wide[wide["sym"] == "BTCUSDT"].set_index("day")["fwd_ret"]
    bm["基准·BTC 买入持有"] = utc_series(btc.values, btc.index)
    allrows += table(bm, "① 市场基准", last)

    ls = {}
    for col, name, d, pool, typ in FACTORS:
        if col not in panel.columns:
            continue
        bt = backtest_factor(pool_panel(panel, pool), col, d)
        if len(bt) < 10:
            continue
        ls[f"{name}({'宽' if pool=='wide' else '大'})"] = utc_series(bt["ls"].values, bt["day"].values)
    allrows += table(ls, "② 单因子 横截面五分位多空 (理想口径, 单边5bps)", last,
                     "  研究口径: 假设可自由做空个币。现货不能做空, 见 ③。")

    real_lo, real_ls = {}, {}
    for col, name, d, pool, typ in FACTORS:
        if col not in panel.columns:
            continue
        bt = backtest_real(panel, col, d, pool)
        if len(bt) < 10:
            continue
        real_ls[name] = utc_series(bt["ls_real"].values, bt["day"].values)
        real_lo[name] = utc_series(bt["long_only"].values, bt["day"].values)
    allrows += table(real_ls, "③a 多空·计提 funding (空腿走 perp)", last)
    allrows += table(real_lo, "③b 纯多头现货超额 (只买最高分位 − 市场等权)", last)

    prods = {}
    specs = [("产品A·宽池top15", "wide", [("AMP_lo_f", -1), ("FUND_f", -1)], 15),
             ("产品B·大币top8", "big", [("AMP_lo_f", -1), ("FUND_f", -1), ("GLS_f", -1)], 8)]
    for nm, pool, facs, k in specs:
        df = pool_panel(panel, pool).copy()
        df["SCORE"] = sum(zc(df, c, s) for c, s in facs)
        df = df.dropna(subset=["SCORE"])
        r, to = backtest_basket(df, "SCORE", k, rebal=7, cost_bps=15)
        r.index = pd.to_datetime(r.index, utc=True)
        bmk = bench(df)
        bmk.index = pd.to_datetime(bmk.index, utc=True)
        prods[f"{nm}·纯现货多头"] = r
        prods[f"{nm}·对冲beta"] = (r - bmk.reindex(r.index)).dropna()
    allrows += table(prods, "④ 可交付产品 (周调仓·15bps; 对冲版 = 现货篮子 − perp 空指数)", last)
    detail(prods, "④' 可交付产品明细", last)
    detail(real_lo, "③b' 纯多头超额明细", last)

    out = os.path.join(ROOT, "report_windows.csv")
    pd.DataFrame(allrows).to_csv(out, index=False)
    print(f"\n三窗口明细已存: {out}")


if __name__ == "__main__":
    main()
