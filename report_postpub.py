# -*- coding: utf-8 -*-
"""
发布后样本外复核: 报告定稿于 2026-06-19 (数据止于 2026-05-31), 之后每一天都是样本外。

与 report_windows.py 用同一套策略定义 (不重新拟合, 全样本算日收益后按日期切片),
窗口改为: 年初至今 / 发布后 (6 月起) / 上次复核后 (8 月 15 日起) / 九月,
另给 6–9 月逐月收益。funding 月档只到上一个完整自然月, 相关策略的末日更早, 逐行标天数。

用法: python report_postpub.py
产物: report_postpub.csv (窗口统计) + report_postpub_monthly.csv (逐月收益) + 控制台表
"""
import os
import numpy as np
import pandas as pd

from backtest import backtest_factor, metrics
from backtest_perp import build_panel, BIG_POOL, FACTORS, pool_panel
from backtest_real import backtest_real
from deploy_spot import zc, backtest_basket, bench
from report_windows import utc_series, stat_row

ROOT = os.path.dirname(os.path.abspath(__file__))
WINDOWS = [
    ("年初至今", "2026-01-01"),
    ("发布后(6月起)", "2026-06-01"),
    ("上次复核后(8/15起)", "2026-08-15"),
    ("九月", "2026-09-01"),
]
MONTHS = ["2026-06", "2026-07", "2026-08", "2026-09"]


def all_series(panel):
    """Same strategy definitions as report_windows.main, returned as {group: {name: daily series}}."""
    groups = {}
    wide = panel.dropna(subset=["fwd_ret"])
    b = wide.groupby("day")["fwd_ret"].mean()
    bg = wide[wide["sym"].isin(BIG_POOL)].groupby("day")["fwd_ret"].mean()
    btc = wide[wide["sym"] == "BTCUSDT"].set_index("day")["fwd_ret"]
    groups["① 市场基准"] = {
        "基准·宽池等权": utc_series(b.values, b.index),
        "基准·大币池等权": utc_series(bg.values, bg.index),
        "基准·BTC 买入持有": utc_series(btc.values, btc.index),
    }

    ls = {}
    for col, name, d, pool, typ in FACTORS:
        if col not in panel.columns:
            continue
        bt = backtest_factor(pool_panel(panel, pool), col, d)
        if len(bt) >= 10:
            ls[f"{name}({'宽' if pool == 'wide' else '大'})"] = utc_series(bt["ls"].values, bt["day"].values)
    groups["② 单因子多空 (理想口径, 单边5bps)"] = ls

    real_ls, real_lo = {}, {}
    for col, name, d, pool, typ in FACTORS:
        if col not in panel.columns:
            continue
        bt = backtest_real(panel, col, d, pool)
        if len(bt) >= 10:
            real_ls[name] = utc_series(bt["ls_real"].values, bt["day"].values)
            real_lo[name] = utc_series(bt["long_only"].values, bt["day"].values)
    groups["③a 多空·计提 funding"] = real_ls
    groups["③b 纯多头现货超额"] = real_lo

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
    groups["④ 可交付产品 (周调仓·15bps)"] = prods
    return groups


def main():
    panel = build_panel()
    last = pd.Timestamp(panel["day"].max())
    print(f"\n面板: {panel['sym'].nunique()} 币 × {panel['day'].nunique()} 天 "
          f"({panel['day'].min().date()} ~ {last.date()})")
    print(f"  funding 覆盖至 {panel.dropna(subset=['FUND_f'])['day'].max().date()};"
          f"  OI 覆盖至 {panel.dropna(subset=['GLS_f'])['day'].max().date()}")

    rows, mrows = [], []
    for gname, series in all_series(panel).items():
        print(f"\n{'=' * 128}\n{gname}\n{'=' * 128}")
        print(f"{'策略':24s}" + "".join(f"{w:>26s}" for w, _ in WINDOWS))
        print(f"{'':24s}" + "".join(f"{'收益':>9s}{'回撤':>8s}{'夏普':>6s}{'天':>4s}" for _ in WINDOWS))
        for name, s in series.items():
            s = s.dropna().sort_index()
            parts, rec = "", {"group": gname, "strategy": name}
            for wname, start in WINDOWS:
                st = stat_row(s[s.index >= pd.Timestamp(start, tz="UTC")])
                if st is None:
                    parts += f"{'--':>9s}{'--':>8s}{'--':>6s}{'--':>4s}"
                    continue
                parts += f"{st['cum'] * 100:+8.1f}%{st['mdd'] * 100:+7.1f}%{st['sharpe']:+6.2f}{st['days']:4d}"
                for k, v in st.items():
                    rec[f"{wname}.{k}"] = v
            print(f"{name:24s}{parts}")
            rows.append(rec)
            mrec = {"group": gname, "strategy": name}
            for m in MONTHS:
                sm = s[s.index.strftime("%Y-%m") == m]
                mrec[m] = float(np.prod(1 + sm.values) - 1) if len(sm) >= 5 else np.nan
                mrec[f"{m}.days"] = len(sm)
            mrows.append(mrec)

    mdf = pd.DataFrame(mrows)
    print(f"\n{'=' * 90}\n逐月收益 (复利; 少于 5 天记为空)\n{'=' * 90}")
    show = mdf[mdf["group"].str.startswith(("①", "④")) | mdf["strategy"].isin(["资金费率(大)", "持仓量变化(大)", "低价位振幅(宽)", "散户多空比(大)", "理想反转(宽)"])]
    for _, r in show.iterrows():
        cells = "".join(f"{(r[m] * 100):+8.1f}%({int(r[m + '.days']):2d})" if pd.notna(r[m]) else f"{'--':>13s}" for m in MONTHS)
        print(f"{r['strategy']:24s}{cells}")
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "report_postpub.csv"), index=False)
    mdf.to_csv(os.path.join(ROOT, "report_postpub_monthly.csv"), index=False)
    print("\n已存: report_postpub.csv, report_postpub_monthly.csv")


if __name__ == "__main__":
    main()
