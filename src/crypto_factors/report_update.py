# -*- coding: utf-8 -*-
"""
更新后的分期业绩报告 —— 同一套策略, 拆成"长期(全样本)" vs "上次更新以来"。

口径原则 (避免制造假结论):
  * 策略**不重新拟合**。所有因子都是横截面排序, 没有需要估计的参数; 所以先在全样本上
    算出每日收益序列, 再**按日期切片**统计。新窗口因此是真正的样本外 —— 上次更新
    (README 定稿于 2026-06-19, 数据止于 2026-05-31) 时这些日子还不存在。
  * 短窗口 (~2 个月) 的年化/夏普是强外推, 只作方向参考; 真正可信的是区间累计收益与胜率。
  * 各因子的可用末日不同 (funding dump 只到上一完整月末), 表里逐行标注实际天数。

用法: python report_update.py [切分日=2026-06-01]
产物: report_update.csv (机器可读) + 控制台分期表
"""
import os, sys
import numpy as np
import pandas as pd

from backtest import backtest_factor, metrics, QUANTILE, COST_BPS
from backtest_perp import build_panel, BIG_POOL, FACTORS, pool_panel
from backtest_real import backtest_real
from deploy_spot import zc, backtest_basket, bench

ROOT = os.path.dirname(os.path.abspath(__file__))
SPLIT = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2026-06-01", tz="UTC")


# ------------------------------------------------------------------ 分期统计
def slice_periods(s, split):
    s = s.dropna()
    idx = pd.to_datetime(s.index, utc=True)
    s = pd.Series(s.values, index=idx).sort_index()
    return {"全样本": s, "旧窗(截至上次更新)": s[s.index < split], "新增(上次更新以来)": s[s.index >= split]}


def stat_row(s):
    m = metrics(s.values)
    if not m:
        return None
    return dict(days=len(s), start=s.index[0].date(), end=s.index[-1].date(),
                cum=m["tot"], ann=m["ann"], sharpe=m["sharpe"], mdd=m["mdd"], win=m["win"])


def table(name2series, title, note=""):
    rows = []
    print(f"\n{'='*104}")
    print(title)
    if note:
        print(note)
    print("=" * 104)
    print(f"{'策略':22s}"
          f"{'全样本':>28s}{'旧窗 (~2026-05-31)':>26s}{'新增 (上次更新以来)':>28s}")
    print(f"{'':22s}" + "".join(f"{'区间收益':>10s}{'夏普':>7s}{'天数':>6s}" for _ in range(3)))
    print("-" * 104)
    for name, s in name2series.items():
        parts, rec = "", {"strategy": name}
        for pname, ps in slice_periods(s, SPLIT).items():
            st = stat_row(ps)
            if st is None:
                parts += f"{'--':>10s}{'--':>7s}{'--':>6s}"
                continue
            parts += f"{st['cum']*100:+9.1f}%{st['sharpe']:+7.2f}{st['days']:6d}"
            for k, v in st.items():
                rec[f"{pname}.{k}"] = v
        print(f"{name:22s}{parts}")
        rows.append(rec)
    print("-" * 104)
    return rows


def detail(name2series, title):
    """新窗口的完整明细 (短窗年化只作参考)。"""
    print(f"\n{title}")
    print(f"{'策略':22s}{'区间收益':>10s}{'年化*':>9s}{'夏普*':>7s}{'回撤':>8s}{'日胜率':>8s}{'天数':>6s}{'起止':>24s}")
    print("-" * 96)
    for name, s in name2series.items():
        st = stat_row(slice_periods(s, SPLIT)["新增(上次更新以来)"])
        if st is None:
            print(f"{name:22s}  (新窗样本不足)"); continue
        print(f"{name:22s}{st['cum']*100:+9.1f}%{st['ann']*100:+8.0f}%{st['sharpe']:+7.2f}"
              f"{st['mdd']*100:+7.1f}%{st['win']*100:7.1f}%{st['days']:6d}"
              f"{str(st['start'])+'~'+str(st['end']):>24s}")
    print("-" * 96)
    print("* 年化/夏普由 ~2 个月外推而来, 噪声极大, 只看方向; 区间收益与天数才是硬事实。")


# ------------------------------------------------------------------ main
def main():
    panel = build_panel()
    last = panel["day"].max()
    print(f"\n面板: {panel['sym'].nunique()} 币 × {panel['day'].nunique()} 天  "
          f"({panel['day'].min().date()} ~ {last.date()})   切分日 = {SPLIT.date()}")
    fnd = panel.dropna(subset=["FUND_f"])
    print(f"  funding 覆盖至 {fnd['day'].max().date()}"
          f" ({'含 proxy' if 'FUND_src' in panel.columns else ''});"
          f"  OI 覆盖至 {panel.dropna(subset=['GLS_f'])['day'].max().date()}")

    allrows = []

    # ---------- 1. 基准 ----------
    bm = {}
    wide = panel.dropna(subset=["fwd_ret"])
    b = wide.groupby("day")["fwd_ret"].mean()
    bm["基准·宽池等权"] = pd.Series(b.values, index=b.index)
    bg = wide[wide["sym"].isin(BIG_POOL)].groupby("day")["fwd_ret"].mean()
    bm["基准·大币池等权"] = pd.Series(bg.values, index=bg.index)
    btc = wide[wide["sym"] == "BTCUSDT"].set_index("day")["fwd_ret"]
    bm["基准·BTC 买入持有"] = btc
    allrows += table(bm, "① 市场基准 (判断策略是 alpha 还是 beta 的参照系)")

    # ---------- 2. 单因子横截面多空 ----------
    ls = {}
    for col, name, d, pool, typ in FACTORS:
        if col not in panel.columns:
            continue
        bt = backtest_factor(pool_panel(panel, pool), col, d)
        if len(bt) < 10:
            continue
        ls[f"{name}({'宽' if pool=='wide' else '大'})"] = pd.Series(
            bt["ls"].values, index=pd.to_datetime(bt["day"].values, utc=True))
    allrows += table(ls, "② 单因子 横截面五分位多空 (理想口径: 可自由做空个币, 单边5bps)",
                     "  注: 这是研究口径, 有水分 —— 现货不能做空, 空腿实际要走 perp 并计提资金费, 见 ③。")

    # ---------- 3. 去水分口径 ----------
    real_lo, real_ls = {}, {}
    for col, name, d, pool, typ in FACTORS:
        if col not in panel.columns:
            continue
        bt = backtest_real(panel, col, d, pool)
        if len(bt) < 10:
            continue
        ix = pd.to_datetime(bt["day"].values, utc=True)
        real_ls[name] = pd.Series(bt["ls_real"].values, index=ix)
        real_lo[name] = pd.Series(bt["long_only"].values, index=ix)
    allrows += table(real_ls, "③a 多空·计提 funding (可部署口径: 空腿走 perp, 收/付资金费)")
    allrows += table(real_lo, "③b 纯多头现货超额 (完全不做空: 只买最高分位 − 市场等权)")

    # ---------- 4. 可交付产品 ----------
    prods = {}
    specs = [("产品A·宽池top15", "wide", [("AMP_lo_f", -1), ("FUND_f", -1)], 15),
             ("产品B·大币top8", "big", [("AMP_lo_f", -1), ("FUND_f", -1), ("GLS_f", -1)], 8)]
    for nm, pool, facs, k in specs:
        df = pool_panel(panel, pool).copy()
        df["SCORE"] = sum(zc(df, c, s) for c, s in facs)
        df = df.dropna(subset=["SCORE"])
        r, to = backtest_basket(df, "SCORE", k, rebal=7, cost_bps=15)
        r.index = pd.to_datetime(r.index, utc=True)
        bmk = bench(df); bmk.index = pd.to_datetime(bmk.index, utc=True)
        prods[f"{nm}·纯现货多头"] = r
        prods[f"{nm}·对冲beta"] = (r - bmk.reindex(r.index)).dropna()
    allrows += table(prods, "④ 可交付产品 (周调仓·15bps; 对冲版 = 现货多头篮子 − perp 空指数)")
    detail(prods, "④' 可交付产品 —— 新增窗口明细")
    detail(real_lo, "③b' 纯多头超额 —— 新增窗口明细")

    out = os.path.join(ROOT, "report_update.csv")
    pd.DataFrame(allrows).to_csv(out, index=False)
    print(f"\n分期明细已存: {out}")


if __name__ == "__main__":
    main()
