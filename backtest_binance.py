# -*- coding: utf-8 -*-
"""
数百天回测 + T2 逐笔真回测 (Binance dump 聚合面板)。
价格类因子 + T2 资金流因子(taker主动买卖) 一起跑，覆盖 ~2 年、24 币。
复用 backtest.py 的 backtest_factor / metrics。
"""
import os, glob
import numpy as np
import pandas as pd
from backtest import backtest_factor, metrics

DATADIR = os.path.join(os.path.dirname(__file__), "data_binance")
SMOOTH_N = 5

# (原始列, 因子列, 名称, 研报, 方向)  -1=反转/contrarian, +1=动量/顺势
RAW = [
    ("Q",       "Q_f",       "聪明钱Q",     "(3)",    -1),
    ("REV_hi",  "REV_hi_f",  "理想反转hi",  "(1,13)", -1),
    (None,      "APM_f",     "APM",         "(5)",    -1),
    ("AMP",     "AMP_f",     "振幅",        "(7)",    -1),
    ("AMP_lo",  "AMP_lo_f",  "低价位振幅",  "(30)",   -1),
    ("SKEW",    "SKEW_f",    "收益偏度",    "(微结构)", -1),
    ("AC1",     "AC1_f",     "日内自相关",  "(19)",   -1),
    ("VCONC",   "VCONC_f",   "量集中度",    "(27)",   -1),
    ("VRCORR",  "VRCORR_f",  "量波同步",    "(27)",   -1),
    # ---- T2 资金流 (taker 主动买卖方向, 真逐笔聚合) ----
    ("OFI",     "OFI_f",     "主动买入率",  "(9/12)", +1),
    ("OFI_big", "OFIbig_f",  "大额主动买", "(16/18)", +1),
    ("AMTPT",   "AMTPT_f",   "单笔金额",    "(15)",   +1),
]


def build_panel():
    files = glob.glob(os.path.join(DATADIR, "*.parquet"))
    recs = []
    for fn in files:
        d = pd.read_parquet(fn).sort_values("day").reset_index(drop=True)
        d["fwd_ret"] = d["close"].shift(-1) / d["close"] - 1.0
        d["Q_f"]      = d["Q"].rolling(SMOOTH_N, min_periods=2).mean()
        d["REV_hi_f"] = d["REV_hi"].rolling(SMOOTH_N, min_periods=2).sum()
        d["APM_f"]    = (d["r_pm"] - d["r_am"]).rolling(SMOOTH_N, min_periods=2).mean()
        d["AMP_f"]    = d["AMP"].rolling(SMOOTH_N, min_periods=2).mean()
        d["AMP_lo_f"] = d["AMP_lo"].rolling(SMOOTH_N, min_periods=2).mean()
        d["SKEW_f"]   = d["SKEW"].rolling(SMOOTH_N, min_periods=2).mean()
        d["AC1_f"]    = d["AC1"].rolling(SMOOTH_N, min_periods=2).mean()
        d["VCONC_f"]  = d["VCONC"].rolling(SMOOTH_N, min_periods=2).mean()
        d["VRCORR_f"] = d["VRCORR"].rolling(SMOOTH_N, min_periods=2).mean()
        d["OFI_f"]    = d["OFI"].rolling(SMOOTH_N, min_periods=2).mean()
        d["OFIbig_f"] = d["OFI_big"].rolling(SMOOTH_N, min_periods=2).mean()
        d["AMTPT_f"]  = (d["AMTPT"] / d["AMTPT"].rolling(20, min_periods=5).mean())  # 相对单笔额
        recs.append(d)
    return pd.concat(recs, ignore_index=True)


def main():
    panel = build_panel()
    nd, nc = panel["day"].nunique(), panel["sym"].nunique()
    print(f"\n{'='*80}")
    print(f"Binance数百天回测 (价格+T2资金流)  {nc}币 × {nd}天  "
          f"{panel['day'].min().date()}~{panel['day'].max().date()}")
    print(f"五分位多空, 日调仓, 单边5bps")
    print(f"{'='*80}")
    print(f"{'研报因子':16s}{'净值':>7s}{'年化':>8s}{'夏普':>7s}{'回撤':>8s}"
          f"{'日胜率':>7s}{'换手':>6s}  类型")
    print("-" * 80)
    eq = {}
    for raw, col, name, rpt, d in RAW:
        bt = backtest_factor(panel, col, d)
        m = metrics(bt["ls"])
        tag = "T2资金流" if raw in ("OFI","OFI_big","AMTPT") else "价格"
        if not m:
            print(f"{name+rpt:16s}  样本不足"); continue
        eq[name] = (bt["day"].values, np.cumprod(1+bt["ls"].values))
        print(f"{name+rpt:16s}{1+m['tot']:7.2f}{m['ann']*100:+7.0f}%{m['sharpe']:7.2f}"
              f"{m['mdd']*100:7.0f}%{m['win']*100:6.0f}%{bt['turnover'].mean()*100:5.0f}%  {tag}")
    print("-" * 80)

    def combo(items, label):
        sub = panel.copy(); zs = []
        for col, dd in items:
            z = sub.groupby("day")[col].transform(lambda x: (x-x.mean())/(x.std()+1e-9))
            zs.append(z.values*dd)
        st = np.vstack(zs)
        sub["cf"] = np.where(np.isnan(st).all(0), np.nan, np.nanmean(st, axis=0))
        bt = backtest_factor(sub, "cf", +1); m = metrics(bt["ls"])
        if m:
            eq[label] = (bt["day"].values, np.cumprod(1+bt["ls"].values))
            print(f"{label:16s}{1+m['tot']:7.2f}{m['ann']*100:+7.0f}%{m['sharpe']:7.2f}"
                  f"{m['mdd']*100:7.0f}%{m['win']*100:6.0f}%{bt['turnover'].mean()*100:5.0f}%")
    combo([(c, d) for _, c, _, _, d in RAW], "组合·全因子")
    combo([("AMP_lo_f",-1),("VRCORR_f",-1),("APM_f",-1)], "组合·价格3")
    combo([("OFI_f",1),("OFIbig_f",1),("AMTPT_f",1)], "组合·T2资金流3")
    print("=" * 80)

    out = os.path.join(os.path.dirname(__file__), "equity_binance.csv")
    rows = [{"factor":k,"day":pd.Timestamp(dd).date(),"equity":e}
            for k,(days,es) in eq.items() for dd,e in zip(days,es)]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"净值曲线已存: {out}")


if __name__ == "__main__":
    main()
