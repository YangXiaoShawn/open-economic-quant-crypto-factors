# -*- coding: utf-8 -*-
"""
capstone 回测: 扩 perp 全币种 + 资金流限大币池 + funding/OI 叠加 + 约束优化组合层。

把仓库三条已验证结论组装成一套可部署的横截面策略:
  1) 价格类因子 (理想反转/动量/振幅切割) 在**宽池**(全 perp ~100 币) 上跑 —— 广度增强 alpha;
  2) T2 资金流 + funding + OI 类因子在**大币池**(24 大市值) 上跑 —— 这些 alpha 只在大币稳定;
  3) 组合层用 **约束优化 (max_sharpe / min_var / risk_parity, walk-forward)** 配权, 降回撤升夏普。

funding/OI 是 A 股没有的衍生品资金流维度 (perp 独有):
  FUND  资金费率 (多头拥挤/carry, 反向)        GLS  全市场账户多空比 (散户拥挤, 反向)
  dOI   持仓量变化 (杠杆增减)                   TLS  大户持仓多空比 (聪明钱, 顺势)
"""
import os, glob
import numpy as np
import pandas as pd
from backtest import backtest_factor, metrics
import portfolio_opt as po

ROOT = os.path.dirname(__file__)
KDIR = os.path.join(ROOT, "data_binance")
FUNDDIR = os.path.join(ROOT, "data_perp", "funding")
OIDIR = os.path.join(ROOT, "data_perp", "oi")
N = 5

BIG_POOL = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT",
            "ADAUSDT","AVAXUSDT","LINKUSDT","DOTUSDT","TRXUSDT","LTCUSDT",
            "BCHUSDT","NEARUSDT","APTUSDT","ARBUSDT","OPUSDT","FILUSDT",
            "ATOMUSDT","UNIUSDT","AAVEUSDT","INJUSDT","ETCUSDT","XLMUSDT"]


def build_panel():
    # ---- 价量 + T2 资金流 (现货/perp 1m 聚合) ----
    recs = []
    for fn in glob.glob(os.path.join(KDIR, "*.parquet")):
        d = pd.read_parquet(fn).sort_values("day").reset_index(drop=True)
        d["fwd_ret"] = d["close"].shift(-1) / d["close"] - 1.0
        d["REV_hi_f"] = d["REV_hi"].rolling(N, min_periods=2).sum()
        d["AMP_lo_f"] = d["AMP_lo"].rolling(N, min_periods=2).mean()
        d["MOM_f"]    = d["close"] / d["close"].shift(10) - 1.0
        d["OFI_f"]    = d["OFI"].rolling(N, min_periods=2).mean()
        d["OFIbig_f"] = d["OFI_big"].rolling(N, min_periods=2).mean()
        recs.append(d)
    panel = pd.concat(recs, ignore_index=True)
    panel["day"] = pd.to_datetime(panel["day"], utc=True)

    # ---- funding (perp 独有) ----
    fund = [pd.read_parquet(f) for f in glob.glob(os.path.join(FUNDDIR, "*.parquet"))]
    if fund:
        fd = pd.concat(fund, ignore_index=True)
        fd["day"] = pd.to_datetime(fd["day"], utc=True)
        fd = fd.sort_values(["sym","day"])
        fd["FUND_f"] = fd.groupby("sym")["FUND_sum"].transform(
            lambda x: x.rolling(N, min_periods=2).mean())
        # 持有期(t→t+1)真实计提的资金费 = 次日 funding, 与 fwd_ret 对齐
        fd["fwd_fund"] = fd.groupby("sym")["FUND_sum"].shift(-1)
        panel = panel.merge(fd[["sym","day","FUND_f","fwd_fund"]], on=["sym","day"], how="left")

    # ---- OI / 多空比 (perp 独有, 仅大币池+限窗) ----
    oi = [pd.read_parquet(f) for f in glob.glob(os.path.join(OIDIR, "*.parquet"))]
    if oi:
        od = pd.concat(oi, ignore_index=True)
        od["day"] = pd.to_datetime(od["day"], utc=True)
        od = od.sort_values(["sym","day"])
        g = od.groupby("sym")
        od["dOI_f"] = g["OI_close"].transform(lambda x: x / x.shift(3) - 1.0)   # 3日OI变化
        od["GLS_f"] = g["GLS"].transform(lambda x: x.rolling(N, min_periods=2).mean())
        od["TLS_f"] = g["TLS"].transform(lambda x: x.rolling(N, min_periods=2).mean())
        panel = panel.merge(od[["sym","day","dOI_f","GLS_f","TLS_f"]],
                            on=["sym","day"], how="left")
    return panel


# (因子列, 名称, 方向, 池, 类型)
FACTORS = [
    ("REV_hi_f", "理想反转",    -1, "wide", "价格"),
    ("MOM_f",    "截面动量",    +1, "wide", "价格"),
    ("AMP_lo_f", "低价位振幅",  -1, "wide", "价格"),
    ("OFI_f",    "主动买入率",  +1, "big",  "T2资金流"),
    ("OFIbig_f", "大额主动买",  +1, "big",  "T2资金流"),
    ("FUND_f",   "资金费率",    -1, "big",  "funding★"),
    ("dOI_f",    "持仓量变化",  +1, "big",  "OI★"),
    ("GLS_f",    "散户多空比",  -1, "big",  "OI★"),
    ("TLS_f",    "大户多空比",  +1, "big",  "OI★"),
]


def pool_panel(panel, pool):
    return panel if pool == "wide" else panel[panel["sym"].isin(BIG_POOL)].copy()


def pool_sweep(panel):
    """同一因子在 大币(24)/中池(60)/宽池(105) 上的夏普 —— 论证'每个因子有各自最优广度'。"""
    try:
        from binance_pipeline import UNIVERSE as SPOT60
    except Exception:
        SPOT60 = None
    pools = {"大币池24": BIG_POOL}
    if SPOT60:
        pools["中池60"] = SPOT60
    pools["宽池105"] = sorted(panel["sym"].unique())
    facs = [("REV_hi_f","理想反转",-1), ("AMP_lo_f","低价位振幅",-1), ("MOM_f","截面动量",+1),
            ("OFIbig_f","大额主动买",+1), ("FUND_f","资金费率",-1)]
    print(f"\n{'='*80}\n[扩池广度扫描] 每个因子在不同币池广度上的夏普 (资金流为何要限池)\n{'='*80}")
    print(f"{'因子':12s}" + "".join(f"{p:>10s}" for p in pools))
    print("-" * 80)
    for col, nm, d in facs:
        if col not in panel.columns:
            continue
        row = f"{nm:12s}"
        for p, syms in pools.items():
            sub = panel[panel["sym"].isin(syms)]
            m = metrics(backtest_factor(sub, col, d)["ls"])
            row += f"{(m['sharpe'] if m else float('nan')):>10.2f}"
        print(row)
    print("解读: 反转/动量峰值在中池(过多垃圾perp反伤); 低价位振幅随广度单调走强; "
          "\n      taker资金流跨池不稳(0.73/0.10/1.62)->只信大币池; funding 各池皆强->唯一可泛化的资金流。")


def ls_series(panel, col, d, pool):
    bt = backtest_factor(pool_panel(panel, pool), col, d)
    return pd.Series(bt["ls"].values, index=pd.to_datetime(bt["day"].values)), bt


def show(m, label, extra=""):
    print(f"{label:14s}{1+m['tot']:6.2f}{m['ann']*100:+7.0f}%{m['sharpe']:+6.2f}"
          f"{m['mdd']*100:6.0f}%{m['win']*100:5.0f}%  {extra}")


def main():
    panel = build_panel()
    nc_wide = panel["sym"].nunique()
    nc_big = panel[panel["sym"].isin(BIG_POOL)]["sym"].nunique()
    nd = panel["day"].nunique()
    print(f"\n{'='*80}")
    print(f"capstone perp 回测  宽池 {nc_wide} 币 / 大币池 {nc_big} 币 × {nd} 天  五分位多空 5bps")
    print(f"  价格因子=宽池(全perp)  资金流/funding/OI=大币池  (★=A股没有的perp衍生品维度)")
    print(f"{'='*80}")
    print(f"{'因子':14s}{'净值':>6s}{'年化':>8s}{'夏普':>6s}{'回撤':>6s}{'胜率':>5s}  类型/池")
    print("-" * 80)

    series = {}
    for col, name, d, pool, typ in FACTORS:
        if col not in panel.columns:
            print(f"{name:14s}  (无数据, 跳过)"); continue
        s, bt = ls_series(panel, col, d, pool)
        m = metrics(s.values)
        if not m:
            print(f"{name:14s}  样本不足"); continue
        series[name] = s
        show(m, name, f"{typ} · {'宽池' if pool=='wide' else '大币池'}({bt['ls'].size}d)")
    print("-" * 80)

    pool_sweep(panel)

    R = pd.DataFrame(series)

    # ---- 相关矩阵 (跨类低相关才有分散价值) ----
    print("\n[因子收益流 相关矩阵 %] (funding/OI 与价量/资金流是否独立?)")
    print((R.corr() * 100).round(0).fillna(0).astype(int).to_string())

    # ---- 组合层: 全窗 (价格+资金流+funding, 不含OI=可用730天) ----
    full_cols = ["理想反转","大额主动买","资金费率"]
    full_cols = [c for c in full_cols if c in R.columns]
    Rf = R[full_cols].dropna()
    print(f"\n{'='*80}\n[组合层 A] 价格×资金流×funding  ({len(Rf)}天, {' + '.join(full_cols)})\n{'='*80}")
    print(f"{'方法':14s}{'净值':>6s}{'年化':>8s}{'夏普':>6s}{'回撤':>6s}{'胜率':>5s}")
    print("-" * 80)
    _combo_block(Rf)

    # ---- 组合层: 含 funding+OI 的全维度 (限 OI 窗口) ----
    allcols = [c for c in ["理想反转","大额主动买","资金费率","持仓量变化","大户多空比"] if c in R.columns]
    Ra = R[allcols].dropna()
    if len(Ra) > 60:
        print(f"\n{'='*80}\n[组合层 B] +OI/多空全维度  ({len(Ra)}天, {' + '.join(allcols)})\n{'='*80}")
        print(f"{'方法':14s}{'净值':>6s}{'年化':>8s}{'夏普':>6s}{'回撤':>6s}{'胜率':>5s}")
        print("-" * 80)
        _combo_block(Ra)

    # 保存净值
    out = os.path.join(ROOT, "equity_perp.csv")
    eq = pd.DataFrame({n: (1 + R[n].fillna(0)).cumprod() for n in R.columns})
    eq.to_csv(out)
    print(f"\n净值曲线已存: {out}\n{'='*80}")


def _combo_block(R):
    bench = {}
    ew = po.equal_weight(R);  bench["等权"] = ew
    iv = po.inverse_vol(R);   bench["逆波动"] = iv
    for label, s in bench.items():
        show(metrics(s.values), label)
    for method, nm in [("max_sharpe","约束·最大夏普"), ("min_var","约束·最小方差"),
                       ("risk_parity","约束·风险平价")]:
        ret, w = po.walk_forward(R, method=method, lookback=120, rebal=20, w_max=0.6)
        show(metrics(ret.values), nm, f"末权重 {dict(zip(R.columns, w.iloc[-1].round(2)))}")


if __name__ == "__main__":
    main()
