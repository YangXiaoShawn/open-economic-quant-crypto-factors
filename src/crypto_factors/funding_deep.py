# -*- coding: utf-8 -*-
"""
深挖 funding 因子 —— 它是最强(夏普1.94)、最独立(与价量相关<5%)、唯一三口径(多空/计费/纯多头)
全活的信号。两个角度榨干它:

A. 横截面条件双重排序: funding × 价格状态。
   假设: "高 funding 且已超买"= 拥挤多头+技术超买 → 反转最猛, 是最该做空的角; 反过来
   "低 funding 且超卖"是最该做多的角。比纯 funding 单排序更锋利吗?

B. 时序择时: 全市场平均 funding 作 risk-on/off 开关。
   假设: 全市场 funding 极高 = 杠杆多头拥挤 → 大盘见顶风险 → 减仓/做空; 极低/负 = 出清 → 加仓。
   用**滚动 z-score (trailing) 防前视**, 与买入持有对比。
"""
import os
import numpy as np
import pandas as pd
from backtest import metrics, QUANTILE, COST_BPS
from backtest_perp import build_panel, BIG_POOL

ANN = 365


def _qf(s, q):                      # 横截面分位 (0..q-1)
    return pd.qcut(s.rank(method="first"), q, labels=False)


# ============================================================ A. 条件双重排序
def conditional_sort(panel):
    big = panel[panel["sym"].isin(BIG_POOL)].sort_values(["sym", "day"]).copy()
    big["RET5"] = big.groupby("sym")["close"].transform(lambda x: x / x.shift(5) - 1.0)
    big = big.dropna(subset=["FUND_f", "RET5", "fwd_ret"])

    # --- 3×3 独立双重排序: 看次日收益的交互格 ---
    big["fq"] = big.groupby("day")["FUND_f"].transform(lambda s: _qf(s, 3))
    big["rq"] = big.groupby("day")["RET5"].transform(lambda s: _qf(s, 3))
    grid = (big.groupby(["fq", "rq"])["fwd_ret"].mean() * ANN * 100).unstack()
    grid.index = ["funding低", "funding中", "funding高"]
    grid.columns = ["近收益低(超卖)", "近收益中", "近收益高(超买)"]
    print(f"\n{'='*78}\nA. 条件双重排序: funding × 近5日收益 → 次日收益(年化%)  (大币池24)\n{'='*78}")
    print(grid.round(0).astype(int).to_string())
    print("解读: 看右下角(高funding+超买)是否最差(最该空), 左上角(低funding+超卖)是否最好(最该多)。")

    # --- 2×2 角策略 (中位数切, 每角~6币, 更稳) vs 纯 funding 单排序 ---
    def run(mode):
        b = big.copy()
        b["fh"] = b.groupby("day")["FUND_f"].transform(lambda s: (s > s.median()).astype(int))
        b["rh"] = b.groupby("day")["RET5"].transform(lambda s: (s > s.median()).astype(int))
        b["fqv"] = b.groupby("day")["FUND_f"].transform(lambda s: _qf(s, QUANTILE))
        rows, pl, ps = [], set(), set()
        for day, g in b.groupby("day"):
            if g["sym"].nunique() < QUANTILE * 2:
                continue
            if mode == "cond":     # 条件角: 多=低funding&超卖, 空=高funding&超买
                L = g[(g.fh == 0) & (g.rh == 0)]; S = g[(g.fh == 1) & (g.rh == 1)]
            else:                  # 纯 funding 五分位
                L = g[g.fqv == 0]; S = g[g.fqv == QUANTILE - 1]
            if len(L) == 0 or len(S) == 0:
                continue
            cl, cs = set(L["sym"]), set(S["sym"])
            to = (len(cl ^ pl) + len(cs ^ ps)) / max(len(cl) + len(cs), 1)
            ff = lambda d: (0.0 if pd.isna(d["fwd_fund"].mean()) else d["fwd_fund"].mean())
            fund_pnl = ff(S) - ff(L)            # 空腿收, 多腿付 (计提资金费)
            rows.append((L["fwd_ret"].mean() - S["fwd_ret"].mean())
                        - to * COST_BPS / 1e4 + fund_pnl)
            pl, ps = cl, cs
        return np.array(rows)

    print(f"\n{'多空策略(计funding)':22s}{'净值':>7s}{'年化':>8s}{'夏普':>7s}{'回撤':>7s}{'胜率':>6s}")
    print("-" * 60)
    for mode, nm in [("plain", "纯 funding 五分位"), ("cond", "funding×超买 条件角")]:
        m = metrics(run(mode))
        print(f"{nm:22s}{1+m['tot']:7.2f}{m['ann']*100:+7.0f}%{m['sharpe']:+7.2f}"
              f"{m['mdd']*100:6.0f}%{m['win']*100:5.0f}%")


# ============================================================ B. 时序择时
def funding_timing(panel):
    big = panel[panel["sym"].isin(BIG_POOL)]
    agg = big.groupby("day")["FUND_f"].mean()               # 全市场拥挤度(大币平均funding)
    mkt = panel.groupby("day")["fwd_ret"].mean()            # 大盘次日收益(全币等权)
    df = pd.concat({"agg": agg, "mkt": mkt}, axis=1).dropna().sort_index()

    print(f"\n{'='*78}\nB. 时序择时: 全市场平均 funding → 次日大盘收益(年化%)  (按当日 funding 分5档)\n{'='*78}")
    df["aq"] = _qf(df["agg"], 5)
    tab = df.groupby("aq")["mkt"].agg(["mean", "count"])
    for q in range(5):
        lab = ["最低(出清)", "低", "中", "高", "最高(拥挤)"][q]
        r = tab.loc[q]
        print(f"  funding {lab:10s}: 次日大盘 {r['mean']*ANN*100:+6.0f}%/年   ({int(r['count'])}天)")

    # 择时: 滚动60日 z-score, funding 越高越减仓/做空 (用截至当日信息, 无前视)
    z = (df["agg"] - df["agg"].rolling(60, min_periods=20).mean()) / \
        (df["agg"].rolling(60, min_periods=20).std() + 1e-12)
    w = (-z).clip(-1, 1)                                    # 高funding→空, 低→多
    df["timed"] = (w * df["mkt"]).fillna(0)
    df["bh"] = df["mkt"]
    df["long_flat"] = (np.where(w > 0, 1.0, 0.0) * df["mkt"])  # 只多/空仓(现货可做)

    print(f"\n{'大盘策略':18s}{'净值':>7s}{'年化':>8s}{'夏普':>7s}{'回撤':>7s}{'胜率':>6s}")
    print("-" * 56)
    for col, nm in [("bh", "买入持有"), ("timed", "funding择时(多空)"),
                    ("long_flat", "funding择时(多/空仓)")]:
        m = metrics(df[col].values)
        print(f"{nm:18s}{1+m['tot']:7.2f}{m['ann']*100:+7.0f}%{m['sharpe']:+7.2f}"
              f"{m['mdd']*100:6.0f}%{m['win']*100:5.0f}%")
    return df


def main():
    panel = build_panel()
    conditional_sort(panel)
    df = funding_timing(panel)
    out = os.path.join(os.path.dirname(__file__), "funding_timing.csv")
    df[["agg", "bh", "timed", "long_flat"]].to_csv(out)
    print(f"\n时序择时序列已存: {out}\n{'='*78}")


if __name__ == "__main__":
    main()
