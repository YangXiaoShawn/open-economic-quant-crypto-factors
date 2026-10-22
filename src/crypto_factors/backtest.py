# -*- coding: utf-8 -*-
"""
逐篇研报回测 —— 把每个微观结构因子做成横截面多空组合，给出净值/年化/夏普/回撤。

口径:
  - 每个 UTC 日: 按因子排序，做 分位组合 (默认五分位)。
  - 方向按"研报预期"定向(反转类 long低 short高)，故展示的是"按原文方向交易"的结果。
  - 多空 = 多组等权 − 空组等权; 多头 = 仅多组(贴近实盘只做多)。
  - 成本: 单边 COST_BPS, 按当日换手计提。
  - 7×24, 每日调仓; 年化按 365 天。
"""
import os, warnings
import numpy as np
import pandas as pd
from factor_library import build_panel
warnings.filterwarnings("ignore", message="Mean of empty slice")

QUANTILE = 5          # 分位数
COST_BPS = 5.0        # 单边成本(基点)
ANN = 365             # crypto 全年交易

# (因子列, 名称, 研报号, 方向)  dir=-1: 反转(long低), +1: 动量(long高)
FACTORS = [
    ("Q_f",      "聪明钱Q",     "(3)",    -1),
    ("REV_hi_f", "理想反转hi",  "(1,13)", -1),
    ("APM_f",    "APM",         "(5)",    -1),
    ("AMP_f",    "振幅",        "(7)",    -1),
    ("AMP_lo_f", "低价位振幅",  "(30)",   -1),
    ("SKEW_f",   "收益偏度",    "(微结构)", -1),
    ("AC1_f",    "日内自相关",  "(19)",   -1),
    ("VCONC_f",  "量集中度",    "(27)",   -1),
    ("VRCORR_f", "量波同步",    "(27)",   -1),
]


def backtest_factor(panel, col, direction):
    """返回每日 多空/多头 收益序列 + 换手。"""
    df = panel.dropna(subset=[col, "fwd_ret"]).copy()
    df["sig"] = df[col] * direction          # 按方向转成"越大越看多"
    ls_ret, long_ret, days, turns = [], [], [], []
    prev_long, prev_short = set(), set()
    for day, g in df.groupby("day"):
        if g["sym"].nunique() < QUANTILE * 2:
            continue
        g = g.copy()
        g["q"] = pd.qcut(g["sig"].rank(method="first"), QUANTILE, labels=False)
        longs = g[g["q"] == QUANTILE - 1]      # 信号最高(看多)
        shorts = g[g["q"] == 0]                # 信号最低(看空)
        lr, sr = longs["fwd_ret"].mean(), shorts["fwd_ret"].mean()
        # 换手(多头腿)
        cur_long = set(longs["sym"]); cur_short = set(shorts["sym"])
        to = (len(cur_long ^ prev_long) + len(cur_short ^ prev_short)) / \
             max(len(cur_long) + len(cur_short), 1)
        cost = to * COST_BPS / 1e4
        ls_ret.append((lr - sr) - cost)
        long_ret.append(lr - (len(cur_long ^ prev_long) / max(len(cur_long),1)) * COST_BPS/1e4)
        days.append(day); turns.append(to)
        prev_long, prev_short = cur_long, cur_short
    return pd.DataFrame({"day": days, "ls": ls_ret, "long": long_ret, "turnover": turns})


def metrics(r):
    r = np.asarray(r, float)
    if len(r) < 5:
        return None
    cum = np.prod(1 + r) - 1
    ann_ret = (1 + cum) ** (ANN / len(r)) - 1
    ann_vol = r.std() * np.sqrt(ANN)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    eq = np.cumprod(1 + r)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return dict(ann=ann_ret, vol=ann_vol, sharpe=sharpe, mdd=dd,
                win=(r > 0).mean(), tot=cum)


def main():
    panel = build_panel()
    ndays = panel["day"].nunique()
    print(f"\n{'='*78}")
    print(f"逐篇研报·crypto 横截面多空回测  ({panel['sym'].nunique()} 币种 × {ndays} 天, "
          f"五分位, 单边{COST_BPS:.0f}bps, 日调仓)")
    print(f"{'='*78}")
    print(f"{'研报因子':16s}{'净值':>7s}{'年化':>9s}{'夏普':>7s}"
          f"{'最大回撤':>9s}{'日胜率':>8s}{'换手':>7s}")
    print("-" * 78)
    eq_store = {}
    for col, name, rpt, d in FACTORS:
        bt = backtest_factor(panel, col, d)
        m = metrics(bt["ls"])
        if not m:
            print(f"{name+rpt:16s}  样本不足")
            continue
        eq_store[name] = (bt["day"].values, np.cumprod(1 + bt["ls"].values))
        print(f"{name+rpt:16s}{1+m['tot']:7.2f}{m['ann']*100:+8.1f}%{m['sharpe']:7.2f}"
              f"{m['mdd']*100:8.1f}%{m['win']*100:7.1f}%{bt['turnover'].mean()*100:6.0f}%")
    print("-" * 78)

    def combo(cols_dirs, label):
        sub = panel.copy()
        zs = []
        for col, d in cols_dirs:
            z = sub.groupby("day")[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
            zs.append(z.values * d)
        stack = np.vstack(zs)
        sub["combo_f"] = np.where(np.isnan(stack).all(0), np.nan, np.nanmean(stack, axis=0))
        bt = backtest_factor(sub, "combo_f", +1)   # 已定向
        m = metrics(bt["ls"])
        if m:
            eq_store[label] = (bt["day"].values, np.cumprod(1 + bt["ls"].values))
            print(f"{label:16s}{1+m['tot']:7.2f}{m['ann']*100:+8.1f}%{m['sharpe']:7.2f}"
                  f"{m['mdd']*100:8.1f}%{m['win']*100:7.1f}%{bt['turnover'].mean()*100:6.0f}%")

    # 全因子 naive 等权 (含失效因子, 会被拖累)
    combo([(c, d) for c, n, r, d in FACTORS], "组合·全因子")
    # 事后选"已迁移"的3个强因子 (仅示意, 含前视选择)
    combo([("VRCORR_f", -1), ("APM_f", -1), ("AMP_lo_f", -1)], "组合·选强3*")
    print("=" * 78)
    print("说明: 净值=期末/期初; 方向按原文(反转类long低分位); 已扣单边5bps成本")

    # 保存净值曲线
    out = os.path.join(os.path.dirname(__file__), "equity_curves.csv")
    rows = []
    for name, (days, eq) in eq_store.items():
        for dd, e in zip(days, eq):
            rows.append({"factor": name, "day": pd.Timestamp(dd).date(), "equity": e})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"净值曲线已存: {out}")


if __name__ == "__main__":
    main()
