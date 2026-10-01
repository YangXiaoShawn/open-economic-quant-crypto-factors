# -*- coding: utf-8 -*-
"""
聪明钱因子 Q 的 crypto 复现  (开源证券·微观结构系列(3)《聪明钱因子模型2.0》)

构造 (单币种·单 UTC 日，用 1m bars)：
  1. 每分钟"聪明度"  S_t = |R_t| / sqrt(V_t)      R_t=分钟收益, V_t=分钟成交量
  2. 按 S_t 降序，累计成交量达当日 20% 的分钟 = "聪明钱时段"
  3. VWAP_smart = Σ(P·V) / ΣV  (仅聪明时段)
     VWAP_all   = Σ(P·V) / ΣV  (全日)
  4. Q = VWAP_smart / VWAP_all
     Q>1: 聪明钱在高位活跃(看空)  Q<1: 在低位活跃(看多)  -> 预期 IC 为负
  5. 因子 = 过去 N 天 Q 的均值 (2.0 平滑版)

检验：每个 UTC 日，对全币种横截面，算 因子 vs 次日收益 的 RankIC。
"""
import os, glob
import numpy as np
import pandas as pd

DATADIR = os.path.join(os.path.dirname(__file__), "data")
SMART_FRAC = 0.20     # 聪明钱成交量占比阈值
SMOOTH_N   = 5        # Q 平滑天数 (回看窗口短，用5)


def daily_Q(g):
    """对单币种单日的分钟 df 计算 Q。"""
    g = g.sort_values("ts")
    p = g["close"].to_numpy(dtype=float)
    v = g["volume"].to_numpy(dtype=float)
    if len(g) < 30 or v.sum() <= 0:
        return np.nan
    r = np.zeros_like(p)
    r[1:] = p[1:] / p[:-1] - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.abs(r) / np.sqrt(np.where(v > 0, v, np.nan))
    order = np.argsort(-np.nan_to_num(s, nan=-1))      # 聪明度降序
    cumv = np.cumsum(v[order])
    thr = SMART_FRAC * v.sum()
    k = np.searchsorted(cumv, thr) + 1                 # 取到累计达20%的分钟
    smart = order[:max(k, 1)]
    vwap_smart = (p[smart] * v[smart]).sum() / v[smart].sum()
    vwap_all   = (p * v).sum() / v.sum()
    return vwap_smart / vwap_all


def main():
    files = glob.glob(os.path.join(DATADIR, "*.parquet"))
    if not files:
        print("没有数据，请先运行 fetch_okx.py")
        return
    print(f"读取 {len(files)} 个币种…")

    qrecs, prets = [], []
    for fn in files:
        sym = os.path.basename(fn).replace(".parquet", "").replace("_", "/")
        df = pd.read_parquet(fn)
        df["day"] = df["dt"].dt.floor("D")
        # 每日 Q
        q = df.groupby("day").apply(daily_Q, include_groups=False).rename("Q")
        # 每日收盘 (当日最后一根分钟 close) -> 次日收益
        last = df.groupby("day")["close"].last()
        fwd = last.shift(-1) / last - 1.0
        d = pd.concat([q, fwd.rename("fwd_ret")], axis=1)
        d["sym"] = sym
        # Q 平滑 (2.0 版)
        d["Qsm"] = d["Q"].rolling(SMOOTH_N, min_periods=2).mean()
        qrecs.append(d.reset_index())

    panel = pd.concat(qrecs, ignore_index=True).dropna(subset=["Qsm", "fwd_ret"])

    # 横截面 RankIC：每个 day 至少 8 个币种
    ics = []
    for day, gg in panel.groupby("day"):
        if len(gg) < 8:
            continue
        ic = gg["Qsm"].corr(gg["fwd_ret"], method="spearman")
        ics.append((day, ic, len(gg)))
    icdf = pd.DataFrame(ics, columns=["day", "ic", "n"]).dropna()

    print("\n================ 聪明钱因子 Q  横截面检验 (crypto/OKX) ================")
    print(f"样本: {panel['sym'].nunique()} 币种 × {icdf['day'].nunique()} 天, "
          f"横截面样本 {len(panel)} 条")
    if len(icdf) == 0:
        print("有效横截面不足。")
        return
    mean_ic = icdf["ic"].mean()
    std_ic  = icdf["ic"].std()
    icir    = mean_ic / std_ic * np.sqrt(len(icdf)) if std_ic > 0 else np.nan
    print(f"日均 RankIC   = {mean_ic:+.4f}")
    print(f"IC 标准差     = {std_ic:.4f}")
    print(f"IC>0 占比     = {(icdf['ic'] > 0).mean():.1%}")
    print(f"ICIR (t值)    = {icir:+.2f}")
    print(f"  -> 预期 Q 越低(聪明钱低位)次日越涨, 即 IC 为负方向有效")

    # 分组检验: 按 Qsm 分 3 组看次日平均收益
    panel["grp"] = panel.groupby("day")["Qsm"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=["低Q","中","高Q"])
        if x.nunique() >= 3 else np.nan)
    grp = panel.groupby("grp", observed=True)["fwd_ret"].mean() * 100
    print("\n按 Qsm 三分组的次日平均收益 (%):")
    for k in ["低Q","中","高Q"]:
        if k in grp.index:
            print(f"  {k:4s}: {grp[k]:+.3f}%")
    if "低Q" in grp.index and "高Q" in grp.index:
        print(f"  多空(低Q-高Q): {grp['低Q']-grp['高Q']:+.3f}% / 日")
    print("=" * 64)


if __name__ == "__main__":
    main()
