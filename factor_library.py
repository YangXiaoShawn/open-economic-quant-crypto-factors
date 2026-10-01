# -*- coding: utf-8 -*-
"""
开源·微观结构系列 —— 多个"仅需分钟OHLCV"的因子在 crypto 上的横截面复现。
一次扫描算出每个 (币种, UTC日) 的日级特征，再组面板做 RankIC 检验。

实现的因子（括号为开源系列报告号）：
  Q       聪明钱因子 Q                         (3)   预期反转, IC<0
  REV_hi  理想反转·高成交时段收益累计          (1,13) 预期反转, IC<0
  REV_lo  理想反转·低成交时段收益累计(对照)    (1,13)
  APM     上/下半日收益差(crypto版APM)         (5)   预期动量耗尽, IC<0
  AMP     日内振幅(高-低)/收盘 累计            (7)   振幅反转/低波, IC<0
  AMP_lo  低价位时段振幅(振幅内部切割)         (30)
  SKEW    分钟收益偏度累计                     (微结构)  彩票偏好, IC<0
"""
import os, glob
import numpy as np
import pandas as pd

DATADIR  = os.path.join(os.path.dirname(__file__), "data")
SMART_FRAC = 0.20      # 聪明钱量占比
TOP_FRAC   = 0.30      # 理想反转 高/低成交时段占比
SMOOTH_N   = 5         # 因子平滑/累计天数


def day_features(g):
    """单币种单 UTC 日的分钟 df -> 日级特征字典。"""
    g = g.sort_values("ts")
    p = g["close"].to_numpy(float)
    v = g["volume"].to_numpy(float)
    h = g["high"].to_numpy(float)
    lo = g["low"].to_numpy(float)
    hour = g["dt"].dt.hour.to_numpy()
    n = len(g)
    if n < 60 or v.sum() <= 0 or p[0] <= 0:
        return None
    r = np.zeros(n); r[1:] = p[1:] / p[:-1] - 1.0
    amt = p * v                                    # 分钟成交额≈turnover
    out = {}

    # --- 聪明钱 Q (3) ---
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.abs(r) / np.sqrt(np.where(v > 0, v, np.nan))
    order = np.argsort(-np.nan_to_num(s, nan=-1))
    k = np.searchsorted(np.cumsum(v[order]), SMART_FRAC * v.sum()) + 1
    sm = order[:max(k, 1)]
    out["Q"] = (p[sm]*v[sm]).sum()/v[sm].sum() / ((p*v).sum()/v.sum())

    # --- 理想反转 高/低成交时段收益 (1,13) ---
    aorder = np.argsort(-amt)                       # 成交额降序
    ktop = max(int(TOP_FRAC * n), 1)
    hi_idx = aorder[:ktop]                          # 高成交分钟
    lo_idx = aorder[-ktop:]                         # 低成交分钟
    out["REV_hi"] = r[hi_idx].sum()
    out["REV_lo"] = r[lo_idx].sum()

    # --- APM: 上/下半日收益 (5) ---
    am = hour < 12
    out["r_am"] = r[am].sum()
    out["r_pm"] = r[~am].sum()

    # --- 振幅 + 价位切割 (7,30) ---
    amp_min = (h - lo) / np.where(p > 0, p, np.nan)
    out["AMP"] = np.nanmean(amp_min)
    med = np.median(p)
    lowp = p <= med                                 # 低价位时段
    out["AMP_lo"] = np.nanmean(amp_min[lowp])

    # --- 分钟收益偏度 (彩票偏好) ---
    out["SKEW"] = pd.Series(r[1:]).skew()

    # --- 日内分钟收益时序自相关 (19) 日内动量/反转结构 ---
    rr = r[1:]
    if len(rr) > 10 and rr.std() > 0:
        out["AC1"] = pd.Series(rr).autocorr(lag=1)
    else:
        out["AC1"] = np.nan

    # --- 高频成交量峰岭谷: 成交量集中度 (27) ---
    # 峰=少数分钟集中放量. 用前10%分钟的成交量占比衡量"峰"的尖锐度
    vs = np.sort(v)[::-1]
    ktop = max(int(0.10 * n), 1)
    out["VCONC"] = vs[:ktop].sum() / v.sum()
    # 量-波动同步性: 放量分钟是否伴随大波动 (峰的"信息含量")
    if np.std(v) > 0 and np.std(np.abs(r)) > 0:
        out["VRCORR"] = np.corrcoef(v, np.abs(r))[0, 1]
    else:
        out["VRCORR"] = np.nan

    out["close"] = p[-1]
    return out


def build_panel():
    files = glob.glob(os.path.join(DATADIR, "*.parquet"))
    recs = []
    for fn in files:
        sym = os.path.basename(fn).replace(".parquet", "").replace("_", "/")
        df = pd.read_parquet(fn)
        df["day"] = df["dt"].dt.floor("D")
        rows = []
        for day, g in df.groupby("day"):
            f = day_features(g)
            if f:
                f["day"] = day; f["sym"] = sym
                rows.append(f)
        if not rows:
            continue
        d = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
        # 次日收益
        d["fwd_ret"] = d["close"].shift(-1) / d["close"] - 1.0
        # 累计/平滑型因子（trailing 内 coin）
        d["Q_f"]      = d["Q"].rolling(SMOOTH_N, min_periods=2).mean()
        d["REV_hi_f"] = d["REV_hi"].rolling(SMOOTH_N, min_periods=2).sum()
        d["REV_lo_f"] = d["REV_lo"].rolling(SMOOTH_N, min_periods=2).sum()
        d["APM_f"]    = (d["r_pm"] - d["r_am"]).rolling(SMOOTH_N, min_periods=2).mean()
        d["AMP_f"]    = d["AMP"].rolling(SMOOTH_N, min_periods=2).mean()
        d["AMP_lo_f"] = d["AMP_lo"].rolling(SMOOTH_N, min_periods=2).mean()
        d["SKEW_f"]   = d["SKEW"].rolling(SMOOTH_N, min_periods=2).mean()
        d["AC1_f"]    = d["AC1"].rolling(SMOOTH_N, min_periods=2).mean()
        d["VCONC_f"]  = d["VCONC"].rolling(SMOOTH_N, min_periods=2).mean()
        d["VRCORR_f"] = d["VRCORR"].rolling(SMOOTH_N, min_periods=2).mean()
        recs.append(d)
    return pd.concat(recs, ignore_index=True)


def ic_report(panel, factors):
    print(f"\n{'='*70}")
    print(f"crypto 微观结构因子 横截面 RankIC  "
          f"({panel['sym'].nunique()} 币种 × {panel['day'].nunique()} 天)")
    print(f"{'='*70}")
    print(f"{'因子':10s}{'日均IC':>9s}{'ICIR(t)':>9s}{'IC>0%':>8s}"
          f"{'多空%/日':>10s}  方向")
    print("-" * 70)
    for col, name, expect in factors:
        sub = panel.dropna(subset=[col, "fwd_ret"])
        ics = []
        for day, gg in sub.groupby("day"):
            if len(gg) < 8:
                continue
            ic = gg[col].corr(gg["fwd_ret"], method="spearman")
            if pd.notna(ic):
                ics.append(ic)
        if len(ics) < 5:
            print(f"{name:10s}   样本不足")
            continue
        ics = np.array(ics)
        icir = ics.mean() / ics.std() * np.sqrt(len(ics)) if ics.std() > 0 else np.nan
        # 五分组多空(按因子升序: 最低组 - 最高组)
        sub = sub.copy()
        sub["g"] = sub.groupby("day")[col].transform(
            lambda x: pd.qcut(x.rank(method="first"), 5, labels=False)
            if x.nunique() >= 5 else np.nan)
        gm = sub.groupby("g", observed=True)["fwd_ret"].mean()
        ls = (gm.get(0, np.nan) - gm.get(4, np.nan)) * 100
        print(f"{name:10s}{ics.mean():+9.4f}{icir:+9.2f}{(ics>0).mean()*100:7.1f}%"
              f"{ls:+10.3f}  {expect}")
    print("=" * 70)
    print("注: IC<0 = 反转有效; 多空=(最低因子组−最高因子组)次日收益")


def main():
    panel = build_panel()
    factors = [
        ("Q_f",      "聪明钱Q",     "期望反转IC<0"),
        ("REV_hi_f", "理想反转hi",  "期望反转IC<0"),
        ("REV_lo_f", "反转lo对照",  "—"),
        ("APM_f",    "APM",         "期望IC<0"),
        ("AMP_f",    "振幅",        "期望反转IC<0"),
        ("AMP_lo_f", "低价位振幅",  "切割对照"),
        ("SKEW_f",   "收益偏度",    "彩票IC<0"),
        ("AC1_f",    "日内自相关",  "时序(19)"),
        ("VCONC_f",  "量集中度",    "峰岭谷(27)"),
        ("VRCORR_f", "量波同步",    "峰岭谷(27)"),
    ]
    ic_report(panel, factors)


if __name__ == "__main__":
    main()
