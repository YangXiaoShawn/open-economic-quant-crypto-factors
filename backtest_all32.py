# -*- coding: utf-8 -*-
"""
逐篇复现 开源·微观结构系列 全 32 篇 + 横截面回测 (Binance 24币 × 730天)。

每篇映射到一个可从 1m OHLCV + taker方向 计算的因子:
  直接: 报告的核心因子可直接复算
  代理: crypto无对应数据(无行业/无市值/无L3委托) -> 用透明代理
  受限: 需逐笔委托(L3 撤单/挂单), 免费dump无 -> 标注, 尽量给代理
  方法/综述: 非独立因子的方法论/回顾 -> 用代表因子或合成因子近似
"""
import os, glob
import numpy as np
import pandas as pd
from backtest import backtest_factor, metrics

DATADIR = os.path.join(os.path.dirname(__file__), "data_binance")
N = 5

def autocorr1(x):
    x = pd.Series(x)
    return x.autocorr(1) if x.std() > 0 and x.notna().sum() > 3 else np.nan

def build_panel():
    recs = []
    for fn in glob.glob(os.path.join(DATADIR, "*.parquet")):
        d = pd.read_parquet(fn).sort_values("day").reset_index(drop=True)
        d["fwd_ret"] = d["close"].shift(-1) / d["close"] - 1.0
        # 价格类
        d["Q_f"]      = d["Q"].rolling(N, min_periods=2).mean()
        d["REV_hi_f"] = d["REV_hi"].rolling(N, min_periods=2).sum()
        d["APM_f"]    = (d["r_pm"] - d["r_am"]).rolling(N, min_periods=2).mean()
        d["AMP_f"]    = d["AMP"].rolling(N, min_periods=2).mean()
        d["AMP_lo_f"] = d["AMP_lo"].rolling(N, min_periods=2).mean()
        d["SKEW_f"]   = d["SKEW"].rolling(N, min_periods=2).mean()
        d["AC1_f"]    = d["AC1"].rolling(N, min_periods=2).mean()
        d["VCONC_f"]  = d["VCONC"].rolling(N, min_periods=2).mean()
        d["VRCORR_f"] = d["VRCORR"].rolling(N, min_periods=2).mean()
        # T2 资金流
        d["OFI_f"]      = d["OFI"].rolling(N, min_periods=2).mean()
        d["OFIbig_f"]   = d["OFI_big"].rolling(N, min_periods=2).mean()
        d["RETAIL_f"]   = d["OFI_small"].rolling(N, min_periods=2).mean()
        d["AMTPT_f"]    = d["AMTPT"] / d["AMTPT"].rolling(20, min_periods=5).mean()
        d["OFIAC_f"]    = d["OFI"].rolling(20, min_periods=8).apply(autocorr1, raw=False)
        # 其它
        d["MOM_f"]   = d["close"] / d["close"].shift(10) - 1.0          # 截面动量(代理行业动量)
        d["SIZE_f"]  = np.log(d["qvol"]).rolling(N, min_periods=2).mean()  # 市值代理
        d["MAXR_f"]  = d["MAXR"].rolling(N, min_periods=2).mean()       # 极端收益(彩票)
        d["DNVOL_f"] = d["DNVOL"].rolling(N, min_periods=2).mean()
        recs.append(d)
    panel = pd.concat(recs, ignore_index=True)
    # 合成因子(代表"GA/DL多因子": 稳健反转+大单资金流)
    def z(col, dirn):
        return panel.groupby("day")[col].transform(lambda x:(x-x.mean())/(x.std()+1e-9))*dirn
    panel["COMBO_f"] = np.nanmean(np.vstack([
        z("REV_hi_f",-1).values, z("OFIbig_f",1).values, z("OFI_f",1).values]), axis=0)
    return panel

# (序号, 简称, 因子列, 方向, 状态)
REPORTS = [
    (1,  "反转之力·微观来源",   "REV_hi_f", -1, "直接"),
    (2,  "交易行为因子2019回顾", "Q_f",      -1, "综述"),
    (3,  "聪明钱因子2.0",       "Q_f",      -1, "直接"),
    (4,  "行业动量精细结构",     "MOM_f",    +1, "代理·无行业"),
    (5,  "APM进阶",            "APM_f",    -1, "直接"),
    (6,  "交易者行为·市值风格",  "SIZE_f",   -1, "代理·市值"),
    (7,  "振幅因子隐藏结构",     "AMP_f",    -1, "直接"),
    (8,  "行业轮动300指增",     "MOM_f",    +1, "代理·应用"),
    (9,  "主动买卖因子",        "OFI_f",    +1, "直接·T2"),
    (10, "因子切割论(方法)",     "AMP_lo_f", -1, "方法"),
    (11, "分层效应普适规律",     "REV_hi_f", -1, "方法"),
    (12, "大单小单资金流",       "OFIbig_f", +1, "直接·T2"),
    (13, "理想反转4年总结",      "REV_hi_f", -1, "直接"),
    (14, "资金流动力学·散户羊群", "RETAIL_f", -1, "直接·T2"),
    (15, "分钟单笔金额序列",     "AMTPT_f",  +1, "直接·T2"),
    (16, "大小单重定标改进",     "OFIbig_f", +1, "直接·T2"),
    (17, "日内极端收益反转",     "MAXR_f",   -1, "直接"),
    (18, "大小单资金流2.0精筛",  "OFIbig_f", +1, "直接·T2"),
    (19, "日内分钟收益时序",     "AC1_f",    -1, "直接"),
    (20, "遗传算法赋能(方法)",   "COMBO_f",  +1, "方法·合成"),
    (21, "订单流变迁(故事)",     "OFI_f",    +1, "受限·L3代理"),
    (22, "撤单行为规律",        None,       0,  "✗需L3委托"),
    (23, "大小单核心行业轮动",   "OFIbig_f", +1, "代理·无行业"),
    (24, "深度学习赋能(方法)",   "COMBO_f",  +1, "方法·合成"),
    (25, "挂单方向长期记忆",     "OFIAC_f",  +1, "受限·L3代理"),
    (26, "因子失效讨论(分析)",   "OFI_f",    +1, "分析"),
    (27, "高频成交量峰岭谷",     "VRCORR_f", -1, "直接"),
    (28, "切割论+DL(方法)",     "AMP_lo_f", -1, "方法"),
    (29, "2023高频因子回顾",    "COMBO_f",  +1, "综述·合成"),
    (30, "高频振幅内部切割",     "AMP_lo_f", -1, "直接"),
    (31, "分钟资金流因子构建",   "OFI_f",    +1, "直接·T2"),
    (32, "深度学习因子挖掘2.0",  "COMBO_f",  +1, "方法·合成"),
]


def main():
    panel = build_panel()
    nd, nc = panel["day"].nunique(), panel["sym"].nunique()
    print(f"\n{'='*86}")
    print(f"开源·微观结构系列 全32篇 crypto复现回测  ({nc}币 × {nd}天, "
          f"{panel['day'].min().date()}~{panel['day'].max().date()}, 五分位多空,日调仓,5bps)")
    print(f"{'='*86}")
    print(f"{'#':>3s} {'研报简称':22s}{'因子':10s}{'净值':>6s}{'夏普':>6s}{'回撤':>6s}"
          f"{'胜率':>6s}  {'状态/判定'}")
    print("-" * 86)
    results = []
    for num, name, col, d, status in REPORTS:
        if col is None:
            print(f"{num:>3d} {name:22s}{'—':10s}{'—':>6s}{'—':>6s}{'—':>6s}{'—':>6s}  {status}")
            results.append((num, name, status, np.nan, np.nan, col))
            continue
        bt = backtest_factor(panel, col, d)
        m = metrics(bt["ls"])
        if not m:
            print(f"{num:>3d} {name:22s}{col[:-2]:10s} 样本不足  {status}")
            continue
        verdict = "✅" if m["sharpe"] > 0.5 else ("◐" if m["sharpe"] > 0 else "✗")
        print(f"{num:>3d} {name:22s}{col[:-2]:10s}{1+m['tot']:6.2f}{m['sharpe']:6.2f}"
              f"{m['mdd']*100:5.0f}%{m['win']*100:5.0f}%  {verdict} {status}")
        results.append((num, name, status, m["tot"], m["sharpe"], col))
    print("-" * 86)

    R = pd.DataFrame(results, columns=["num","name","status","tot","sharpe","col"]).dropna(subset=["sharpe"])
    uniq = R.drop_duplicates("col")
    print(f"\n【汇总】32篇: 可回测 {R.shape[0]} 篇 (含代理/合成), 受L3限无法回测 1 篇(#22撤单)")
    print(f"  独立因子 {uniq.shape[0]} 个; 其中夏普>0.5(稳健迁移) "
          f"{(uniq['sharpe']>0.5).sum()} 个, 0~0.5 {(uniq['sharpe'].between(0,0.5)).sum()} 个, "
          f"<0(不迁移) {(uniq['sharpe']<0).sum()} 个")
    print("\n  独立因子按夏普排序(去重):")
    for _, r in uniq.sort_values("sharpe", ascending=False).iterrows():
        print(f"    {r['col'][:-2]:10s} 夏普{r['sharpe']:+.2f} 净值{1+r['tot']:.2f}  ({r['name']})")
    print("=" * 86)


if __name__ == "__main__":
    main()
