# -*- coding: utf-8 -*-
"""
跨交易所样本外验证 —— 在 **OKX** (独立于 Binance) 上复算两个最关键因子, 看是否是"币安单所伪象":
  - 资金费率 funding (王牌, A股没有的 perp 维度)
  - 振幅 amplitude (日内高低幅, 内部 OOS 里参数最稳的因子) —— 用日线 (H-L)/C 作代理
数据: OKX 现货日线 OHLCV (反转/动量/振幅) + 永续 funding 历史 (ccxt 直连, 本环境唯一可用所)。
缓存到 data_okx/, 重跑免重拉。最后与 Binance 同口径夏普对比 —— 同号即"非单所过拟合"。
"""
import os, time
import numpy as np
import pandas as pd
import ccxt
import backtest
from backtest import backtest_factor, metrics

ROOT = os.path.dirname(__file__)
OKXDIR = os.path.join(ROOT, "data_okx")
os.makedirs(OKXDIR, exist_ok=True)

COINS = ["BTC","ETH","SOL","XRP","DOGE","ADA","AVAX","LINK","DOT","TRX","LTC","BCH",
         "NEAR","APT","ARB","OP","FIL","ATOM","UNI","AAVE","INJ","ETC","XLM","SUI",
         "TIA","SEI","WLD","RENDER","LDO","ICP"]            # 去掉 BNB(OKX 不上)
START = "2024-06-01T00:00:00Z"


def fetch_okx(ex):
    fn = os.path.join(OKXDIR, "okx_panel.parquet")
    if os.path.exists(fn):
        return pd.read_parquet(fn)
    since0 = ex.parse8601(START)
    rows = []
    for i, c in enumerate(COINS, 1):
        spot, swap = f"{c}/USDT", f"{c}/USDT:USDT"
        # --- 日线 OHLCV ---
        try:
            ohlcv, since = [], since0
            while True:
                batch = ex.fetch_ohlcv(spot, "1d", since=since, limit=300)
                if not batch:
                    break
                ohlcv += batch
                since = batch[-1][0] + 86400_000
                if len(batch) < 300:
                    break
            d = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"]).drop_duplicates("ts")
            d["day"] = pd.to_datetime(d["ts"], unit="ms", utc=True).dt.floor("D")
        except Exception as e:
            print(f"[{i:2d}] {c:8s} OHLCV FAIL {str(e)[:40]}"); continue
        # --- 永续 funding 历史 ---
        fund = {}
        try:
            since = since0
            while True:
                fb = ex.fetch_funding_rate_history(swap, since=since, limit=100)
                if not fb:
                    break
                for r in fb:
                    day = pd.to_datetime(r["timestamp"], unit="ms", utc=True).floor("D")
                    fund[day] = fund.get(day, 0.0) + (r["fundingRate"] or 0.0)
                since = fb[-1]["timestamp"] + 1
                if len(fb) < 100:
                    break
        except Exception as e:
            print(f"[{i:2d}] {c:8s} funding FAIL {str(e)[:40]}")
        d["FUND_sum"] = d["day"].map(fund)
        d["sym"] = c
        rows.append(d[["day","sym","o","h","l","c","v","FUND_sum"]])
        print(f"[{i:2d}] {c:8s} {len(d):4d}d  funding {d['FUND_sum'].notna().sum():4d}d")
    panel = pd.concat(rows, ignore_index=True)
    panel.to_parquet(fn, index=False)
    return panel


def build(panel):
    recs = []
    for c, d in panel.groupby("sym"):
        d = d.sort_values("day").reset_index(drop=True)
        d["fwd_ret"] = d["c"].shift(-1) / d["c"] - 1.0
        d["AMP_f"]  = ((d["h"] - d["l"]) / d["c"]).rolling(5, min_periods=2).mean()   # 日振幅代理
        d["REV_f"]  = (d["c"] / d["c"].shift(5) - 1.0)                                # 5日反转
        d["MOM_f"]  = (d["c"] / d["c"].shift(10) - 1.0)                               # 10日动量
        d["FUND_f"] = d["FUND_sum"].rolling(5, min_periods=2).mean()
        recs.append(d)
    return pd.concat(recs, ignore_index=True)


def main():
    ex = ccxt.okx({"enableRateLimit": True})
    ex.load_markets()
    print(f"拉 OKX {len(COINS)} 币 日线+funding (缓存 data_okx/) ...")
    panel = build(fetch_okx(ex))
    nd, nc = panel["day"].nunique(), panel["sym"].nunique()
    print(f"\n{'='*78}\n跨交易所 OOS: OKX {nc}币 × {nd}天 vs Binance 同口径  (五分位多空,5bps)\n{'='*78}")
    print(f"{'因子':14s}{'方向':>4s}{'OKX净值':>9s}{'OKX夏普':>9s}{'Binance夏普(参考)':>18s}  判定")
    print("-" * 78)
    # Binance 参考值 (来自 backtest_perp 大币池/宽池, 此处对照)
    bin_ref = {"资金费率": "+1.42(大币)", "日振幅": "+1.63(低价位振幅,宽池)",
               "5日反转": "+0.25(理想反转,宽池)", "10日动量": "-0.38"}
    for col, name, dirn in [("FUND_f","资金费率",-1), ("AMP_f","日振幅",-1),
                            ("REV_f","5日反转",-1), ("MOM_f","10日动量",+1)]:
        bt = backtest_factor(panel.dropna(subset=[col,"fwd_ret"]), col, dirn)
        m = metrics(bt["ls"])
        if not m:
            print(f"{name:14s} 样本不足"); continue
        verdict = "✅同号复现" if (m["sharpe"]>0) == (not name=="10日动量") else "◐"
        print(f"{name:14s}{dirn:+4d}{1+m['tot']:9.2f}{m['sharpe']:+9.2f}{bin_ref.get(name,''):>18s}  {verdict}")
    print("-" * 78)
    print("结论: funding/振幅 若在 OKX 同号为正 → 非币安单所伪象, 跨所稳健。")


if __name__ == "__main__":
    main()
