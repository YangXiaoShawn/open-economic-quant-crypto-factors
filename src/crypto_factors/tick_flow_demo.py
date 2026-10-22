# -*- coding: utf-8 -*-
"""
T2 逐笔成交因子 demo —— 证明 crypto 免费拿到"主动买卖方向 + 大小单"。
对应开源系列 (9)主动买卖因子 / (12)(14)(16)大小单资金流 / (31)分钟资金流。

A 股: 这些要付费 Level-2 逐笔。
crypto: OKX `fetch_trades` 直接带 `side`(taker 主动方向)，单笔 notional 即可分大小单。

本脚本拉每个币种最近一批逐笔成交，算横截面订单流因子：
  OFI   = (主动买额 - 主动卖额) / 总成交额      净主动买入占比
  LARGE = 大单(notional 前10%)净买入占比         大单资金流方向
注: REST 只返回最近成交(快照)，做真回测需 WS 落库或 Tardis 历史。此处演示"可得性"。
"""
import time
import numpy as np
import pandas as pd
import ccxt

UNIVERSE = ["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT","DOGE/USDT",
            "ADA/USDT","AVAX/USDT","LINK/USDT","SUI/USDT","APT/USDT","ARB/USDT"]
N_TRADES = 1000     # 每币种抓多少笔


def fetch_recent_trades(ex, sym, n):
    rows, before = [], None
    while len(rows) < n:
        params = {} if before is None else {"end": before}
        batch = ex.fetch_trades(sym, limit=300, params=params)
        if not batch:
            break
        rows = batch + rows
        before = str(batch[0]["timestamp"])
        if len(batch) < 300:
            break
        time.sleep(ex.rateLimit / 1000)
    return rows[-n:]


def flow_factors(trades):
    df = pd.DataFrame([{"px": t["price"], "amt": t["amount"], "side": t["side"]}
                       for t in trades if t["price"] and t["amount"]])
    if df.empty:
        return None
    df["notional"] = df["px"] * df["amt"]
    tot = df["notional"].sum()
    buy = df.loc[df["side"] == "buy", "notional"].sum()
    sell = df.loc[df["side"] == "sell", "notional"].sum()
    ofi = (buy - sell) / tot if tot > 0 else np.nan
    # 大单: notional 前 10%
    thr = df["notional"].quantile(0.90)
    big = df[df["notional"] >= thr]
    btot = big["notional"].sum()
    bbuy = big.loc[big["side"] == "buy", "notional"].sum()
    bsell = big.loc[big["side"] == "sell", "notional"].sum()
    large = (bbuy - bsell) / btot if btot > 0 else np.nan
    return ofi, large, len(df), tot


def main():
    ex = ccxt.okx({"timeout": 15000, "enableRateLimit": True})
    print(f"抓取每币种最近 {N_TRADES} 笔逐笔成交 (OKX, 带主动买卖方向)…\n")
    print(f"{'币种':12s}{'笔数':>6s}{'成交额(USDT)':>16s}{'净主动买入%':>12s}{'大单净买入%':>12s}")
    print("-" * 60)
    recs = []
    for sym in UNIVERSE:
        try:
            tr = fetch_recent_trades(ex, sym, N_TRADES)
            r = flow_factors(tr)
        except Exception as e:
            print(f"{sym:12s}  FAIL {type(e).__name__}: {str(e)[:40]}")
            continue
        if not r:
            continue
        ofi, large, n, tot = r
        recs.append((sym, ofi, large))
        print(f"{sym:12s}{n:6d}{tot:16,.0f}{ofi*100:+11.1f}%{large*100:+11.1f}%")
    print("-" * 60)
    if recs:
        d = pd.DataFrame(recs, columns=["sym", "OFI", "LARGE"])
        print(f"\n横截面: 净主动买入 OFI 均值 {d['OFI'].mean()*100:+.1f}%, "
              f"大单方向与总流向相关 {d['OFI'].corr(d['LARGE']):+.2f}")
        print("→ 主动买卖方向 + 大小单 在 crypto 免费即得 (A股需付费Level-2)")
        print("  真回测: 用 WebSocket 实时落库 或 Tardis 历史逐笔，对齐次日收益算 IC。")


if __name__ == "__main__":
    main()
