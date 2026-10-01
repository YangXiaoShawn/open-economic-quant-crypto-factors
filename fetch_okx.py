# -*- coding: utf-8 -*-
"""
拉取 OKX 现货 1 分钟 K 线 (微观结构因子复现的 T1 数据档)。
- Binance/Bybit/KuCoin 在本环境被 geo-block (451/403)，OKX 可用。
- 增量 checkpoint 到 parquet：每个币种单独落盘，崩溃不丢已拉数据。
"""
import os, time, sys
import datetime as dt
import pandas as pd
import ccxt

OUTDIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTDIR, exist_ok=True)

# 流动性较好的 USDT 现货对 (跨市值/板块，便于做横截面)
UNIVERSE = [
    "BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT","DOGE/USDT",
    "ADA/USDT","AVAX/USDT","LINK/USDT","DOT/USDT","TRX/USDT","LTC/USDT",
    "BCH/USDT","NEAR/USDT","APT/USDT","ARB/USDT","OP/USDT","SUI/USDT",
    "FIL/USDT","ATOM/USDT","UNI/USDT","AAVE/USDT","INJ/USDT","TIA/USDT",
    "SEI/USDT","LDO/USDT","ETC/USDT","XLM/USDT","ICP/USDT","RENDER/USDT",
]

TIMEFRAME = "1m"
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 20   # 回看天数
LIMIT = 300                                            # OKX 单次上限


def fetch_symbol(ex, symbol, since_ms):
    """分页拉满 since→now 的 1m bars。"""
    all_rows = []
    cursor = since_ms
    now = ex.milliseconds()
    while cursor < now:
        try:
            batch = ex.fetch_ohlcv(symbol, TIMEFRAME, since=cursor, limit=LIMIT)
        except Exception as e:
            print(f"    retry {symbol} @ {cursor}: {type(e).__name__}")
            time.sleep(2)
            continue
        if not batch:
            break
        all_rows += batch
        cursor = batch[-1][0] + 60_000      # 下一根
        if len(batch) < LIMIT:
            break
    df = pd.DataFrame(all_rows, columns=["ts","open","high","low","close","volume"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def main():
    ex = ccxt.okx({"timeout": 15000, "enableRateLimit": True})
    target_since = ex.milliseconds() - DAYS * 24 * 3600 * 1000
    print(f"拉取 {len(UNIVERSE)} 个币种 {TIMEFRAME} K线，回看 {DAYS} 天 -> {OUTDIR}")
    for i, sym in enumerate(UNIVERSE, 1):
        fn = os.path.join(OUTDIR, sym.replace("/", "_") + ".parquet")
        old = None
        if os.path.exists(fn):
            old = pd.read_parquet(fn)
            have_min = int(old["ts"].min())
            if have_min <= target_since + 60_000:
                print(f"[{i:2d}/{len(UNIVERSE)}] {sym:12s} 已覆盖 {DAYS} 天，跳过")
                continue
            fetch_from = target_since           # 只补更早的缺口
        else:
            fetch_from = target_since
        t0 = time.time()
        try:
            df = fetch_symbol(ex, sym, fetch_from)
        except Exception as e:
            print(f"[{i:2d}/{len(UNIVERSE)}] {sym:12s} FAIL {type(e).__name__}: {str(e)[:60]}")
            continue
        if old is not None and len(old):
            df = (pd.concat([old, df], ignore_index=True)
                    .drop_duplicates("ts").sort_values("ts").reset_index(drop=True))
        if len(df) == 0:
            print(f"[{i:2d}/{len(UNIVERSE)}] {sym:12s} 无数据")
            continue
        df.to_parquet(fn, index=False)       # 增量落盘
        print(f"[{i:2d}/{len(UNIVERSE)}] {sym:12s} {len(df):6d} bars  "
              f"{df['dt'].min()} ~ {df['dt'].max()}  {time.time()-t0:.1f}s")
    print("完成。")


if __name__ == "__main__":
    main()
