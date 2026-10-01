# -*- coding: utf-8 -*-
"""
扩 perp 全币种 + 叠加 funding/OI —— Binance USDT 永续 (futures/um) 免费 dump。

为什么走 perp:
  1) 永续合约币种数 (732 USDT-perp) 远多于现货大币 → 给"价格类因子"更大横截面广度;
  2) **funding 资金费率** 与 **OI 持仓量 / 多空比** 是 A 股完全没有的资金流维度,
     只在衍生品市场存在, 永续 dump 免费可回溯 (funding→2021, OI metrics→2023)。

数据源 (data.binance.vision, 默认 TLS 校验):
  klines  : futures/um/monthly/klines/<SYM>/1m/<SYM>-1m-YYYY-MM.zip   (与现货同 12 列)
  funding : futures/um/monthly/fundingRate/<SYM>/<SYM>-fundingRate-YYYY-MM.zip
  OI/多空 : futures/um/daily/metrics/<SYM>/<SYM>-metrics-YYYY-MM-DD.zip (5min, 仅日档)

产物:
  data_binance/<SYM>.parquet         新增 perp 币种的日频价量+T2资金流特征 (复用 day_features)
  data_perp/funding/<SYM>.parquet    日频 funding 特征
  data_perp/oi/<SYM>.parquet         日频 OI / 多空比 / taker 比 特征

用法:
  python perp_pipeline.py klines    # 下载扩池 perp 1m K线 -> data_binance (可后台)
  python perp_pipeline.py funding   # 下载 funding (全池)
  python perp_pipeline.py oi        # 下载 OI metrics (大币池, 限窗)
  python perp_pipeline.py all
"""
import os, io, sys, zipfile, urllib.request
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

from binance_pipeline import day_features, months, COLS, UNIVERSE as SPOT_UNIVERSE

FBASE = "https://data.binance.vision/data/futures/um"
ROOT = os.path.dirname(__file__)
KDIR = os.path.join(ROOT, "data_binance")            # 与现货特征同目录, 面板自动合并
FUNDDIR = os.path.join(ROOT, "data_perp", "funding")
OIDIR = os.path.join(ROOT, "data_perp", "oi")
for d in (KDIR, FUNDDIR, OIDIR):
    os.makedirs(d, exist_ok=True)

START, END = (2024, 6), (2026, 5)        # 与现货面板对齐
OI_START = (2025, 1)                      # OI 仅日档, 下载量大 -> 限近窗 (~17 个月)

# 大币池: 资金流 / OI 因子只在此池有效 (与 README 扩池实验一致, 前 24 个大市值)
BIG_POOL = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT",
            "ADAUSDT","AVAXUSDT","LINKUSDT","DOTUSDT","TRXUSDT","LTCUSDT",
            "BCHUSDT","NEARUSDT","APTUSDT","ARBUSDT","OPUSDT","FILUSDT",
            "ATOMUSDT","UNIUSDT","AAVEUSDT","INJUSDT","ETCUSDT","XLMUSDT"]

# 扩 perp: 现货 60 之外再加 ~45 个流动性较好的永续币种 (404/短历史会自动跳过)
EXTRA_PERP = ["1000PEPEUSDT","1000FLOKIUSDT","1000BONKUSDT","1000SATSUSDT","1000SHIBUSDT",
              "ENAUSDT","ONDOUSDT","TAOUSDT","TONUSDT","ARUSDT","SUSHIUSDT","COMPUSDT",
              "YFIUSDT","GMXUSDT","DYDXUSDT","BLURUSDT","JTOUSDT","STRKUSDT","PIXELUSDT",
              "MANTAUSDT","ALTUSDT","ZKUSDT","ZROUSDT","WUSDT","ETHFIUSDT","SAGAUSDT",
              "BOMEUSDT","NOTUSDT","IOUSDT","ZRXUSDT","KSMUSDT","FLOWUSDT","EGLDUSDT",
              "XTZUSDT","QNTUSDT","PENDLEUSDT","ARKMUSDT","SSVUSDT","AGLDUSDT","NEOUSDT",
              "IOTAUSDT","GASUSDT","ACEUSDT","AEVOUSDT","OMUSDT"]

# 全 perp 池 (现货已下载的 60 + 新增 perp) ; funding 对全池下载
FULL_POOL = list(dict.fromkeys(SPOT_UNIVERSE + EXTRA_PERP))


def _get(url, timeout=30):
    return urllib.request.urlopen(url, timeout=timeout).read()


# ---------------------------------------------------------------- perp klines
def load_kline_month(sym, y, m):
    url = f"{FBASE}/monthly/klines/{sym}/1m/{sym}-1m-{y:04d}-{m:02d}.zip"
    try:
        raw = _get(url)
    except Exception:
        return None
    z = zipfile.ZipFile(io.BytesIO(raw))
    df = pd.read_csv(z.open(z.namelist()[0]), header=None)
    if str(df.iloc[0, 0]).startswith("open"):
        df = df.iloc[1:].reset_index(drop=True)
    df = df.iloc[:, :12]; df.columns = COLS
    for c in ["o","h","l","c","v","qv","tbb","tbq","n"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ot = pd.to_numeric(df["ot"], errors="coerce").astype("int64")
    if ot.iloc[0] > 1e15:
        ot = ot // 1000
    df["dt"] = pd.to_datetime(ot, unit="ms", utc=True)
    return df.dropna(subset=["c","v"])


def build_kline(sym):
    fn = os.path.join(KDIR, sym + ".parquet")
    if os.path.exists(fn):
        return sym, "skip(exists)"
    rows = []
    for y, m in months(START, END):
        dfm = load_kline_month(sym, y, m)
        if dfm is None:
            continue
        dfm["day"] = dfm["dt"].dt.floor("D")
        for day, g in dfm.groupby("day"):
            f = day_features(g)
            if f:
                f["day"] = day; f["sym"] = sym; rows.append(f)
    if not rows:
        return sym, "no-data"
    d = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    d.to_parquet(fn, index=False)
    return sym, f"{len(d)}d {d['day'].min().date()}~{d['day'].max().date()}"


# ---------------------------------------------------------------- funding
def build_funding(sym):
    fn = os.path.join(FUNDDIR, sym + ".parquet")
    if os.path.exists(fn):
        return sym, "skip"
    rows = []
    for y, m in months(START, END):
        url = f"{FBASE}/monthly/fundingRate/{sym}/{sym}-fundingRate-{y:04d}-{m:02d}.zip"
        try:
            raw = _get(url, 20)
        except Exception:
            continue
        z = zipfile.ZipFile(io.BytesIO(raw))
        df = pd.read_csv(z.open(z.namelist()[0]))
        df["calc_time"] = pd.to_numeric(df["calc_time"], errors="coerce")
        df["last_funding_rate"] = pd.to_numeric(df["last_funding_rate"], errors="coerce")
        df["day"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True).dt.floor("D")
        rows.append(df[["day","last_funding_rate"]])
    if not rows:
        return sym, "no-data"
    f = pd.concat(rows, ignore_index=True)
    g = f.groupby("day")["last_funding_rate"].agg(
        FUND_sum="sum", FUND_mean="mean", FUND_n="count").reset_index()
    g["sym"] = sym
    g.sort_values("day").to_parquet(fn, index=False)
    return sym, f"{len(g)}d"


# ---------------------------------------------------------------- OI / 多空
def load_oi_day(sym, day):
    url = f"{FBASE}/daily/metrics/{sym}/{sym}-metrics-{day:%Y-%m-%d}.zip"
    try:
        raw = _get(url, 20)
    except Exception:
        return None
    z = zipfile.ZipFile(io.BytesIO(raw))
    return pd.read_csv(z.open(z.namelist()[0]))


def build_oi(sym):
    fn = os.path.join(OIDIR, sym + ".parquet")
    if os.path.exists(fn):
        return sym, "skip"
    start = dt.date(OI_START[0], OI_START[1], 1)
    end = dt.date(END[0], END[1], 28)
    rows = []
    day = start
    while day <= end:
        m = load_oi_day(sym, day)
        if m is not None and len(m):
            m["create_time"] = pd.to_datetime(m["create_time"], utc=True)
            oiv = pd.to_numeric(m["sum_open_interest_value"], errors="coerce")
            rows.append(dict(
                day=pd.Timestamp(day, tz="UTC"),
                OI_close=float(oiv.iloc[-1]),
                OI_mean=float(oiv.mean()),
                # 全市场账户多空比 (散户拥挤度, 反向)
                GLS=float(pd.to_numeric(m["count_long_short_ratio"], errors="coerce").mean()),
                # 大户持仓多空比 (聪明钱定位, 顺势)
                TLS=float(pd.to_numeric(m["sum_toptrader_long_short_ratio"], errors="coerce").mean()),
                # taker 主动买/卖量比 (5min 级资金流方向)
                TKLS=float(pd.to_numeric(m["sum_taker_long_short_vol_ratio"], errors="coerce").mean()),
            ))
        day += dt.timedelta(days=1)
    if not rows:
        return sym, "no-data"
    d = pd.DataFrame(rows).sort_values("day")
    d["sym"] = sym
    d.to_parquet(fn, index=False)
    return sym, f"{len(d)}d"


# ---------------------------------------------------------------- runners
def run_pool(fn, pool, label, workers=10):
    print(f"[{label}] {len(pool)} 币种, {workers} 线程")
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, s): s for s in pool}
        for fu in as_completed(futs):
            sym = futs[fu]
            try:
                s, msg = fu.result()
            except Exception as e:
                s, msg = sym, f"FAIL {type(e).__name__}:{str(e)[:40]}"
            done += 1
            print(f"  [{done:3d}/{len(pool)}] {s:14s} {msg}")
    print(f"[{label}] 完成。")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("klines", "all"):
        run_pool(build_kline, EXTRA_PERP, "perp klines (扩池新币)", workers=8)
    if cmd in ("funding", "all"):
        run_pool(build_funding, FULL_POOL, "funding (全池)", workers=12)
    if cmd in ("oi", "all"):
        run_pool(build_oi, BIG_POOL, "OI metrics (大币池, 限窗)", workers=12)


if __name__ == "__main__":
    main()
