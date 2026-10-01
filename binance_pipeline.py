# -*- coding: utf-8 -*-
"""
数百天 + T2 逐笔真回测 —— 用 Binance 官方免费 dump (data.binance.vision)。

为什么这是"真 T2":
  Binance 1m kline dump 每根含 taker_buy_base / taker_buy_quote / count(成交笔数)，
  这是对**逐笔成交**按方向聚合的结果 (taker=主动成交方)。于是无需逐笔原始数据即可得:
    OFI    = taker 主动买入额 / 总成交额        (主动买卖资金流, 报告 9/12/31)
    单笔额 = 成交额 / 成交笔数                   (分钟单笔金额, 报告 15)
  同一份 dump 的 OHLCV 还能算全部价格类因子 → 一份数据跑通价量+资金流, 跨数百天。

流程: 按月下载 zip -> 解析 1m -> 聚合到 (币, UTC日) 特征 -> 缓存 per-coin parquet。
"""
import os, io, zipfile, urllib.request
import datetime as dt
import numpy as np
import pandas as pd

BASE = "https://data.binance.vision/data/spot/monthly/klines"
OUTDIR = os.path.join(os.path.dirname(__file__), "data_binance")
os.makedirs(OUTDIR, exist_ok=True)

# Binance 现货符号(无斜杠)。挑流动性好、历史较长的
UNIVERSE = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT",
            "ADAUSDT","AVAXUSDT","LINKUSDT","DOTUSDT","TRXUSDT","LTCUSDT",
            "BCHUSDT","NEARUSDT","APTUSDT","ARBUSDT","OPUSDT","FILUSDT",
            "ATOMUSDT","UNIUSDT","AAVEUSDT","INJUSDT","ETCUSDT","XLMUSDT",
            # --- 扩池: 增厚每腿样本 (新币历史短会自动少天数) ---
            "SUIUSDT","TIAUSDT","SEIUSDT","WLDUSDT","PEPEUSDT","WIFUSDT",
            "JUPUSDT","ORDIUSDT","FETUSDT","RUNEUSDT","ALGOUSDT","GALAUSDT",
            "SANDUSDT","MANAUSDT","AXSUSDT","GRTUSDT","IMXUSDT","CHZUSDT",
            "EOSUSDT","MKRUSDT","SNXUSDT","CRVUSDT","ENSUSDT","LDOUSDT",
            "STXUSDT","THETAUSDT","HBARUSDT","VETUSDT","ICPUSDT","APEUSDT",
            "GMTUSDT","SHIBUSDT","JASMYUSDT","CFXUSDT","PYTHUSDT","KAVAUSDT"]

START = (2024, 6)      # 起始年月
END   = (2026, 5)      # 结束年月(含)
COLS = ["ot","o","h","l","c","v","ct","qv","n","tbb","tbq","ig"]


def months(a, b):
    y, m = a
    while (y, m) <= b:
        yield y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


def load_month(sym, y, m):
    fn = f"{sym}-1m-{y:04d}-{m:02d}.zip"
    url = f"{BASE}/{sym}/1m/{fn}"
    try:
        raw = urllib.request.urlopen(url, timeout=30).read()   # 默认TLS证书校验
    except Exception:
        return None                                  # 404=该月未上市
    z = zipfile.ZipFile(io.BytesIO(raw))
    df = pd.read_csv(z.open(z.namelist()[0]), header=None)
    if str(df.iloc[0, 0]).startswith("open"):        # 新版带表头
        df = df.iloc[1:].reset_index(drop=True)
    df = df.iloc[:, :12]
    df.columns = COLS
    for c in ["o","h","l","c","v","qv","tbb","tbq","n"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ot = pd.to_numeric(df["ot"], errors="coerce").astype("int64")
    if ot.iloc[0] > 1e15:                            # 微秒 -> 毫秒 (2025+ dump)
        ot = ot // 1000
    df["dt"] = pd.to_datetime(ot, unit="ms", utc=True)
    return df.dropna(subset=["c","v"])


def day_features(g):
    g = g.sort_values("dt")
    p = g["c"].to_numpy(float); v = g["v"].to_numpy(float)
    h = g["h"].to_numpy(float); lo = g["l"].to_numpy(float)
    qv = g["qv"].to_numpy(float); tbq = g["tbq"].to_numpy(float)
    n_tr = g["n"].to_numpy(float); hour = g["dt"].dt.hour.to_numpy()
    n = len(g)
    if n < 120 or v.sum() <= 0 or p[0] <= 0:
        return None
    r = np.zeros(n); r[1:] = p[1:] / p[:-1] - 1.0
    o = {}
    # ---- 价格类 ----
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.abs(r) / np.sqrt(np.where(v > 0, v, np.nan))
    order = np.argsort(-np.nan_to_num(s, nan=-1))
    k = np.searchsorted(np.cumsum(v[order]), 0.2 * v.sum()) + 1
    sm = order[:max(k,1)]
    o["Q"] = (p[sm]*v[sm]).sum()/v[sm].sum() / ((p*v).sum()/v.sum())
    amt = p*v; ao = np.argsort(-amt); kt = max(int(0.3*n),1)
    o["REV_hi"] = r[ao[:kt]].sum()
    am = hour < 12; o["r_am"] = r[am].sum(); o["r_pm"] = r[~am].sum()
    amp = (h-lo)/np.where(p>0,p,np.nan); o["AMP"] = np.nanmean(amp)
    o["AMP_lo"] = np.nanmean(amp[p <= np.median(p)])
    o["SKEW"] = pd.Series(r[1:]).skew()
    rr = r[1:]; o["AC1"] = pd.Series(rr).autocorr(1) if rr.std()>0 else np.nan
    vs = np.sort(v)[::-1]; o["VCONC"] = vs[:max(int(0.1*n),1)].sum()/v.sum()
    o["VRCORR"] = np.corrcoef(v, np.abs(r))[0,1] if v.std()>0 and np.std(np.abs(r))>0 else np.nan
    # ---- T2 资金流 (taker 方向) ----
    o["OFI"]   = tbq.sum()/qv.sum() if qv.sum()>0 else np.nan      # 主动买入额占比(9/12/31)
    o["AMTPT"] = qv.sum()/n_tr.sum() if n_tr.sum()>0 else np.nan   # 平均单笔金额(15)
    big = ao[:kt]; small = ao[-kt:]                               # 成交额前/后30%分钟
    o["OFI_big"]   = (tbq[big].sum()/qv[big].sum() if qv[big].sum()>0 else np.nan)   # 大单主动买(16/18)
    o["OFI_small"] = (tbq[small].sum()/qv[small].sum() if qv[small].sum()>0 else np.nan) # 小单/散户(14)
    # ---- 其它单日特征 ----
    o["qvol"]  = qv.sum()                                         # 日成交额 -> 市值代理(6)
    o["ret"]   = p[-1]/p[0] - 1.0                                 # 日收益 -> 动量(4/8)
    o["MAXR"]  = float(np.nanmax(r))                              # 日内极端正收益(17)
    dn = r[r < 0]
    o["DNVOL"] = (np.sqrt((dn**2).sum()) / np.sqrt((r**2).sum())) if (r**2).sum()>0 else np.nan  # 下行波动占比
    o["close"] = p[-1]
    return o


def build_coin(sym):
    fn = os.path.join(OUTDIR, sym + ".parquet")
    if os.path.exists(fn):
        return pd.read_parquet(fn)
    rows = []
    for y, m in months(START, END):
        dfm = load_month(sym, y, m)
        if dfm is None:
            continue
        dfm["day"] = dfm["dt"].dt.floor("D")
        for day, g in dfm.groupby("day"):
            f = day_features(g)
            if f:
                f["day"] = day; f["sym"] = sym; rows.append(f)
    if not rows:
        return None
    d = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    d.to_parquet(fn, index=False)
    return d


def main():
    print(f"下载 Binance 1m dump: {len(UNIVERSE)}币 × {START}~{END}  -> {OUTDIR}")
    for i, sym in enumerate(UNIVERSE, 1):
        try:
            d = build_coin(sym)
        except Exception as e:
            print(f"[{i:2d}] {sym:10s} FAIL {type(e).__name__}: {str(e)[:50]}")
            continue
        if d is None:
            print(f"[{i:2d}] {sym:10s} 无数据"); continue
        print(f"[{i:2d}] {sym:10s} {len(d):5d} 天  {d['day'].min().date()} ~ {d['day'].max().date()}")
    print("下载+聚合完成。")


if __name__ == "__main__":
    main()
