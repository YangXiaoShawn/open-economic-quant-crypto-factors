# -*- coding: utf-8 -*-
"""
增量更新数据到最新 —— 只补 last_day 之后的部分, 已有 parquet 原地追加。

原 binance_pipeline / perp_pipeline 是"文件存在就 skip"的一次性下载器, 无法续期。
本脚本按 (币, 数据集) 找出缓存里的最后一天, 只下载缺口:
  * 完整自然月     -> monthly zip (1 个请求/月, 快)
  * 当月不完整部分 -> daily zip  (1 个请求/天)

三个数据集:
  klines  data_binance/<SYM>.parquet      现货(SPOT_UNIVERSE)或永续(其余) 1m -> 日频特征
  funding data_perp/funding/<SYM>.parquet 资金费率日频聚合 (FUND_sum/mean/n)
  oi      data_perp/oi/<SYM>.parquet      OI/多空比 (仅 BIG_POOL, 仅 daily metrics)

⚠ fundingRate dump **只有 monthly**(无 daily), 所以当月的 funding 只能靠 premiumIndexKlines
  按币安公式反推 (见 funding_proxy_days)。是否采用由 `python update_data.py validate` 在上一个
  完整月上做实测校验后决定 —— 不校验就不该用。

  【2026-06 实测校验结论: proxy 不合格, 已弃用, 请勿运行 `proxy` 子命令】
    逐点 corr 0.798; 横截面秩相关 全池 0.56 / 大币池 0.91 (判定线 0.95)。
    公式本身是对的 (F = P + clamp(I−P) 的残差恰好等于 clamp 值 5bp), 1m 采样也已修正
    (1h close 采样只有 0.65~0.73), 但**每个币的 interest rate / clamp / cap 是币安逐品种
    设定的**, 无法从公开 dump 反推 —— 这正是全池秩相关只有 0.56 的原因。
    => funding 硬止于**上一个完整自然月末**。价格/OI 类因子可以更新到昨天, funding 类不行,
       两者末日不同, 报告里必须逐行标注天数 (report_update.py 已如此处理)。

用法:
  python update_data.py probe      # 探测各数据集最新可得日期 + 缓存现状
  python update_data.py klines
  python update_data.py funding
  python update_data.py oi
  python update_data.py validate   # 用上一完整月校验 funding proxy 保真度
  python update_data.py proxy      # 把校验通过的 proxy 写入 funding parquet (标记 FUND_src)
  python update_data.py all        # klines + funding + oi
"""
import os, io, sys, zipfile, urllib.request, urllib.error
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

from binance_pipeline import day_features, COLS, UNIVERSE as SPOT_UNIVERSE
from perp_pipeline import BIG_POOL

SBASE = "https://data.binance.vision/data/spot"
FBASE = "https://data.binance.vision/data/futures/um"
ROOT = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(ROOT, "data_binance")
FUNDDIR = os.path.join(ROOT, "data_perp", "funding")
OIDIR = os.path.join(ROOT, "data_perp", "oi")

SPOT_SET = set(SPOT_UNIVERSE)


# ------------------------------------------------------------------ http
def _get(url, timeout=40, retries=2):
    last = None
    for _ in range(retries + 1):
        try:
            return urllib.request.urlopen(url, timeout=timeout).read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None            # 该月/日不存在 (未上市/已下架/尚未发布)
            last = e
        except Exception as e:
            last = e
    raise last


def _exists(url, timeout=20):
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout)
        return True
    except Exception:
        return False


def _unzip_csv(raw, **kw):
    z = zipfile.ZipFile(io.BytesIO(raw))
    return pd.read_csv(z.open(z.namelist()[0]), **kw)


# ------------------------------------------------------------------ 缺口规划
def plan_chunks(start_day, end_day):
    """把 [start_day, end_day] 拆成 [('M', y, m), ...] + [('D', date), ...]。
    整月完全落在区间内 -> 走 monthly; 否则该月逐日。"""
    out = []
    y, m = start_day.year, start_day.month
    while (y, m) <= (end_day.year, end_day.month):
        first = dt.date(y, m, 1)
        nxt = dt.date(y + (m == 12), (m % 12) + 1, 1)
        last = nxt - dt.timedelta(days=1)
        if first >= start_day and last <= end_day:
            out.append(("M", y, m))
        else:
            d = max(first, start_day)
            while d <= min(last, end_day):
                out.append(("D", d))
                d += dt.timedelta(days=1)
        y, m = nxt.year, nxt.month
    return out


def cache_last_day(path):
    if not os.path.exists(path):
        return None
    d = pd.read_parquet(path, columns=["day"])
    return pd.Timestamp(d["day"].max()).date()


# ------------------------------------------------------------------ klines
def _parse_klines(raw):
    df = _unzip_csv(raw, header=None)
    if str(df.iloc[0, 0]).startswith("open"):
        df = df.iloc[1:].reset_index(drop=True)
    df = df.iloc[:, :12]
    df.columns = COLS
    for c in ["o", "h", "l", "c", "v", "qv", "tbb", "tbq", "n"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ot = pd.to_numeric(df["ot"], errors="coerce").astype("int64")
    if ot.iloc[0] > 1e15:                       # 微秒 -> 毫秒 (2025+ dump)
        ot = ot // 1000
    df["dt"] = pd.to_datetime(ot, unit="ms", utc=True)
    return df.dropna(subset=["c", "v"])


def kline_url(sym, chunk):
    base = SBASE if sym in SPOT_SET else FBASE
    if chunk[0] == "M":
        _, y, m = chunk
        return f"{base}/monthly/klines/{sym}/1m/{sym}-1m-{y:04d}-{m:02d}.zip"
    d = chunk[1]
    return f"{base}/daily/klines/{sym}/1m/{sym}-1m-{d:%Y-%m-%d}.zip"


def update_klines(sym, end_day):
    fn = os.path.join(KDIR, sym + ".parquet")
    last = cache_last_day(fn)
    if last is None:
        return sym, "no-cache(用 pipeline 首建)"
    if last >= end_day:
        return sym, "up-to-date"
    frames = []
    for ch in plan_chunks(last + dt.timedelta(days=1), end_day):
        raw = _get(kline_url(sym, ch))
        if raw is None and ch[0] == "M":
            # 刚结束的月份: monthly zip 要月末后几天才发布, 未发布前改走 daily
            d = dt.date(ch[1], ch[2], 1)
            month_end = dt.date(ch[1] + (ch[2] == 12), ch[2] % 12 + 1, 1) - dt.timedelta(days=1)
            if month_end >= end_day - dt.timedelta(days=40):
                while d <= month_end:
                    r = _get(kline_url(sym, ("D", d)))
                    if r is not None:
                        frames.append(_parse_klines(r))
                    d += dt.timedelta(days=1)
                continue
        if raw is None:
            continue
        frames.append(_parse_klines(raw))
    if not frames:
        return sym, f"no-new(停在 {last})"
    dfm = pd.concat(frames, ignore_index=True)
    dfm["day"] = dfm["dt"].dt.floor("D")
    rows = []
    for day, g in dfm.groupby("day"):
        if day.date() <= last or day.date() > end_day:
            continue
        f = day_features(g)
        if f:
            f["day"] = day
            f["sym"] = sym
            rows.append(f)
    if not rows:
        return sym, f"no-new(停在 {last})"
    old = pd.read_parquet(fn)
    new = pd.DataFrame(rows)
    out = (pd.concat([old, new], ignore_index=True)
             .drop_duplicates(subset=["sym", "day"], keep="last")
             .sort_values("day").reset_index(drop=True))
    out.to_parquet(fn, index=False)
    return sym, f"+{len(rows)}d -> {out['day'].max().date()}"


# ------------------------------------------------------------------ funding
def update_funding(sym, end_month):
    """fundingRate 只有 monthly -> 只能补到 end_month 末。"""
    fn = os.path.join(FUNDDIR, sym + ".parquet")
    last = cache_last_day(fn)
    if last is None:
        return sym, "no-cache"
    y, m = last.year, last.month
    # 从 last 所在月开始重下(该月可能只覆盖了一部分), 逐月到 end_month
    frames = []
    while (y, m) <= end_month:
        url = f"{FBASE}/monthly/fundingRate/{sym}/{sym}-fundingRate-{y:04d}-{m:02d}.zip"
        raw = _get(url, 25)
        if raw is not None:
            df = _unzip_csv(raw)
            df["calc_time"] = pd.to_numeric(df["calc_time"], errors="coerce")
            df["last_funding_rate"] = pd.to_numeric(df["last_funding_rate"], errors="coerce")
            df["day"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True).dt.floor("D")
            frames.append(df[["day", "last_funding_rate"]])
        m += 1
        if m > 12:
            y, m = y + 1, 1
    if not frames:
        return sym, f"no-new(停在 {last})"
    f = pd.concat(frames, ignore_index=True)
    g = (f.groupby("day")["last_funding_rate"]
           .agg(FUND_sum="sum", FUND_mean="mean", FUND_n="count").reset_index())
    g["sym"] = sym
    g["FUND_src"] = "dump"
    old = pd.read_parquet(fn)
    if "FUND_src" not in old.columns:
        old["FUND_src"] = "dump"
    add = g[g["day"].dt.date > last]
    if not len(add):
        return sym, f"no-new(停在 {last})"
    out = (pd.concat([old, g], ignore_index=True)
             .drop_duplicates(subset=["sym", "day"], keep="last")
             .sort_values("day").reset_index(drop=True))
    out.to_parquet(fn, index=False)
    return sym, f"+{len(add)}d -> {out['day'].max().date()}"


# ------------------------------------------------------------------ funding proxy
# 币安公式: F = P + clamp(I - P, -cap_c, +cap_c);  I(8h)=0.01%, clamp=0.05%
# 4h 结算的币种按时长等比缩放 (I=0.005%, clamp=0.025%)。最后按 ±0.75%(8h)/±0.375%(4h) 截断。
def funding_proxy_days(sym, days, hours):
    """用 premiumIndexKlines 1m 反推 [days] 的日聚合 funding。hours=8 或 4。

    必须用 1m: 币安取的是结算窗口内**逐分钟** premium index 的均值。用 1h close 采样
    (每小时只取最后一分钟) 误差可达 0.6bp/日, 相关只 0.65~0.73; 1m 可到 0.98。
    """
    need = sorted(set(days) | {min(days) - dt.timedelta(days=1)})   # 跨零点窗口要前一天
    frames = []
    for d in need:
        raw = _get(f"{FBASE}/daily/premiumIndexKlines/{sym}/1m/{sym}-1m-{d:%Y-%m-%d}.zip", 25)
        if raw is None:
            continue
        df = _unzip_csv(raw, header=None)
        if str(df.iloc[0, 0]).startswith("open"):
            df = df.iloc[1:].reset_index(drop=True)
        df = df.iloc[:, :5]
        df.columns = ["ot", "o", "h", "l", "c"]
        ot = pd.to_numeric(df["ot"], errors="coerce").astype("int64")
        if ot.iloc[0] > 1e15:
            ot = ot // 1000
        df["dt"] = pd.to_datetime(ot, unit="ms", utc=True)
        df["prem"] = pd.to_numeric(df["c"], errors="coerce")
        frames.append(df[["dt", "prem"]])
    if not frames:
        return None
    px = pd.concat(frames, ignore_index=True).dropna().sort_values("dt")
    # 每根 1h 属于哪个结算窗口: 结算时刻 = 该小时之后下一个 hours 的整点边界。
    # floor("8h")/floor("4h") 以 epoch 对齐, 恰好落在 00/08/16 (或 00/04/.../20) UTC。
    px["settle"] = px["dt"].dt.floor(f"{hours}h") + pd.Timedelta(hours=hours)
    P = px.groupby("settle")["prem"].mean()
    I = 1e-4 * hours / 8.0
    clamp = 5e-4 * hours / 8.0
    cap = 7.5e-3 * hours / 8.0
    # 只保留分钟数足够完整的结算窗口 (缺 dump 的窗口均值不可信)
    ok = px.groupby("settle")["prem"].size() >= hours * 60 * 0.9
    P = P[ok.reindex(P.index, fill_value=False)]
    F = P + (I - P).clip(-clamp, clamp)
    F = F.clip(-cap, cap)
    out = (F.rename("r").reset_index())
    out["day"] = out["settle"].dt.floor("D")
    g = out.groupby("day")["r"].agg(FUND_sum="sum", FUND_mean="mean", FUND_n="count").reset_index()
    g = g[g["FUND_n"] == 24 // hours]        # 当日结算次数必须齐全, 否则 FUND_sum 会被低估
    keep = set(days)
    return g[[d in keep for d in g["day"].dt.date]].reset_index(drop=True)


def sym_funding_hours(sym):
    """从缓存最近 30 天的 FUND_n 判断结算间隔 (3/天->8h, 6/天->4h)。"""
    fn = os.path.join(FUNDDIR, sym + ".parquet")
    if not os.path.exists(fn):
        return None
    d = pd.read_parquet(fn)
    d = d[d["day"] >= d["day"].max() - pd.Timedelta(days=30)]
    n = d["FUND_n"].median()
    if n >= 5:
        return 4
    if n >= 2:
        return 8
    return None


# ------------------------------------------------------------------ OI
def update_oi(sym, end_day):
    fn = os.path.join(OIDIR, sym + ".parquet")
    last = cache_last_day(fn)
    if last is None:
        return sym, "no-cache"
    if last >= end_day:
        return sym, "up-to-date"
    rows = []
    day = last + dt.timedelta(days=1)
    while day <= end_day:
        raw = _get(f"{FBASE}/daily/metrics/{sym}/{sym}-metrics-{day:%Y-%m-%d}.zip", 25)
        if raw is not None:
            m = _unzip_csv(raw)
            if len(m):
                oiv = pd.to_numeric(m["sum_open_interest_value"], errors="coerce")
                rows.append(dict(
                    day=pd.Timestamp(day, tz="UTC"),
                    OI_close=float(oiv.iloc[-1]), OI_mean=float(oiv.mean()),
                    GLS=float(pd.to_numeric(m["count_long_short_ratio"], errors="coerce").mean()),
                    TLS=float(pd.to_numeric(m["sum_toptrader_long_short_ratio"], errors="coerce").mean()),
                    TKLS=float(pd.to_numeric(m["sum_taker_long_short_vol_ratio"], errors="coerce").mean()),
                ))
        day += dt.timedelta(days=1)
    if not rows:
        return sym, f"no-new(停在 {last})"
    new = pd.DataFrame(rows)
    new["sym"] = sym
    out = (pd.concat([pd.read_parquet(fn), new], ignore_index=True)
             .drop_duplicates(subset=["sym", "day"], keep="last")
             .sort_values("day").reset_index(drop=True))
    out.to_parquet(fn, index=False)
    return sym, f"+{len(rows)}d -> {out['day'].max().date()}"


# ------------------------------------------------------------------ 探测最新可得日
def latest_available(kind="klines", back=8):
    today = dt.date.today()
    for i in range(back):
        d = today - dt.timedelta(days=i)
        if kind == "klines":
            # 现货和永续的 daily 发布时间不同步; 取两者都已发布的日期, 否则末日截面只剩现货币
            u = f"{SBASE}/daily/klines/BTCUSDT/1m/BTCUSDT-1m-{d:%Y-%m-%d}.zip"
            if not _exists(f"{FBASE}/daily/klines/BTCUSDT/1m/BTCUSDT-1m-{d:%Y-%m-%d}.zip"):
                continue
        elif kind == "metrics":
            u = f"{FBASE}/daily/metrics/BTCUSDT/BTCUSDT-metrics-{d:%Y-%m-%d}.zip"
        else:
            u = f"{FBASE}/daily/premiumIndexKlines/BTCUSDT/1h/BTCUSDT-1h-{d:%Y-%m-%d}.zip"
        if _exists(u):
            return d
    return None


def latest_funding_month():
    today = dt.date.today()
    y, m = today.year, today.month
    for _ in range(3):
        if _exists(f"{FBASE}/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{y:04d}-{m:02d}.zip"):
            return (y, m)
        m -= 1
        if m < 1:
            y, m = y - 1, 12
    return None


# ------------------------------------------------------------------ runners
def run_pool(fn, pool, label, workers=10, **kw):
    print(f"[{label}] {len(pool)} 币种, {workers} 线程")
    done, changed = 0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, s, **kw): s for s in pool}
        for fu in as_completed(futs):
            sym = futs[fu]
            try:
                s, msg = fu.result()
            except Exception as e:
                s, msg = sym, f"FAIL {type(e).__name__}:{str(e)[:50]}"
            done += 1
            if msg.startswith("+"):
                changed += 1
            print(f"  [{done:3d}/{len(pool)}] {s:14s} {msg}", flush=True)
    print(f"[{label}] 完成: {changed}/{len(pool)} 有新增。")


def cached_syms(d):
    return sorted(f[:-8] for f in os.listdir(d) if f.endswith(".parquet"))


def cmd_probe():
    print("=" * 76)
    print("数据源最新可得 (data.binance.vision)")
    print("=" * 76)
    for k, lbl in [("klines", "现货/永续 1m daily klines"), ("metrics", "OI metrics daily"),
                   ("premium", "premiumIndexKlines 1h daily")]:
        print(f"  {lbl:32s} -> {latest_available(k)}")
    print(f"  {'fundingRate monthly (最新完整月)':32s} -> {latest_funding_month()}")
    print("\n本地缓存现状:")
    for d, nm in [(KDIR, "klines"), (FUNDDIR, "funding"), (OIDIR, "oi")]:
        syms = cached_syms(d)
        lasts = [cache_last_day(os.path.join(d, s + ".parquet")) for s in syms]
        import collections
        c = collections.Counter(str(x) for x in lasts)
        print(f"  {nm:8s} {len(syms):3d} 币  末日分布 {c.most_common(4)}")


def cmd_validate():
    """在上一个完整月上比对 proxy vs 真实 funding。"""
    fm = latest_funding_month()
    y, m = fm
    first = dt.date(y, m, 1)
    nxt = dt.date(y + (m == 12), (m % 12) + 1, 1)
    days = [first + dt.timedelta(days=i) for i in range((nxt - first).days)]
    syms = [s for s in cached_syms(FUNDDIR)]
    print(f"[validate] 用 {y}-{m:02d} ({len(days)}天) 全池 {len(syms)} 币 校验 funding proxy")
    recs = []

    def one(sym):
        h = sym_funding_hours(sym)
        if h is None:
            return None
        p = funding_proxy_days(sym, days, h)
        if p is None or not len(p):
            return None
        a = pd.read_parquet(os.path.join(FUNDDIR, sym + ".parquet"))
        a = a[(a["day"] >= pd.Timestamp(first, tz="UTC")) & (a["day"] < pd.Timestamp(nxt, tz="UTC"))]
        mg = a[["day", "FUND_sum"]].merge(p[["day", "FUND_sum"]], on="day", suffixes=("_act", "_prx"))
        mg["sym"] = sym
        return mg

    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(one, s): s for s in syms}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:
                print(f"  {futs[fu]} FAIL {type(e).__name__}:{str(e)[:40]}"); r = None
            if r is not None and len(r):
                recs.append(r)
            if i % 20 == 0:
                print(f"  ...{i}/{len(syms)}", flush=True)
    D = pd.concat(recs, ignore_index=True)
    D.to_csv(os.path.join(ROOT, "funding_proxy_validation.csv"), index=False)
    print("\n" + "=" * 76)
    print(f"样本 {len(D)} 币日  ({D['sym'].nunique()} 币 × {D['day'].nunique()} 天)")
    print(f"  逐点相关       corr(actual, proxy) = {D['FUND_sum_act'].corr(D['FUND_sum_prx']):.4f}")
    err = (D["FUND_sum_prx"] - D["FUND_sum_act"]) * 1e4
    print(f"  误差(bp/日)    均值 {err.mean():+.3f}  中位 {err.median():+.3f}  |误差|中位 {err.abs().median():.3f}")
    # 因子真正用的是"每日横截面排名", 且 funding 因子只跑大币池 -> 两个池都看
    for lbl, sub in [("全池", D), ("大币池24", D[D["sym"].isin(BIG_POOL)])]:
        xs = sub.groupby("day").apply(
            lambda g: g["FUND_sum_act"].corr(g["FUND_sum_prx"], method="spearman"),
            include_groups=False)
        # 因子实际用 5 日滚动均值, 平滑后保真度更高
        w = (sub.sort_values("day").groupby("sym")[["FUND_sum_act", "FUND_sum_prx"]]
                .rolling(5, min_periods=2).mean().reset_index())
        w["day"] = sub.sort_values("day").reset_index(drop=True)["day"].values
        xs5 = w.groupby("day").apply(
            lambda g: g["FUND_sum_act"].corr(g["FUND_sum_prx"], method="spearman"),
            include_groups=False).dropna()
        print(f"  [{lbl}] 横截面 Spearman: 日频 均值 {xs.mean():.4f} 最小 {xs.min():.4f} "
              f"(<0.9 的天数 {int((xs < 0.9).sum())}/{len(xs)})")
        print(f"           因子口径(5日滚动均值): 均值 {xs5.mean():.4f} 最小 {xs5.min():.4f}")
    print("=" * 76)
    print("判定: 因子口径秩相关均值 >0.95 且最小 >0.85 -> proxy 可用于补当月缺口; 否则不要用。")


def cmd_proxy(end_day):
    """把最后一个完整月之后的日子用 proxy 补进 funding parquet, 标 FUND_src='proxy'。"""
    y, m = latest_funding_month()
    nxt = dt.date(y + (m == 12), (m % 12) + 1, 1)
    if nxt > end_day:
        print("[proxy] 无缺口"); return
    days = [nxt + dt.timedelta(days=i) for i in range((end_day - nxt).days + 1)]
    syms = cached_syms(FUNDDIR)
    print(f"[proxy] 补 {days[0]} ~ {days[-1]} ({len(days)}天) × {len(syms)} 币")

    def one(sym):
        fn = os.path.join(FUNDDIR, sym + ".parquet")
        last = cache_last_day(fn)
        if last is None or last < nxt - dt.timedelta(days=1):
            return sym, f"skip(缓存停在 {last}, 非当前币)"
        need = [d for d in days if d > last]
        if not need:
            return sym, "up-to-date"
        h = sym_funding_hours(sym)
        if h is None:
            return sym, "skip(无法判定结算间隔)"
        p = funding_proxy_days(sym, need, h)
        if p is None or not len(p):
            return sym, "no-new"
        p["sym"] = sym
        p["FUND_src"] = "proxy"
        old = pd.read_parquet(fn)
        if "FUND_src" not in old.columns:
            old["FUND_src"] = "dump"
        out = (pd.concat([old, p], ignore_index=True)
                 .drop_duplicates(subset=["sym", "day"], keep="first")   # dump 优先
                 .sort_values("day").reset_index(drop=True))
        out.to_parquet(fn, index=False)
        return sym, f"+{len(p)}d -> {out['day'].max().date()}"

    run_pool(lambda s: one(s), syms, "funding proxy", workers=12)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if cmd == "probe":
        return cmd_probe()
    if cmd == "validate":
        return cmd_validate()

    end_k = latest_available("klines")
    end_o = latest_available("metrics")
    fm = latest_funding_month()
    print(f"目标: klines->{end_k}  metrics->{end_o}  funding(monthly)->{fm}")

    if cmd == "proxy":
        return cmd_proxy(end_k)
    if cmd in ("klines", "all"):
        run_pool(update_klines, cached_syms(KDIR), "klines 增量", workers=8, end_day=end_k)
    if cmd in ("funding", "all"):
        run_pool(update_funding, cached_syms(FUNDDIR), "funding 增量", workers=12, end_month=fm)
    if cmd in ("oi", "all"):
        run_pool(update_oi, [s for s in cached_syms(OIDIR)], "OI 增量", workers=12, end_day=end_o)


if __name__ == "__main__":
    main()
