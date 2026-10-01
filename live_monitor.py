# -*- coding: utf-8 -*-
"""
实时检测后端 —— 把"经样本外验证的王牌信号"接到 OKX 实时行情, 浏览器看板用。

只用标准库 http.server (无需 Flask), ccxt 直连 OKX (本环境唯一可用所)。
信号 = OOS 验证最稳的两个跨所因子:
  • 资金费率 funding (王牌): 高 funding = 杠杆多头拥挤 → 看空; 低/负 = 看多。
  • 振幅 amplitude (24h H-L/Last): 低振幅(平静) → 看多腿; 高振幅(躁动) → 看空腿。
复合分 = z(−funding) + z(−amplitude); 越高越是 LONG 候选, 越低越是 SHORT 候选。
全市场平均 funding 作 risk-on/off 拥挤度计 (见 funding_deep.py: 极高拥挤→大盘见顶风险)。

跑: python live_monitor.py   然后浏览器开 http://127.0.0.1:8765
"""
import json, time, threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import numpy as np
import ccxt

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = 8765
POOL = ["BTC","ETH","SOL","XRP","DOGE","ADA","AVAX","LINK","DOT","TRX","LTC","BCH",
        "NEAR","APT","ARB","OP","FIL","ATOM","UNI","AAVE","INJ","ETC","XLM","SUI",
        "TIA","SEI","WLD","LDO","ICP","RENDER"]
TOPK = 6
CACHE_TTL = 30                      # 秒, 避免频繁打 API
FUND_INTERVALS_PER_DAY = 3         # OKX 8h 结算

_ex = ccxt.okx({"enableRateLimit": True})
_lock = threading.Lock()
_cache = {"t": 0, "data": None}
_hist = deque(maxlen=120)          # (ts, 全市场平均funding年化%)
_equity = {"t": 0, "hist": None, "stats": None}   # 市场中性策略净值(缓存)
EQ_TTL = 3600                      # 历史净值用日线, 1h 重算足够


def _z(x):
    x = np.asarray(x, float)
    mu, sd = np.nanmean(x), np.nanstd(x)
    return (x - mu) / (sd + 1e-12)


def build_strategy_equity(top_k=TOPK, rebal=7, cost_bps=15):
    """
    用 OKX 日线缓存 (data_okx/okx_panel.parquet, 由 oos_okx.py 生成) 回算"市场中性"策略日净值:
    复合分 z(−funding)+z(−振幅) → top_k 等权多头 − 全池等权 (= 对冲 beta 的纯 alpha), 周调仓。
    funding 历史经 ccxt 仅 ~3 月 → 曲线约 90 个交易日 (OKX 独立交易所的近窗实证)。
    """
    fn = os.path.join(ROOT, "data_okx", "okx_panel.parquet")
    if not os.path.exists(fn):
        return [], None
    import pandas as pd
    df = pd.read_parquet(fn)
    recs = []
    for c, d in df.groupby("sym"):
        d = d.sort_values("day").reset_index(drop=True)
        d["fwd_ret"] = d["c"].shift(-1) / d["c"] - 1.0
        d["AMP_f"] = ((d["h"] - d["l"]) / d["c"]).rolling(5, min_periods=2).mean()
        d["FUND_f"] = d["FUND_sum"].rolling(5, min_periods=2).mean()
        recs.append(d)
    p = pd.concat(recs, ignore_index=True).dropna(subset=["FUND_f", "AMP_f", "fwd_ret"])
    zt = lambda s: (s - s.mean()) / (s.std() + 1e-9)
    p["zc"] = -p.groupby("day")["FUND_f"].transform(zt) - p.groupby("day")["AMP_f"].transform(zt)
    held, prev, rows = set(), set(), []
    for i, (day, g) in enumerate(p.groupby("day")):
        if g["sym"].nunique() < top_k * 2:
            continue
        if i % rebal == 0:
            held = set(g.nlargest(top_k, "zc")["sym"])
        sub = g[g["sym"].isin(held)]
        if not len(sub):
            continue
        to = len(held ^ prev) / max(len(held), 1) if i % rebal == 0 else 0.0
        r = sub["fwd_ret"].mean() - g["fwd_ret"].mean() - to * cost_bps / 1e4   # 对冲beta
        rows.append((day, float(r)))
        prev = held
    if len(rows) < 5:
        return [], None
    eq, v = [], 1.0
    rets = np.array([r for _, r in rows])
    for (day, r) in rows:
        v *= (1 + r)
        eq.append([day.strftime("%m-%d"), round(v, 4)])
    ann = (eq[-1][1]) ** (365 / len(rets)) - 1
    sharpe = float(rets.mean() / (rets.std() + 1e-12) * np.sqrt(365))
    cum = np.cumprod(1 + rets); mdd = float((cum / np.maximum.accumulate(cum) - 1).min())
    stats = {"sharpe": round(sharpe, 2), "ann": round(ann * 100, 0),
             "mdd": round(mdd * 100, 0), "ndays": len(rets), "tot": round(eq[-1][1], 3)}
    return eq, stats


def get_equity():
    now = time.time()
    if _equity["hist"] is None or now - _equity["t"] > EQ_TTL:
        try:
            _equity["hist"], _equity["stats"] = build_strategy_equity()
        except Exception:
            _equity["hist"], _equity["stats"] = [], None
        _equity["t"] = now
    return _equity["hist"], _equity["stats"]


def compute_snapshot():
    spot = [f"{c}/USDT" for c in POOL]
    swap = [f"{c}/USDT:USDT" for c in POOL]
    tk = _ex.fetch_tickers(spot)
    fr = _ex.fetch_funding_rates(swap)
    rows = []
    for c in POOL:
        t = tk.get(f"{c}/USDT"); f = fr.get(f"{c}/USDT:USDT")
        if not t or not f or not t.get("last"):
            continue
        last = float(t["last"]); hi = float(t.get("high") or last); lo = float(t.get("low") or last)
        op = float(t.get("open") or last)
        amp = (hi - lo) / last if last else np.nan
        ret = (last / op - 1.0) if op else np.nan
        rate = f.get("fundingRate")
        if rate is None:
            continue
        rows.append(dict(coin=c, price=last, amp=amp, ret=ret,
                         fund=float(rate), fund_ann=float(rate) * FUND_INTERVALS_PER_DAY * 365))
    if len(rows) < TOPK * 2:
        raise RuntimeError("行情样本不足")
    fund = np.array([r["fund"] for r in rows])
    amp = np.array([r["amp"] for r in rows])
    zf, za = _z(fund), _z(amp)
    for i, r in enumerate(rows):
        r["fund_z"] = float(zf[i])
        r["amp_z"] = float(za[i])
        r["score"] = float(-zf[i] - za[i])        # 低funding+低振幅 → 高分(LONG候选)
    rows.sort(key=lambda r: r["score"], reverse=True)
    longs = [r["coin"] for r in rows[:TOPK]]
    shorts = [r["coin"] for r in rows[-TOPK:]]
    for r in rows:
        r["signal"] = "LONG" if r["coin"] in longs else ("SHORT" if r["coin"] in shorts else "")

    # ---- 目标权重 / 下单清单 ----
    wl = round(1.0 / TOPK, 4)
    ret_by = {r["coin"]: r["ret"] for r in rows}
    long_w = [{"coin": c, "w": wl, "px": next(x["price"] for x in rows if x["coin"] == c)} for c in longs]
    short_w = [{"coin": c, "w": -wl, "px": next(x["price"] for x in rows if x["coin"] == c)} for c in shorts]
    # ---- 实时组合净值: 历史日线净值 + 当日实时 mark ----
    eq_hist, eq_stats = get_equity()
    pool_ret = float(np.mean([r["ret"] for r in rows]))
    long_ret = float(np.mean([ret_by[c] for c in longs]))
    live_hedged = long_ret - pool_ret                 # 市场中性当日实时收益(24h口径)
    equity = [list(p) for p in eq_hist]
    base = equity[-1][1] if equity else 1.0
    equity.append(["实时", round(base * (1 + live_hedged), 4)])
    net_value = equity[-1][1]

    agg_ann = float(np.mean([r["fund_ann"] for r in rows])) * 100
    _hist.append((int(time.time() * 1000), round(agg_ann, 2)))
    # 拥挤度判定 (年化funding %)
    if agg_ann > 30:
        regime, rcolor = "过热·拥挤多头 (风险偏空)", "hot"
    elif agg_ann > 10:
        regime, rcolor = "偏热", "warm"
    elif agg_ann < -5:
        regime, rcolor = "出清·负费率 (风险偏多)", "cold"
    else:
        regime, rcolor = "中性", "neutral"

    return {
        "ts": int(time.time() * 1000),
        "pool": len(rows),
        "rows": rows,
        "longs": longs, "shorts": shorts,
        "long_w": long_w, "short_w": short_w,
        "hedge": {"name": "大币等权指数 perp (≈ short BTC/ETH 或各 −1/N)", "w": -1.0},
        "agg_funding_ann": round(agg_ann, 2),
        "regime": regime, "regime_color": rcolor,
        "hist": list(_hist),
        "equity": equity, "net_value": net_value,
        "live_pnl": round(live_hedged * 100, 2),
        "strat": eq_stats or {},
    }


def get_signals():
    now = time.time()
    with _lock:
        if _cache["data"] and now - _cache["t"] < CACHE_TTL:
            return _cache["data"]
    data = compute_snapshot()           # 网络调用放锁外
    with _lock:
        _cache["t"], _cache["data"] = now, data
    return data


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):          # 静默
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode("utf-8"))

    def do_GET(self):
        if self.path.startswith("/api/signals"):
            try:
                self._send(200, json.dumps(get_signals(), ensure_ascii=False), "application/json; charset=utf-8")
            except Exception as e:
                self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False),
                           "application/json; charset=utf-8")
        elif self.path in ("/", "/index.html", "/dashboard.html"):
            try:
                with open(os.path.join(ROOT, "dashboard.html"), "rb") as fh:
                    self._send(200, fh.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(404, "dashboard.html not found", "text/plain")
        else:
            self._send(404, "not found", "text/plain")


def main():
    print(f"实时检测后端启动: http://127.0.0.1:{PORT}  (OKX live, 池 {len(POOL)} 币, 缓存 {CACHE_TTL}s)")
    print("浏览器打开上面地址即看实时看板。Ctrl+C 退出。")
    try:                                 # 预热一次, 早暴露连通性问题
        d = get_signals()
        print(f"预热成功: {d['pool']} 币, 全市场funding {d['agg_funding_ann']:+.1f}%/年 ({d['regime']})")
    except Exception as e:
        print(f"预热失败(看板仍会启动, 稍后重试): {type(e).__name__}: {e}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
