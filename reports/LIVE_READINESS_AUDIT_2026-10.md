# Live-Trading Reliability & Implementation Shortfall Audit (`2024-06 ~ 2026-10-21`)

**Scope**: Empirical stress testing of `CryptoFactors` single factors, Deliverable Product B (`deploy_spot.py`), and the v2.0 regime-conditioned composite (`inv_vol_funding_gated_voltarget_15pct`) under realistic live execution frictions:
1. **Execution Delay**: `0m` (ideal 00:00 UTC close) vs `1m` (`00:01 UTC` entry reconstructed from 1m klines `p0 = close / (1 + ret)`), `12h` (`12:00 UTC` entry), and `24h` (`+1d` lag);
2. **Transaction Cost Grid & Break-Even Threshold**: `5 bps`, `15 bps`, `30 bps`, `50 bps` per trade plus exact break-even cost (`breakeven_bps_full`);
3. **Realistic Perpetual Hedge Legs & AUM Capacity**: Ideal 24-coin equal-weight index subtraction vs realistic 24-perp basket short (with 8h funding accrual and rebalance fees) vs BTC/ETH 50/50 perpetual shorts, and square-root market impact across `$10k ~ $5M` AUM;
4. **Pre-Trade Liquidity & Bad-Print Dislocation Guard**: Filtering single-venue price dislocations (`QNTUSDT`, `SAGAUSDT`) via trailing 7-day ADV $\ge \$2\text{M}$ and daily return outlier guards.

---

## 1. Single-Factor Execution Delay & Break-Even Cost Matrix (`reports/live_readiness_2026_10/cost_and_delay_stress.csv`)

| Factor (with Funding Carry) | Daily Turnover | Break-Even Cost (bps) | `0m` Ideal (5 bps) Sharpe / Cum | `1m` Delay `00:01 UTC` (5 bps) Sharpe / Cum | `1m` Delay `00:01 UTC` (15 bps) Sharpe / Cum | `24h` Delay `+1d` (5 bps) Sharpe / Cum | Post-Pub 142d (`1m`, 15 bps) | Sep–Oct 50d (`1m`, 15 bps) | Live Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`散户多空比(大)` (`GLS_f`)** | **13.4%** | **99.7 bps** | **1.44** (+110.18%) | **1.43** (+108.83%) | **1.27** (+91.24%) | **1.99** (+181.21%) | **+17.32%** (Sharpe 1.63) | **+10.58%** | **PASS (Tier-1 Live Signal)**: ultra-low turnover, ~100 bps cost buffer, zero delay decay |
| **`资金费率(大)` (`FUND_f`)** | 38.6% | **44.8 bps** | **1.37** (+213.02%) | **1.35** (+209.96%) | **1.01** (+121.68%) | **1.20** (+148.58%) | -6.78% | **+6.47%** | **PASS with Weekly Rebalance / Maker Execution ($\le 15$ bps)** |
| **`低价位振幅(宽)` (`AMP_lo_f`)** | 15.4% | **69.1 bps** | **0.86** (+91.77%) | **0.88** (+95.38%) | **0.75** (+70.86%) | **0.69** (+61.14%) | -14.87% | -11.43% | **PASS on Cost/Delay**, but vulnerable to altcoin short-squeeze regimes |
| `低价位振幅(大)` (`AMP_lo_f`, 24 coins) | 18.0% | 10.0 bps | 0.06 (-26.52%) | 0.06 (-26.44%) | -0.06 (-37.16%) | 0.12 (-20.43%) | -34.37% | -34.89% | **FAIL Alone in 24-Coin Pool** (insufficient cross-sectional dispersion) |
| `大额主动买(大)` (`OFIbig_f`) | 57.8% | 14.8 bps | 0.60 (+42.63%) | 0.57 (+39.13%) | -0.04 (-15.86%) | 0.14 (-3.86%) | -17.12% | -18.11% | **FAIL Live Execution** (wiped out at 15 bps or >12h lag) |
| `持仓量变化(大)` (`dOI_f`) | **96.9%** | **8.9 bps** | 0.35 (+11.99%) | 0.24 (+3.41%) | -0.65 (-45.14%) | 0.02 (-14.41%) | -13.16% | -22.13% | **FAIL Live Execution** (96.9% daily turnover destroys net returns above 9 bps) |

---

## 2. v2.0 Regime-Conditioned Composite Delay & Cost Stress (`reports/live_readiness_2026_10/v2_composite_delay_cost_stress.csv`)

Evaluated on the 3-factor pre-trade guarded composite (`inv_vol_funding_gated_voltarget_15pct`):

| Execution Delay Mode | Cost (bps) | Full 835d Cum (%) | Full 835d Sharpe | Full 835d MaxDD (%) | Post-Pub 142d Cum (%) | Sep–Oct 50d Cum (%) | Since Sep 28 (22d, %) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0m_ideal_close` | 5.0 | +72.34 | 1.48 | -14.32 | +2.30 | +0.72 | +3.15 |
| `1m_delay_00:01_24h` | 5.0 | **+70.76** | **1.46** | -15.13 | +1.66 | +0.67 | **+3.13** |
| `1m_delay_00:01_24h` | 15.0 | **+53.14** | **1.18** | -15.83 | -0.97 | -0.57 | **+2.26** |
| `1m_delay_00:01_24h` | 30.0 | +28.26 | 0.72 | -18.39 | -4.31 | -2.10 | +1.69 |
| `1m_delay_00:01_24h` | 50.0 | +1.14 | 0.11 | -24.83 | -8.91 | -3.66 | +0.96 |
| `24h_delay_next_day` | 15.0 | **+54.92** | **1.23** | **-14.79** | **+1.75** | **+2.43** | **+1.41** |

---

## 3. Deliverable Product B (`产品B·大币top8·对冲beta`) Realistic Perpetual Hedge Legs & AUM Capacity (`product_b_realistic_hedges.csv`, `product_b_capacity_stress.csv`)

| Hedge Leg Specification | Execution Timing | Full 537d Cum (%) | Full 537d Sharpe | Full 537d MaxDD (%) | Post-Pub 142d (%) | Sep–Oct 50d (%) | Oct 20d (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| `ideal_ew24_no_funding` (Backtest baseline) | `0m_close` | +30.56 | 0.86 | -12.45 | -2.61 | +4.21 | +2.70 |
| `ideal_ew24_no_funding` | `1m_00:01_delay` | +30.51 | 0.86 | -12.60 | -3.05 | +3.91 | +2.37 |
| **`real_ew24_perp_with_funding` (24-Perp Short + Carry + Fees)** | **`1m_00:01_delay`** | **+28.41** | **0.81** | **-13.25** | **-3.24** | **+4.03** | **+2.37** |
| `real_btc_eth_50_50_perp` (1x Short BTC/ETH 50/50) | `1m_00:01_delay` | -8.52 | -0.03 | -40.44 | +2.28 | +29.97 | +3.97 |
| `real_btc_eth_beta_adjusted_perp` (Rolling 60d Beta BTC/ETH) | `1m_00:01_delay` | +11.22 | 0.35 | -30.35 | +3.57 | +29.71 | +3.46 |

### AUM Capacity Stress (with `real_ew24_perp_with_funding`, `1m_00:01_delay`, 15 bps base fee + square-root market impact):
* **`$10,000` AUM**: Full 537d **`+27.93%`** (Sharpe **0.80**, MaxDD `-13.43%`, Sep–Oct 50d **`+3.98%`**)
* **`$100,000` AUM**: Full 537d **`+26.90%`** (Sharpe **0.78**, MaxDD `-13.83%`, Sep–Oct 50d **`+3.87%`**)
* **`$500,000` AUM**: Full 537d **`+25.06%`** (Sharpe **0.74**, MaxDD `-14.53%`, Sep–Oct 50d **`+3.67%`**)
* **`$1,000,000` AUM**: Full 537d **`+23.69%`** (Sharpe **0.70**, MaxDD `-15.06%`, Sep–Oct 50d **`+3.52%`**)
* **`$5,000,000` AUM**: Full 537d **`+18.09%`** (Sharpe **0.57**, MaxDD `-17.24%`, Sep–Oct 50d **`+2.89%`**)

---

## 4. Pre-Trade Liquidity & Bad-Print Dislocation Guard (`bad_print_dislocation_guard.csv`)

Enforcing `adv7_usd >= $2M`, `|ret_t| <= 40%`, and `±50%` single-day forward outlier protection (at `1m` delay + `10 bps` cost):
* **Improves `散户多空比(大)` Sharpe from `1.35` to `1.40`** while keeping October 2026 return unchanged (`+7.81%`).
* **Exposes Wide-Pool Illiquidity Illusion**: In September 2026, raw `资金费率(宽)` appeared slightly positive (`+0.85%`) solely due to `QNTUSDT`/`SAGAUSDT` single-venue dislocations; after pre-trade dislocation filtering, wide-pool funding was `-16.05%` in September, whereas **large-cap `资金费率(大)` remained genuinely profitable (`+4.29%` raw / `+1.94%` guarded in Sep, `+3.28%` in Oct)**.
