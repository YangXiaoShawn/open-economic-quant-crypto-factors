# Status

Last updated: 2026-10-26

## Current evidence

- **Evidence grade**: Backtest and post-publication out-of-sample backtest association (`PUBLIC_RESEARCH_SNAPSHOT`).
- **Pre-publication panel**: 105 coins x 730 days (2024-06 to 2026-05) on Binance spot and USDT-margined perpetuals, plus OKX cross-exchange replication.
- **Surviving signals**: Perpetual funding rate (730d Sharpe +1.42, 91% rolling windows positive, confirmed on OKX), retail long/short ratio (+1.39), and low-price range (+1.63). Cross-sectional momentum (-0.38) and top-trader ratio (-1.36) fail.
- **Cost erratum (2026-10-26)**: The published long-short Sharpe ratios above and in the reports charged about 2.5 bp per side instead of the stated 5 bp (the formula averaged the two legs' turnover instead of summing it), and the v2.0 composite overlays charged nothing for re-weighting. The code now charges both correctly (`src/crypto_factors/costs.py`); the factor tables await a rerun on the raw Binance panels, which are not in the repository or the public dataset. See the README for the v2.0 composite's overlay-charged figures and the 2026-10 audit's cost and delay stress results.
- **Post-publication tracking (frozen recipe, no re-fitting)**: Over June-August 2026 (92 days), the market-neutral composite returned +4.6% in June, -6.9% in July, and -4.0% in August (-6.5% cumulative, max drawdown -11.9%). Only retail positioning remained positive (+11.8% L/S through September), while funding (-13.4%) and range (-9.7%) suffered during altcoin short squeezes.
