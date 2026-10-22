# Methodology & Post-Publication Tracking Protocol

1. **Cross-Market Transfer Design**: Translate 32 equity microstructure factor specifications (originally developed on China A-share minute and order-flow data) to 105 crypto spot and perpetual-futures pairs on Binance (730 days, 2024-06 to 2026-05), charging 5 bp per side on daily quintile portfolios.
2. **Multi-Gate Robustness**: Require sign consistency across sample halves, parameter stability across lookback horizons, rolling-window positivity, universe-breadth checks, and cross-exchange confirmation on OKX.
3. **Frozen Post-Publication Out-of-Sample Tracking**: After the 2026-06-19 report cutoff (data through 2026-05-31), freeze the equal-weighted z-score composite (funding rate + retail long/short ratio + low-price range, top 8 large-cap coins, weekly rebalance, 15 bp cost) and track June, July, August, and September 2026 without re-fitting weights.
