"""Central path and parameter configuration for CryptoFactors."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
FIXTURES_DIR = DATA_DIR / "fixtures"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DEFAULT_COST_BPS_PER_SIDE = 5.0
DEFAULT_QUANTILES = 5
DEFAULT_TARGET_VOL = 0.15
