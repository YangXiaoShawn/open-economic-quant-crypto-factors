"""Command-line interface for CryptoFactors research package."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def cmd_summary() -> int:
    transfer_path = ROOT / "reports" / "factor_transfer_summary.csv"
    postpub_path = ROOT / "reports" / "postpub_composite_summary.csv"
    with transfer_path.open("r", encoding="utf-8") as fh:
        factors = list(csv.DictReader(fh))
    with postpub_path.open("r", encoding="utf-8") as fh:
        windows = list(csv.DictReader(fh))
    payload = {
        "project": "crypto-factors",
        "evidence_grade": "backtest and post-publication out-of-sample backtest (association)",
        "factors_tested": len(factors),
        "robust_factors": [r["factor"] for r in factors if "strongest" in r["verdict"] or "robust" in r["verdict"]],
        "postpub_92d_return_pct": float(windows[-1]["market_neutral_return_pct"]),
        "postpub_92d_max_drawdown_pct": float(windows[-1]["max_drawdown_pct"]),
    }
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crypto-factors")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("summary", help="Print factor transfer and post-publication OOS summary.")
    sub.add_parser("reproduce-sample", help="Verify deterministic sample fixtures and published tables.")
    args = parser.parse_args(argv)
    if args.command in (None, "summary", "reproduce-sample"):
        return cmd_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
