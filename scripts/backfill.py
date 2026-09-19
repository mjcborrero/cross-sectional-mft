"""Backfill Binance historical data into local parquet files.

Examples:
    python scripts/backfill.py --datasets funding --symbols BTCUSDT --start 2024-01 --end 2024-03
    python scripts/backfill.py --datasets all --symbols universe --start 2020-01
    python scripts/backfill.py --datasets all --symbols universe --dry-run
    python scripts/backfill.py --delivery            # BTC/ETH dated futures
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mft.paths import DATA_DIR
from mft.data.client import BinanceVisionClient
from mft.data.manifest import Manifest
from mft.data.planner import discover_delivery_symbols, plan_symbol
from mft.data.runner import BackfillRunner
from mft.data.specs import (
    DATASETS, DEFAULT_DATASETS, DELIVERY_UNDERLYINGS_CM, DELIVERY_UNDERLYINGS_UM,
)

UNIVERSE_FILE = REPO_ROOT / "config" / "universe.txt"


def load_universe() -> list:
    symbols = []
    for line in UNIVERSE_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            symbols.append(line.upper())
    return symbols


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="all",
                        help="comma list or 'all' (= %s)" % ",".join(DEFAULT_DATASETS))
    parser.add_argument("--symbols", default="universe",
                        help="comma list or 'universe' (config/universe.txt)")
    parser.add_argument("--start", default="2020-01", help="inclusive, YYYY-MM")
    parser.add_argument("--end", default=None, help="inclusive, YYYY-MM (default: now)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--dry-run", action="store_true", help="plan only, no downloads")
    parser.add_argument("--delivery", action="store_true",
                        help="backfill dated BTC/ETH delivery futures klines "
                             "(discovers contract symbols; ignores --datasets/--symbols)")
    args = parser.parse_args()

    end = args.end or datetime.now(timezone.utc).strftime("%Y-%m")
    data_dir = Path(args.data_dir)
    client = BinanceVisionClient()
    manifest = Manifest(data_dir / "manifest.db")
    done = manifest.converted_keys()

    # Build the (dataset, symbol) work list.
    work = []
    if args.delivery:
        for ds_name, underlyings in (("um_delivery_klines", DELIVERY_UNDERLYINGS_UM),
                                     ("cm_delivery_klines", DELIVERY_UNDERLYINGS_CM)):
            spec = DATASETS[ds_name]
            for underlying in underlyings:
                contracts = discover_delivery_symbols(client, spec, underlying)
                print(f"{ds_name}: {underlying} -> {len(contracts)} contracts")
                work.extend((ds_name, c) for c in contracts)
    else:
        names = list(DEFAULT_DATASETS) if args.datasets == "all" \
            else [n.strip() for n in args.datasets.split(",")]
        for name in names:
            if name not in DATASETS:
                parser.error(f"unknown dataset '{name}' (known: {', '.join(DATASETS)})")
        symbols = load_universe() if args.symbols == "universe" \
            else [s.strip().upper() for s in args.symbols.split(",")]
        work = [(name, sym) for name in names for sym in symbols]

    # Plan: listing API minus manifest.
    plan = []
    for ds_name, symbol in work:
        items = plan_symbol(client, DATASETS[ds_name], symbol, args.start, end, done)
        if items:
            print(f"plan: {ds_name:20s} {symbol:16s} {len(items)} files")
        plan.extend(items)

    print(f"\ntotal: {len(plan)} files to download ({len(done)} already in manifest)")
    if args.dry_run or not plan:
        return 0

    runner = BackfillRunner(client, manifest, raw_dir=data_dir / "raw",
                            normalized_dir=data_dir / "normalized",
                            workers=args.workers)
    stats = runner.run(plan)

    print(f"\ndone: ok={stats.ok} failed={stats.failed} rows={stats.rows:,}")
    if stats.errors:
        print("errors (first 10):")
        for err in stats.errors[:10]:
            print("  " + err)
    print("\nmanifest summary:")
    print(manifest.summary())
    manifest.close()
    return 1 if stats.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
