"""Write a train-only copy of the panel, so research CANNOT read the holdout.

The split in mft/splits.py is a rule. This makes it a fact: everything under
DATA_DIR/train/ ends at the purged train boundary, so a study pointed at that
directory has no way to touch 2025 or 2026 -- not by a forgotten filter, not by
a default argument, not by a bug.

Purging is applied at write time using the FULL 120h horizon. That is
deliberate: the label columns of a bar dated 2024-12-30 are computed from
January 2025 prices, so copying that row would copy holdout information into
the train set no matter how the file is later filtered.

Files are streamed batch-by-batch; the 2GB panel is never fully resident.

    python scripts/materialize_train.py            # write DATA_DIR/train/
    python scripts/materialize_train.py --force    # overwrite an existing copy
    python scripts/materialize_train.py --verify   # re-check an existing copy

Some panel files carry no timestamp, only `bar_id`. bar_id is a global,
time-ordered key (one close time each), so those are split through a
bar_id -> bar_close_time map built from the feature panel.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft import splits
from mft.paths import DATA_DIR

BATCH_ROWS = 100_000

# (relative source path, time key kind, key name)
#   "time"   -- column holds epoch ms directly
#   "bar_id" -- column holds a global bar index, resolved via the bar_id map
#   "index"  -- pandas index holds epoch ms (small files, read whole)
SOURCES = [
    ("panel/panel-200M-features.parquet", "time", "bar_close_time"),
    ("panel/relative-200M.parquet", "bar_id", "bar_id"),
    ("panel/beta_catchup-200M.parquet", "bar_id", "bar_id"),
    ("panel/market_index.parquet", "time", "grid_ms"),
    ("panel/market_weights.parquet", "index", None),
    ("labels/panel-200M-labels-k1.0.parquet", "time", "bar_close_time"),
]

TRAIN_ROOT = DATA_DIR / "train"


def fmt(ms) -> str:
    return splits._fmt(int(ms))


def build_bar_id_map(cutoff_ms: int) -> tuple[int, int]:
    """Return (max train bar_id, n train bar_ids), asserting bar_id is ordered."""
    src = DATA_DIR / "panel" / "panel-200M-features.parquet"
    if not src.exists():
        raise SystemExit(
            f"No panel to materialize: {src} does not exist.\n"
            "Expected on a fresh MFT_zero data directory -- the source data was\n"
            "carried over but every derived product was left behind on purpose.\n"
            "Rebuild bars -> features -> labels first, then run this."
        )
    p = pq.read_table(src, columns=["bar_id", "bar_close_time"]).to_pandas()
    m = p.drop_duplicates("bar_id").set_index("bar_id")["bar_close_time"].sort_index()
    # bar_id must increase with time for a scalar cutoff to be valid.
    if not m.is_monotonic_increasing:
        raise SystemExit(
            "bar_id is not time-ordered; a scalar bar_id cutoff would be wrong. "
            "Split those files on an explicit bar_id set instead."
        )
    train_ids = m.index[m < cutoff_ms]
    return int(train_ids.max()), len(train_ids)


def copy_filtered(src: Path, dst: Path, kind: str, key: str | None,
                  cutoff_ms: int, max_bar_id: int) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if kind == "index":
        df = pd.read_parquet(src)
        kept = df.loc[df.index < cutoff_ms]
        kept.to_parquet(dst)
        return len(df), len(kept)

    limit = cutoff_ms if kind == "time" else max_bar_id
    op = (lambda col: col < limit) if kind == "time" else (lambda col: col <= limit)

    pf = pq.ParquetFile(src)
    if key not in pf.schema_arrow.names:
        raise SystemExit(f"{src.name}: expected key column {key!r}, has {pf.schema_arrow.names[:6]}")

    writer = None
    n_in = n_out = 0
    try:
        for batch in pf.iter_batches(batch_size=BATCH_ROWS):
            n_in += batch.num_rows
            col = batch.column(batch.schema.get_field_index(key)).to_pandas()
            keep = op(col).to_numpy()
            if not keep.any():
                continue
            filtered = batch.filter(pa.array(keep))
            if writer is None:
                writer = pq.ParquetWriter(dst, filtered.schema, compression="snappy")
            writer.write_batch(filtered)
            n_out += filtered.num_rows
    finally:
        if writer is not None:
            writer.close()
    return n_in, n_out


def verify(cutoff_ms: int, max_bar_id: int) -> int:
    print("\nVERIFY -- nothing in the train copy crosses the wall\n" + "=" * 74)
    failures = 0
    for rel, kind, key in SOURCES:
        dst = TRAIN_ROOT / rel
        if not dst.exists():
            print(f"  [FAIL] {rel}: missing")
            failures += 1
            continue
        if kind == "index":
            idx = pd.read_parquet(dst).index
            worst, ok = (int(idx.max()) if len(idx) else 0), (idx < cutoff_ms).all()
            detail = fmt(worst)
        elif kind == "time":
            col = pq.read_table(dst, columns=[key]).to_pandas()[key]
            worst, ok = int(col.max()), bool((col < cutoff_ms).all())
            detail = fmt(worst)
        else:
            col = pq.read_table(dst, columns=[key]).to_pandas()[key]
            worst, ok = int(col.max()), bool((col <= max_bar_id).all())
            detail = f"bar_id {worst} (train max {max_bar_id})"
        print(f"  [{'PASS' if ok else 'FAIL'}] {rel:<44} last {detail}")
        failures += (not ok)

    # The guard research will actually call, run against the materialized panel.
    p = pq.read_table(TRAIN_ROOT / "panel" / "panel-200M-features.parquet",
                      columns=["bar_close_time"]).to_pandas()
    try:
        splits.assert_train_only(p)
        print(f"  [PASS] assert_train_only on the materialized panel ({len(p):,} rows)")
    except splits.HoldoutLeak as e:
        print(f"  [FAIL] assert_train_only: {str(e)[:90]}")
        failures += 1
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite an existing train copy")
    ap.add_argument("--verify", action="store_true", help="only re-check an existing copy")
    args = ap.parse_args()

    cutoff = splits.get(splits.TRAIN).usable_end_ms()  # 120h purge
    print(f"Train copy -> {TRAIN_ROOT}")
    print(f"Cutoff: bar_close_time < {fmt(cutoff)}  "
          f"({splits.MAX_LABEL_HORIZON_H}h purge before {fmt(splits.HOLDOUT1_START_MS)})")

    max_bar_id, n_ids = build_bar_id_map(cutoff)
    print(f"bar_id cutoff: <= {max_bar_id:,}  ({n_ids:,} train bars)\n")

    if args.verify:
        return 1 if verify(cutoff, max_bar_id) else 0

    if TRAIN_ROOT.exists():
        if not args.force:
            raise SystemExit(f"{TRAIN_ROOT} already exists; pass --force to overwrite")
        shutil.rmtree(TRAIN_ROOT)

    print(f"  {'file':<44} {'rows in':>11} {'rows out':>11} {'size':>9}")
    print("  " + "-" * 78)
    for rel, kind, key in SOURCES:
        src = DATA_DIR / rel
        if not src.exists():
            print(f"  {rel:<44} MISSING -- skipped")
            continue
        dst = TRAIN_ROOT / rel
        n_in, n_out = copy_filtered(src, dst, kind, key, cutoff, max_bar_id)
        mb = dst.stat().st_size / 1e6
        print(f"  {rel:<44} {n_in:>11,} {n_out:>11,} {mb:>8.0f}M")

    failures = verify(cutoff, max_bar_id)
    print("\n" + "=" * 74)
    if failures:
        print(f"FAILED: {failures} check(s). Do not use this copy.")
        return 1
    total = sum(f.stat().st_size for f in TRAIN_ROOT.rglob("*.parquet")) / 1e9
    print(f"Train copy is sealed at {fmt(cutoff)} ({total:.2f} GB).")
    print("Point research at DATA_DIR/train; holdout1 and holdout2 are not reachable from it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
