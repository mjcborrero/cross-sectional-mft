"""Prove the train/holdout split is correct against the actual panel data.

Checks, in order of how badly a failure would hurt:

  1. LABEL LEAK   -- no purged row's label resolves at or past the next
                     period's start. Checked against the real forward-return
                     ladder AND the triple barrier's recorded end time, not
                     against the nominal horizon.
  2. PARTITION    -- the three splits are disjoint and, before purging, cover
                     every row exactly once.
  3. PURGE COST   -- how many rows the purge removes, so the cost is on the
                     record rather than silent.
  4. CENSUS       -- rows, symbols, and date range per split.

Exit code is non-zero if any check fails.

    python scripts/verify_split.py
    python scripts/verify_split.py --horizon-hours 24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft import splits
from mft.paths import DATA_DIR

PANEL = DATA_DIR / "panel" / "panel-200M-features.parquet"
LABELS = DATA_DIR / "labels" / "panel-200M-labels-k1.0.parquet"

FAILURES: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def fmt(ms) -> str:
    return splits._fmt(int(ms))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon-hours", type=int, default=splits.MAX_LABEL_HORIZON_H,
                    help="purge horizon (default: the 120h ladder max)")
    args = ap.parse_args()
    H = args.horizon_hours

    missing = [p for p in (PANEL, LABELS) if not p.exists()]
    if missing:
        print("Nothing to verify yet -- the panel has not been rebuilt.\n")
        for p in missing:
            print(f"  missing: {p}")
        print("\nThis is expected on a fresh MFT_zero data directory: the source\n"
              "data was carried over but every derived feature/label product was\n"
              "deliberately left behind. Rebuild bars -> features -> labels, then\n"
              "run this again. The split itself is already defined and unit-tested\n"
              "(tests/test_splits.py); this script checks it against real data.")
        return 0

    print(f"Loading panel + labels\n" + "=" * 72)
    panel = pd.read_parquet(PANEL, columns=["symbol", "bar_close_time"])
    lab = pd.read_parquet(
        LABELS,
        columns=["symbol", "bar_close_time", "rel_tb_label_end_time", "rel_target_24h"],
    )
    data_end = int(panel.bar_close_time.max())

    print(f"\nSplit definition (purge = {H}h, data ends {fmt(data_end)})\n" + "=" * 72)
    for name in splits.SPLIT_NAMES:
        sp = splits.get(name)
        print(f"  {sp.label()}")
        print(f"      usable through {fmt(sp.usable_end_ms(H, data_end))}  (label-purged)")
    print()
    print(f"  panel  {len(panel):>9,} rows   {fmt(panel.bar_close_time.min())} -> {fmt(panel.bar_close_time.max())}")
    print(f"  labels {len(lab):>9,} rows   {fmt(lab.bar_close_time.min())} -> {fmt(lab.bar_close_time.max())}")

    # ---- 1. label leak -------------------------------------------------
    print(f"\n1. LABEL LEAK -- does any purged row's outcome resolve in the next period?\n" + "=" * 72)
    for name in splits.SPLIT_NAMES:
        sp = splits.get(name)
        rows = splits.take(lab, name, horizon_hours=H)
        if rows.empty:
            check(False, f"{name}: non-empty", "no rows selected")
            continue

        # For a dated split the bound is the next period's start; for the
        # open-ended one it is the end of the data (the label must have
        # actually happened).
        bound = data_end if sp.is_open_ended else sp.end_ms

        # Ladder: the longest forward label resolves H hours after the bar.
        ladder_end = rows.bar_close_time.max() + H * 3_600_000
        check(
            ladder_end <= bound,
            f"{name}: {H}h ladder label resolves by {fmt(bound)}",
            f"last resolves {fmt(ladder_end)}",
        )

        # Triple barrier: use the recorded end time, not an assumed horizon.
        tb = rows["rel_tb_label_end_time"].dropna()
        if len(tb):
            check(
                tb.max() <= bound,
                f"{name}: triple-barrier label resolves by {fmt(bound)}",
                f"last resolves {fmt(tb.max())}",
            )

    # ---- 2. partition --------------------------------------------------
    print("\n2. PARTITION -- disjoint, and exhaustive before purging\n" + "=" * 72)
    assigned = splits.assign(panel.bar_close_time)
    counts = assigned.value_counts()
    check(
        int(counts.sum()) == len(panel),
        "every row assigned to exactly one split",
        f"{int(counts.sum()):,} / {len(panel):,}",
    )
    idx = {n: set(splits.take(panel, n, horizon_hours=H).index) for n in splits.SPLIT_NAMES}
    for a, b in (("train", "holdout1"), ("train", "holdout2"), ("holdout1", "holdout2")):
        check(not (idx[a] & idx[b]), f"{a} n {b} = empty", f"{len(idx[a] & idx[b])} shared")

    # The guard research code will actually call.
    try:
        splits.assert_train_only(splits.take(panel, "train", horizon_hours=H))
        check(True, "assert_train_only accepts the train slice")
    except splits.HoldoutLeak as e:
        check(False, "assert_train_only accepts the train slice", str(e)[:80])
    try:
        splits.assert_train_only(panel)
        check(False, "assert_train_only rejects the full panel", "it did not raise")
    except splits.HoldoutLeak:
        check(True, "assert_train_only rejects the full panel")

    # ---- 3. purge cost -------------------------------------------------
    print("\n3. PURGE COST -- rows dropped at each boundary\n" + "=" * 72)
    for name in splits.SPLIT_NAMES:
        raw = int(splits.mask(panel.bar_close_time, name, purge=False).sum())
        kept = int(splits.mask(panel.bar_close_time, name, horizon_hours=H).sum())
        pct = 100.0 * (raw - kept) / raw if raw else 0.0
        print(f"  {name:<9} {raw:>8,} -> {kept:>8,}   dropped {raw - kept:>6,} ({pct:.2f}%)")

    # ---- 4. census -----------------------------------------------------
    print("\n4. CENSUS -- what each split actually contains\n" + "=" * 72)
    print(f"  {'split':<9} {'rows':>9} {'symbols':>8}  {'first bar':<17} {'last bar':<17}")
    for name in splits.SPLIT_NAMES:
        s = splits.take(panel, name, horizon_hours=H)
        print(f"  {name:<9} {len(s):>9,} {s.symbol.nunique():>8}  "
              f"{fmt(s.bar_close_time.min()):<17} {fmt(s.bar_close_time.max()):<17}")

    # The real trainable dataset is panel INNER JOIN labels: the labels file
    # carries rows the panel does not (each symbol's pre-warmup history, before
    # its 90d feature lookbacks are satisfied), so counting labels alone
    # overstates what research can actually use.
    print("\n  trainable rows (panel JOIN labels, with a usable 24h target):")
    joined = panel.merge(
        lab[["symbol", "bar_close_time", "rel_target_24h"]],
        on=["symbol", "bar_close_time"], how="inner", validate="one_to_one",
    )
    check(
        len(joined) <= len(panel),
        "join does not duplicate panel rows",
        f"{len(joined):,} joined vs {len(panel):,} panel",
    )
    for name in splits.SPLIT_NAMES:
        s = splits.take(joined, name, horizon_hours=H, data_end_ms=data_end)
        ok = int(s["rel_target_24h"].notna().sum())
        pct = 100.0 * ok / len(s) if len(s) else 0.0
        print(f"    {name:<9} {ok:>9,} / {len(s):>9,}  ({pct:.1f}%)")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print("All checks passed. Research runs on train; holdout1/holdout2 stay sealed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
