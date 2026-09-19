"""Stage 3 -- persistence. Per docs/FEATURE_EVALUATION.md §5.

Target-free. Costs nothing against the multiple-testing budget.

THE QUESTION
------------
Does the ordering a feature produces survive to a period that shares NO data
with the one it was measured on? Snapshots are taken 270 bars (90 days) apart
-- the longest estimation window declared by any family -- so consecutive
snapshots are guaranteed disjoint in their inputs.

WHY THIS STAGE EXISTS
---------------------
It has already earned its place once. The correlation term structure beat
every other criterion in its group -- only +0.295 redundant with existing
columns, apparently almost pure new information -- and scored +0.038 here. It
was noise.

**Low redundancy has two explanations, new information and noise, and noise
correlates with nothing.** No redundancy check can separate them, which is why
Stage 3 must run BEFORE Stage 4: clustering would otherwise happily keep a
noise feature as its own cluster representative, precisely because nothing
else looks like it.

WHO IS GATED
------------
Only windowed estimators, identified the same way as in Stage 2 (lag-1
autocorrelation). A fast feature SHOULD reorder between disjoint periods:
`resid_reversal_8h` measures a transient state, not a coin characteristic, and
persistence near zero is the correct behaviour rather than a defect. Fast
features are reported for information and not gated.

Context (bar-level) columns have no cross-sectional ordering at all, so the
question is undefined for them; they are exempt.

Reference points from measurements already taken: corr_8h(90d) scored +0.411
(real), the correlation term structure +0.038 (noise).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft.paths import DATA_DIR
from mft.featdefs.carry import Carry
from mft.featdefs.positioning import Positioning
from mft.featdefs.liquidity import Liquidity
from mft.featdefs.path import PathAttention
from mft.featdefs.regime import ReturnRegime
from mft.featdefs.relational import Relational
from mft.featdefs.spotperp import SpotPerp
from mft.featdefs.inherited import Inherited
from mft.featdefs.inherited2 import Inherited2

_FAMS = (Carry, Positioning, Liquidity, PathAttention, ReturnRegime,
         Relational, SpotPerp, Inherited, Inherited2)
CONTEXT, CHANGE = set(), set()
for _f in _FAMS:
    CONTEXT |= set(getattr(_f, "context_columns", ()))
    CHANGE |= set(getattr(_f, "change_columns", ()))

SNAPSHOT_BARS = 270          # 90 days: the longest declared estimation window
PERSISTENCE_MIN = 0.15       # pre-registered §5
AUTOCORR_WINDOWED = 0.90     # same scoping rule as Stage 2
MIN_COINS = 8

FAMILIES = ["family_A_carry", "family_B_positioning", "family_C_liquidity",
            "family_D_path", "family_E_regime", "family_F_relational",
            "family_G_spotperp", "family_H_inherited",
            "family_I_inherited2"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features/stage3_persistence.json")
    args = ap.parse_args()

    parts = []
    for f in FAMILIES:
        p = DATA_DIR / f"features/{f}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p).set_index(["t_obs", "symbol"]))
    X = pd.concat(parts, axis=1)
    feats = list(X.columns)
    print(f"Stage 3 -- persistence   {len(feats)} features, {len(X):,} rows")
    print(f"snapshots every {SNAPSHOT_BARS} bars (90d) -> disjoint inputs\n")

    prior = {}
    s2 = DATA_DIR / "features/stage2_estimation.json"
    if s2.exists():
        prior = json.loads(s2.read_text())["results"]

    rows = []
    for c in feats:
        W = X[c].unstack("symbol")
        snaps = W.iloc[::SNAPSHOT_BARS]
        vals = []
        for i in range(len(snaps) - 1):
            a, b = snaps.iloc[i], snaps.iloc[i + 1]
            m = a.notna() & b.notna()
            if m.sum() >= MIN_COINS:
                vals.append(a[m].rank().corr(b[m].rank()))
        vals = [v for v in vals if np.isfinite(v)]
        mean = float(np.mean(vals)) if vals else np.nan
        se = float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
        ac = prior.get(c, {}).get("autocorr", W.apply(lambda s: s.autocorr(1)).median())
        rows.append({"feature": c, "persistence": mean, "se": se,
                     "n_pairs": len(vals), "autocorr": float(ac)})

    R = pd.DataFrame(rows).set_index("feature")
    R["context"] = [c in CONTEXT for c in R.index]
    R["change"] = [c in CHANGE for c in R.index]
    # Persistence asks whether a feature is a stable coin CHARACTERISTIC. A
    # feature built as a difference, a ratio to a lagged value, or a deviation
    # from its own history is transient by construction -- failing would be
    # arithmetic, not evidence. Classified from construction, never from score.
    R["gated"] = (R["autocorr"] > AUTOCORR_WINDOWED) & ~R["context"] & ~R["change"]

    ctxr, gated = R[R["context"]], R[R["gated"]]
    chg = R[R["change"] & ~R["context"]]
    fast = R[~R["gated"] & ~R["context"] & ~R["change"]]
    if len(chg):
        print(f"CHANGE TIER -- exempt ({len(chg)}): transient by construction")
        for c in chg.sort_values("persistence", ascending=False).index[:6]:
            print(f"    {c:<32} {chg.loc[c,'persistence']:+.3f}")
        print(f"    ... and {max(0, len(chg) - 6)} more\n")

    if len(ctxr):
        print("CONTEXT TIER -- exempt (no cross-sectional ordering to persist)")
        for c in ctxr.index:
            print(f"    {c}")
        print()

    print(f"GATED -- windowed estimators ({len(gated)}), bar {PERSISTENCE_MIN}")
    print("=" * 74)
    failed = []
    for c in gated.sort_values("persistence").index:
        v, se = gated.loc[c, "persistence"], gated.loc[c, "se"]
        ok = v > PERSISTENCE_MIN
        if not ok:
            failed.append(c)
        print(f"  [{'PASS' if ok else 'FAIL'}] {c:<32} {v:+.3f} ± {se:.3f}")

    print(f"\nREPORTED ONLY -- fast features ({len(fast)}); low persistence is")
    print("EXPECTED here, since they measure a transient state, not a coin")
    print("characteristic\n" + "=" * 74)
    for c in fast.sort_values("persistence", ascending=False).index:
        print(f"  {c:<32} {fast.loc[c,'persistence']:+.3f} "
              f"(autocorr {fast.loc[c,'autocorr']:+.3f})")

    print("\n" + "=" * 74)
    if failed:
        print(f"FAILED ({len(failed)}): " + ", ".join(failed))
        print("Stage 3 REPORTS; removal is recorded alongside Stage 4 evidence.")
    else:
        print("Every windowed estimator clears the persistence bar.")

    out = DATA_DIR / args.out
    out.write_text(json.dumps({
        "stage": 3, "snapshot_bars": SNAPSHOT_BARS,
        "persistence_min": PERSISTENCE_MIN,
        "reference_points": {"corr_btc_8h_90d_measured_earlier": 0.411,
                             "correlation_term_structure_rejected": 0.038},
        "results": {c: {"persistence": None if pd.isna(R.loc[c, "persistence"])
                        else float(R.loc[c, "persistence"]),
                        "se": None if pd.isna(R.loc[c, "se"]) else float(R.loc[c, "se"]),
                        "n_pairs": int(R.loc[c, "n_pairs"]),
                        "gated": bool(R.loc[c, "gated"]),
                        "context": bool(R.loc[c, "context"])} for c in R.index},
        "failed": failed,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
