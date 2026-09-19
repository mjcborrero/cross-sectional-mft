"""Stage 4 -- redundancy at three levels. Per docs/FEATURE_EVALUATION.md §6.

Target-free. This is the stage that does the volume reduction.

THE RIGHT METRIC
----------------
Redundancy here means "produces the same ORDERING", because that is all a
ranking objective consumes. So features are compared by WITHIN-BAR rank
correlation, not pooled Pearson. Two features can have modest pooled
correlation and still rank every cross-section identically -- which is exactly
what the relative transforms did (rank correlation 1.0000000000 while looking
superficially distinct).

Implementation: convert each feature to within-bar percentile ranks, demean by
bar, then take one pooled correlation. That equals the bar-weighted average of
per-bar rank correlations without looping over 4,558 bars.

THREE LEVELS
------------
PAIRWISE      |rank corr| >= 0.90 marks two features as the same thing.
CLUSTERING    Leader clustering: take the highest-priority feature, absorb
              everything correlated at or above the bar, repeat. Deterministic,
              and the representative falls out of the priority order rather
              than being chosen after the fact.
MULTIVARIATE  R^2 of each survivor regressed on all OTHER survivors, bar 0.80.
              Pairwise misses a feature that is a linear combination of three
              others; this catches it.

REPRESENTATIVE CHOICE IS TARGET-FREE, AND DELIBERATELY SO
---------------------------------------------------------
§6: "Choose the representative by interpretability and cost, not by any
target-related quantity -- selecting the cluster member with the best IC would
be target contact smuggled into a target-free stage."

Priority uses only prior target-free evidence:
    +4  passed Stage 3 persistence (or is validly exempt)
    +2  designed feature with a stated prior (Families A-G) over inherited (H)
    +2  passed Stage 2 estimation quality
    +1  scaled by coverage
Ties break alphabetically, so the outcome does not depend on column order.
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

PAIRWISE_MAX = 0.90       # pre-registered §6

# The benchmark is retained by RULE, not by score. FEATURE_LIST_FROZEN.md §6
# decision rule 1: "The set must beat the benchmark, not zero. The comparison
# is against E1 alone." A benchmark that clustering deleted cannot serve as the
# reference point for the whole evaluation, so E1 leads its cluster whatever
# the priority arithmetic says. This is a pre-registered exception, not a
# preference expressed after seeing the result.
BENCHMARK = "resid_reversal_8h"
MULTIVARIATE_R2_MAX = 0.80

FAMILIES = {
    "A": "family_A_carry", "B": "family_B_positioning", "C": "family_C_liquidity",
    "D": "family_D_path", "E": "family_E_regime", "F": "family_F_relational",
    "G": "family_G_spotperp", "H": "family_H_inherited",
    "I": "family_I_inherited2",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features/stage4_redundancy.json")
    args = ap.parse_args()

    owner, parts = {}, []
    for code, fname in FAMILIES.items():
        p = DATA_DIR / f"features/{fname}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p).set_index(["t_obs", "symbol"])
        for c in d.columns:
            owner[c] = code
        parts.append(d)
    X = pd.concat(parts, axis=1)
    feats = list(X.columns)
    print(f"Stage 4 -- redundancy   {len(feats)} features, {len(X):,} rows\n")

    s2 = json.loads((DATA_DIR / "features/stage2_estimation.json").read_text())
    s3 = json.loads((DATA_DIR / "features/stage3_persistence.json").read_text())

    # ---- within-bar rank space -----------------------------------------
    Rk = X.groupby(level="t_obs").rank(pct=True)
    # Demean within bar so a pooled correlation equals the bar-weighted average
    # of per-bar rank correlations, without looping over every bar.
    Rk = Rk - Rk.groupby(level="t_obs").transform("mean")
    C = Rk.corr(min_periods=500)

    # ---- priority, from target-free evidence only ------------------------
    cov = X.notna().mean()
    s3fail = set(s3.get("failed", []))
    s2fail = set(s2.get("flagged", []))
    prio = {}
    for c in feats:
        p = 0.0
        p += 0.0 if c in s3fail else 4.0
        p += 2.0 if owner[c] not in ("H", "I") else 0.0
        p += 0.0 if c in s2fail else 2.0
        p += float(cov[c])
        if c == BENCHMARK:
            p += 100.0        # see BENCHMARK above
        prio[c] = p

    order = sorted(feats, key=lambda c: (-prio[c], c))

    # ---- leader clustering ----------------------------------------------
    A = C.abs()
    unassigned = set(feats)
    clusters = []
    for c in order:
        if c not in unassigned:
            continue
        members = [c]
        for o in list(unassigned):
            if o == c:
                continue
            v = A.at[c, o]
            if pd.notna(v) and v >= PAIRWISE_MAX:
                members.append(o)
        unassigned -= set(members)
        clusters.append((c, sorted(m for m in members if m != c)))

    reps = [c for c, _ in clusters]
    absorbed = {m: c for c, ms in clusters for m in ms}

    print(f"CLUSTERS  ({len(clusters)} from {len(feats)} features, "
          f"bar |rank corr| >= {PAIRWISE_MAX})\n" + "=" * 74)
    merged = [(c, ms) for c, ms in clusters if ms]
    if merged:
        for c, ms in merged:
            print(f"  {c}  [{owner[c]}]  absorbs:")
            for m in ms:
                print(f"      {m:<34} [{owner[m]}]  rank corr {C.at[c, m]:+.3f}")
    else:
        print("  no pair reaches the bar -- every feature is its own cluster")

    # ---- near-misses, for visibility -------------------------------------
    print(f"\nHIGHEST PAIRS BELOW THE BAR (0.70 .. {PAIRWISE_MAX})\n" + "=" * 74)
    seen, near = set(), []
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            v = A.at[a, b]
            if pd.notna(v) and 0.70 <= v < PAIRWISE_MAX:
                near.append((v, a, b))
    for v, a, b in sorted(near, reverse=True)[:12]:
        print(f"  {C.at[a,b]:+.3f}  {a:<30} ~ {b}")
    if not near:
        print("  none")

    # ---- multivariate: a combination of several others -------------------
    print(f"\nMULTIVARIATE R^2 vs other survivors (bar {MULTIVARIATE_R2_MAX})")
    print("=" * 74)
    S = Rk[reps].dropna()
    flagged = []
    if len(S) > 1000 and len(reps) > 2:
        for c in reps:
            others = [o for o in reps if o != c]
            Ymat = S[others].to_numpy()
            y = S[c].to_numpy()
            Ymat = np.column_stack([np.ones(len(Ymat)), Ymat])
            beta, *_ = np.linalg.lstsq(Ymat, y, rcond=None)
            resid = y - Ymat @ beta
            r2 = 1.0 - resid.var() / y.var() if y.var() > 0 else np.nan
            if pd.notna(r2) and r2 >= MULTIVARIATE_R2_MAX:
                flagged.append((c, float(r2)))
        if flagged:
            for c, r2 in sorted(flagged, key=lambda x: -x[1]):
                print(f"  [FLAG] {c:<32} R2 {r2:.3f}")
        else:
            print("  no survivor is explained by the others above the bar")
    else:
        print("  skipped: insufficient complete rows")

    print("\n" + "=" * 74)
    print(f"SURVIVORS: {len(reps)} of {len(feats)}  "
          f"({len(feats) - len(reps)} absorbed)")
    by_fam = pd.Series([owner[c] for c in reps]).value_counts().sort_index()
    print("  by family: " + ", ".join(f"{k}={v}" for k, v in by_fam.items()))

    out = DATA_DIR / args.out
    out.write_text(json.dumps({
        "stage": 4, "pairwise_max": PAIRWISE_MAX,
        "multivariate_r2_max": MULTIVARIATE_R2_MAX,
        "metric": "within-bar rank correlation",
        "n_in": len(feats), "n_survivors": len(reps),
        "clusters": {c: ms for c, ms in clusters},
        "absorbed": absorbed,
        "multivariate_flagged": dict(flagged),
        "priority": {c: round(prio[c], 3) for c in feats},
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
