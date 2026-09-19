"""The freeze point. Per docs/FEATURE_EVALUATION.md §8.

Takes the consolidated removal decision across Stages 2-5 evidence and writes
the frozen candidate list. After this runs, nothing may be re-run without being
counted against the multiple-testing budget.

THE DECISION RULES, ALL PRE-REGISTERED BEFORE ANY EVIDENCE WAS SEEN
-------------------------------------------------------------------
§5  Stage 3, persistence > 0.15, on gated windowed estimators. This is the
    stage that earned its place empirically: the correlation term structure
    looked like the best candidate in its group on every other criterion and
    scored +0.038 here. Low redundancy has two explanations, new information
    and noise, and only persistence separates them. FAILURES ARE REMOVED.

§6  Stage 4, multivariate R^2 < 0.80 against all other accepted features.
    REMOVED, but ITERATIVELY -- see below.

§4  Stage 2 REPORTS and does not delete; §7 Stage 5 is explicitly not a kill
    criterion. Neither removes anything here.

WHY THE MULTIVARIATE PASS MUST BE ITERATIVE
-------------------------------------------
R^2 was computed for each survivor against ALL other survivors at once, so the
flagged set partly explains itself: nine features each scoring >= 0.80 does not
mean nine redundant features, it can mean two or three that mutually predict
one another. Dropping all nine together would delete information that only
looked redundant because its explainers were still in the set.

So: drop the single worst, RECOMPUTE against what remains, repeat until nothing
is above the bar. That is the only reading under which the pre-registered
threshold means what it says.

The benchmark is exempt by the same rule Stage 4 used -- FEATURE_LIST_FROZEN.md
§6 decision rule 1 makes `resid_reversal_8h` the reference point for the whole
evaluation, and a benchmark the pipeline deleted cannot serve as one.

NOTHING HERE TOUCHES THE TARGET.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft.paths import DATA_DIR

BENCHMARK = "resid_reversal_8h"
MULTIVARIATE_R2_MAX = 0.80        # pre-registered §6

FAMILIES = {
    "A": "family_A_carry", "B": "family_B_positioning", "C": "family_C_liquidity",
    "D": "family_D_path", "E": "family_E_regime", "F": "family_F_relational",
    "G": "family_G_spotperp", "H": "family_H_inherited",
    "I": "family_I_inherited2",
}


def multivariate_r2(S: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """R^2 of each column regressed on all the others, in within-bar rank space."""
    out = {}
    M = S[cols].to_numpy()
    for i, c in enumerate(cols):
        y = M[:, i]
        X = np.delete(M, i, axis=1)
        X = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        out[c] = float(1.0 - resid.var() / y.var()) if y.var() > 0 else np.nan
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features/frozen_list.json")
    args = ap.parse_args()

    d = DATA_DIR / "features"
    s2 = json.loads((d / "stage2_estimation.json").read_text())
    s3 = json.loads((d / "stage3_persistence.json").read_text())
    s4 = json.loads((d / "stage4_redundancy.json").read_text())
    s5 = json.loads((d / "stage5_turnover.json").read_text())

    owner, parts = {}, []
    for code, fname in FAMILIES.items():
        p = d / f"{fname}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p).set_index(["t_obs", "symbol"])
        for c in df.columns:
            owner[c] = code
        parts.append(df)
    X = pd.concat(parts, axis=1)

    survivors = [c for c in X.columns if c not in s4["absorbed"]]
    print(f"FREEZE POINT -- starting from {len(survivors)} Stage 4 survivors\n")

    removed: dict[str, str] = {}

    # ---- rule 1: Stage 3 persistence ------------------------------------
    s3fail = [c for c in s3["failed"] if c in survivors]
    print(f"RULE 1 -- Stage 3 persistence > 0.15 (pre-registered §5)")
    print("=" * 74)
    for c in s3fail:
        v = s3["results"][c]["persistence"]
        print(f"  REMOVE  {c:<32} [{owner[c]}]  persistence {v:+.3f}")
        removed[c] = f"stage3_persistence={v:.3f}"
    keep = [c for c in survivors if c not in removed]
    print(f"  -> {len(keep)} remain\n")

    # ---- rule 2: multivariate, iteratively -------------------------------
    print(f"RULE 2 -- Stage 4 multivariate R² < {MULTIVARIATE_R2_MAX} "
          f"(pre-registered §6), applied ONE AT A TIME")
    print("=" * 74)
    Rk = X[keep].groupby(level="t_obs").rank(pct=True)
    Rk = Rk - Rk.groupby(level="t_obs").transform("mean")
    S = Rk.dropna()
    print(f"  {len(S):,} complete rows across {len(keep)} features")

    ctx = set()
    for c in keep:
        if s5["results"].get(c, {}).get("context"):
            ctx.add(c)

    while True:
        cols = [c for c in keep if c in S.columns]
        r2 = multivariate_r2(S, cols)
        cand = {c: v for c, v in r2.items()
                if pd.notna(v) and v >= MULTIVARIATE_R2_MAX and c != BENCHMARK}
        if not cand:
            break
        worst = max(cand, key=cand.get)
        print(f"  REMOVE  {worst:<32} [{owner[worst]}]  R² {cand[worst]:.3f}"
              f"   ({len(cand)} over the bar at this step)")
        removed[worst] = f"multivariate_r2={cand[worst]:.3f}"
        keep = [c for c in keep if c != worst]
        S = S.drop(columns=[worst])

    print(f"  -> {len(keep)} remain")
    if BENCHMARK in keep:
        print(f"  [rule] {BENCHMARK} exempt: it is the evaluation's reference point")
    print()

    # ---- retained despite a flag, recorded --------------------------------
    print("RETAINED DESPITE A FLAG (recorded, not silent)")
    print("=" * 74)
    for c in s2.get("flagged", []):
        if c in keep:
            print(f"  KEEP    {c:<32} [{owner[c]}]  Stage 2 flag "
                  f"{s2['results'][c]['rank_stability']:.3f} vs 0.90 bar")
            print(f"          §4 states Stage 2 reports and does not delete; it "
                  f"passes Stage 3 at {s3['results'][c]['persistence']:+.3f}")
    print()

    # ---- the frozen list --------------------------------------------------
    keep = sorted(keep)
    by_fam = pd.Series([owner[c] for c in keep]).value_counts().sort_index()
    print("=" * 74)
    print(f"FROZEN: {len(keep)} features  ({len(survivors) - len(keep)} removed "
          f"by the consolidated decision)")
    print("  by family: " + ", ".join(f"{k}={v}" for k, v in by_fam.items()))

    manifest = {
        "frozen": True,
        "n_features": len(keep),
        "features": {c: {
            "family": owner[c],
            "context": bool(s5["results"].get(c, {}).get("context", False)),
            "persistence": s3["results"].get(c, {}).get("persistence"),
            "autocorr": s5["results"].get(c, {}).get("autocorr"),
            "turnover": s5["results"].get(c, {}).get("turnover"),
        } for c in keep},
        "removed": removed,
        "rules": {
            "stage3_persistence_min": 0.15,
            "stage4_multivariate_r2_max": MULTIVARIATE_R2_MAX,
            "multivariate_applied": "iteratively, recomputed after each removal",
            "benchmark_exempt": BENCHMARK,
            "stage2_deletes": False,
            "stage5_deletes": False,
        },
    }
    blob = json.dumps(manifest["features"], sort_keys=True).encode()
    manifest["content_hash"] = hashlib.sha256(blob).hexdigest()[:16]
    print(f"  content hash: {manifest['content_hash']}")

    out = DATA_DIR / args.out
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    print("The list is FROZEN. Anything re-run past this point is counted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
