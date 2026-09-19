"""Stage 6 -- univariate screen. Per docs/FEATURE_SELECTION.md §3.

THE FIRST STAGE THAT TOUCHES THE TARGET. Everything before this was free;
this is counted.

LENIENT BY PRE-REGISTRATION, NOT BY TIMIDITY
--------------------------------------------
FEATURE_EVALUATION.md §8: "remove features that are provably nothing, not pick
winners." Selecting winners univariately does two bad things -- it overfits,
and it deletes features that only work in combination. `crowd_winning` is
EXPECTED to be flat alone and to matter interacted; a selective screen would
delete it for being exactly what it was designed to be.

So the removal rule has two clauses and BOTH must hold:

    |IC| is small          the effect, if any, is negligible
    AND the SE is small    the sample could have detected an effect this size

The second clause is what makes it lenient. A wide error bar means "not
measured", which is not the same as "nothing", and a feature with one is kept.
Deleting on a small |IC| alone would silently delete the underpowered.

THE METRIC
----------
Per-bar Spearman IC between the feature and the target, over coins present in
both. The distribution of those per-bar ICs gives the point estimate and its
standard error directly.

Consecutive bars are NON-OVERLAPPING by construction -- bar t's label spans
[t+1h, t+9h] and bar t+1's spans [t+9h, t+17h] -- so the per-bar ICs are
serially independent and the t-statistic needs no Newey-West adjustment. That
is a property of the target design, verified in FEATURE_SELECTION.md §1.

COMPUTED PER FOLD, THEN AGGREGATED
----------------------------------
Per docs/FEATURE_SELECTION.md §2, no statistic is computed once on all of
train. Each fold's TEST block yields an independent IC; the five are reported
side by side, and a feature is removed only on the pooled evidence. The folds
also make the sign-consistency column meaningful.

CONTEXT COLUMNS ARE EXEMPT. They have no cross-sectional ordering, so a
cross-sectional IC is undefined for them. They enter the model only as
interactions and cannot be screened here.

NOTHING IS REMOVED BY THIS SCRIPT. It reports; removal is a recorded decision
like every other stage in this project.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft import folds as foldmod
from mft import splits
from mft.paths import DATA_DIR

MIN_COINS = 8
IC_NEGLIGIBLE = 0.005     # pre-registered: |IC| below this is negligible
SE_RESOLVED = 0.005       # pre-registered: SE below this means it was measured

FAMILIES = {
    "A": "family_A_carry", "B": "family_B_positioning", "C": "family_C_liquidity",
    "D": "family_D_path", "E": "family_E_regime", "F": "family_F_relational",
    "G": "family_G_spotperp", "H": "family_H_inherited",
    "I": "family_I_inherited2",
}


def per_bar_ic(f: pd.Series, y: pd.Series) -> pd.Series:
    """Spearman IC per bar, on coins where both exist."""
    d = pd.DataFrame({"f": f, "y": y}).dropna()
    g = d.groupby(level="t_obs")
    n = g.size()
    d = d[d.index.get_level_values("t_obs").isin(n[n >= MIN_COINS].index)]
    if not len(d):
        return pd.Series(dtype="float64")
    r = d.groupby(level="t_obs").rank()
    r = r - r.groupby(level="t_obs").transform("mean")
    num = (r["f"] * r["y"]).groupby(level="t_obs").sum()
    den = np.sqrt((r["f"] ** 2).groupby(level="t_obs").sum()
                  * (r["y"] ** 2).groupby(level="t_obs").sum())
    return (num / den.replace(0.0, np.nan)).dropna()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features/stage6_univariate.json")
    args = ap.parse_args()

    fz = json.loads((DATA_DIR / "features/frozen_list.json").read_text())
    cols = list(fz["features"])
    ctx = {c for c, m in fz["features"].items() if m.get("context")}

    parts = []
    for fname in FAMILIES.values():
        p = DATA_DIR / f"features/{fname}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p).set_index(["t_obs", "symbol"]))
    X = pd.concat(parts, axis=1)[cols]

    t = pd.read_parquet(DATA_DIR / "grid/target_8h_lag60.parquet")
    t = t[t["t_obs"] < splits.HOLDOUT1_START_MS]
    splits.assert_train_only(t.assign(bar_close_time=t["t_obs"]))
    Y = t.set_index(["t_obs", "symbol"])["target"]

    X = X[X.index.get_level_values("t_obs") < splits.HOLDOUT1_START_MS]
    idx = X.index.intersection(Y.index)
    X, Y = X.loc[idx], Y.loc[idx]

    t_all = np.sort(X.index.get_level_values("t_obs").unique())
    F = foldmod.make_folds(t_all)

    print("STAGE 6 -- univariate screen   *** FIRST TARGET CONTACT ***")
    print(f"  frozen list {fz['n_features']} features, hash {fz['content_hash']}")
    print(f"  {len(X):,} rows, {len(t_all):,} train bars, "
          f"{len(ctx)} context columns exempt")
    print("\n" + foldmod.describe(F) + "\n")
    print("Removal needs BOTH |IC| < %.3f AND SE < %.3f -- a wide error bar"
          % (IC_NEGLIGIBLE, SE_RESOLVED))
    print("means NOT MEASURED, which is not the same as nothing.\n")

    tested = [c for c in cols if c not in ctx]
    rows = []
    for c in tested:
        ics, per_fold = [], []
        for f in F:
            s = per_bar_ic(X[c].loc[X.index.get_level_values("t_obs").isin(f.test)],
                           Y.loc[Y.index.get_level_values("t_obs").isin(f.test)])
            per_fold.append(float(s.mean()) if len(s) else np.nan)
            ics.append(s)
        allic = pd.concat(ics) if ics else pd.Series(dtype="float64")
        mean = float(allic.mean()) if len(allic) else np.nan
        se = float(allic.std(ddof=1) / np.sqrt(len(allic))) if len(allic) > 1 else np.nan
        pf = np.array(per_fold, dtype="float64")
        same = int(np.nansum(np.sign(pf) == np.sign(mean))) if np.isfinite(mean) else 0
        rows.append({"feature": c, "ic": mean, "se": se,
                     "t": mean / se if se and np.isfinite(se) and se > 0 else np.nan,
                     "n_bars": int(len(allic)), "folds_same_sign": same,
                     "per_fold": per_fold})

    R = pd.DataFrame(rows).set_index("feature")
    R["negligible"] = R["ic"].abs() < IC_NEGLIGIBLE
    R["resolved"] = R["se"] < SE_RESOLVED
    R["remove"] = R["negligible"] & R["resolved"]

    S = R.sort_values("ic", key=lambda s: s.abs(), ascending=False)
    print(f"{'feature':<34} {'IC':>8} {'SE':>7} {'t':>7} {'folds':>6}")
    print("=" * 74)
    for c in S.index:
        r = S.loc[c]
        mark = "  REMOVE" if r["remove"] else ("" if abs(r["t"]) < 2 else "  *")
        print(f"{c:<34} {r['ic']:>+8.4f} {r['se']:>7.4f} {r['t']:>+7.2f} "
              f"{r['folds_same_sign']:>4}/5{mark}")

    rem = list(R.index[R["remove"]])
    kept_unresolved = list(R.index[R["negligible"] & ~R["resolved"]])

    print("\n" + "=" * 74)
    print(f"IC magnitude: median |IC| {R['ic'].abs().median():.4f}, "
          f"max {R['ic'].abs().max():.4f} ({R['ic'].abs().idxmax()})")
    print(f"|t| >= 2: {int((R['t'].abs() >= 2).sum())} of {len(R)}")
    print(f"5/5 folds same sign: {int((R['folds_same_sign'] == 5).sum())}")
    print(f"\nFLAGGED FOR REMOVAL ({len(rem)}): "
          + (", ".join(rem) if rem else "none"))
    if kept_unresolved:
        print(f"\nKEPT THOUGH |IC| IS SMALL ({len(kept_unresolved)}) -- the SE is "
              f"too wide\nto call it measured, and 'not measured' is not "
              f"'nothing':")
        for c in kept_unresolved:
            print(f"    {c:<32} IC {R.loc[c,'ic']:+.4f}  SE {R.loc[c,'se']:.4f}")
    print("\nStage 6 REPORTS. Removal is a recorded decision.")

    out = DATA_DIR / args.out
    out.write_text(json.dumps({
        "stage": 6, "first_target_contact": True,
        "frozen_hash": fz["content_hash"],
        "ic_negligible": IC_NEGLIGIBLE, "se_resolved": SE_RESOLVED,
        "folds": {"n": foldmod.N_FOLDS, "min_train": foldmod.MIN_TRAIN_BARS,
                  "test_bars": foldmod.TEST_BARS, "purge": foldmod.PURGE_BARS},
        "context_exempt": sorted(ctx),
        "results": {c: {"ic": None if pd.isna(R.loc[c, "ic"]) else float(R.loc[c, "ic"]),
                        "se": None if pd.isna(R.loc[c, "se"]) else float(R.loc[c, "se"]),
                        "t": None if pd.isna(R.loc[c, "t"]) else float(R.loc[c, "t"]),
                        "n_bars": int(R.loc[c, "n_bars"]),
                        "folds_same_sign": int(R.loc[c, "folds_same_sign"]),
                        "per_fold": R.loc[c, "per_fold"]} for c in R.index},
        "flagged_for_removal": rem,
        "kept_unresolved": kept_unresolved,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
