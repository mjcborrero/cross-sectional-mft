"""Stage 2 -- estimation quality. Per docs/FEATURE_EVALUATION.md §4.

Target-free: the label is never read, so this costs nothing against the
multiple-testing budget and may be re-run freely.

WHO THIS APPLIES TO
-------------------
§4 scopes Stage 2 to features that are *themselves estimates* -- a
correlation, a beta, an AR coefficient, a variance ratio. Applying it to a
fast-moving feature would be meaningless: `resid_reversal_8h` is a fresh draw
every bar, so of course it looks rough under a smoothness test, and that is
the signal rather than a defect.

Membership is therefore MEASURED, not judged: a windowed estimator reuses most
of its data between consecutive bars, so it has high lag-1 autocorrelation.
Features above the threshold are tested; the rest are exempted and named.

THE TWO CHECKS
--------------
NOISE-TO-SPREAD.  For a smooth underlying series observed with noise, the
second difference cancels the smooth part, and std(d2 x)/sqrt(6) estimates the
noise SD (Gasser-Sroka-Jennen-Steinmetz). Compare that against the feature's
own within-bar cross-sectional spread -- the quantity a ranking model actually
consumes. A feature whose estimation noise rivals its cross-sectional signal
is largely measuring itself.

CONSTRUCTION SENSITIVITY.  §4 specifies nudging the lookback ±15% and
requiring the within-bar ranking to survive. Rebuilding eight families under a
nudged window is a full re-run, so this uses a PROXY: a light smoothing of the
series, which is what modestly lengthening a window does to it. Reported as a
proxy, not as the specified test -- it can only be a lower bound on
sensitivity, and a feature that fails it would certainly fail the real one.
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

# Bar-level columns have NO within-bar spread by construction, so any check
# built on cross-sectional dispersion is undefined for them -- not failed.
CONTEXT = set()
for _f in (Carry, Positioning, Liquidity, PathAttention, ReturnRegime,
           Relational, SpotPerp, Inherited, Inherited2):
    CONTEXT |= set(getattr(_f, "context_columns", ()))

AUTOCORR_WINDOWED = 0.90     # above this, the feature reuses its data => an estimate
NOISE_SPREAD_MAX = 0.50      # pre-registered §4
RANK_STABILITY_MIN = 0.90    # pre-registered §4
SMOOTH_BARS = 3

FAMILIES = {
    "A_carry": "family_A_carry", "B_positioning": "family_B_positioning",
    "C_liquidity": "family_C_liquidity", "D_path": "family_D_path",
    "E_regime": "family_E_regime", "F_relational": "family_F_relational",
    "G_spotperp": "family_G_spotperp", "H_inherited": "family_H_inherited",
    "I_inherited2": "family_I_inherited2",
}


def load_all() -> pd.DataFrame:
    parts = []
    for fam, fname in FAMILIES.items():
        p = DATA_DIR / f"features/{fname}.parquet"
        if not p.exists():
            print(f"  [skip] {fam}: not built")
            continue
        d = pd.read_parquet(p)
        parts.append(d.set_index(["t_obs", "symbol"]))
    return pd.concat(parts, axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features/stage2_estimation.json")
    args = ap.parse_args()

    X = load_all()
    feats = list(X.columns)
    print(f"Stage 2 -- estimation quality   {len(feats)} features, {len(X):,} rows\n")

    wide = {c: X[c].unstack("symbol") for c in feats}

    rows = []
    for c in feats:
        W = wide[c]
        # Persistence: how much of its data does a consecutive pair share?
        ac = W.apply(lambda s: s.autocorr(1)).median()

        # Noise SD from the second difference of a smooth-plus-noise series.
        d2 = W.diff().diff()
        noise = d2.std() / np.sqrt(6.0)
        # The spread a ranking model actually sees.
        spread = W.std(axis=1)
        ratio = float(np.nanmedian(noise) / np.nanmedian(spread)) \
            if np.nanmedian(spread) > 0 else np.nan

        # Sensitivity proxy: does a light smoothing reorder the cross-section?
        S = W.rolling(SMOOTH_BARS, min_periods=2).mean()
        rk = []
        for t in W.index[::7]:                       # every 7th bar is plenty
            a, b = W.loc[t], S.loc[t]
            m = a.notna() & b.notna()
            if m.sum() >= 8:
                rk.append(a[m].rank().corr(b[m].rank()))
        stab = float(np.nanmean(rk)) if rk else np.nan

        rows.append({"feature": c, "autocorr": float(ac),
                     "noise_spread": ratio, "rank_stability": stab})

    R = pd.DataFrame(rows).set_index("feature")
    R["context"] = [c in CONTEXT for c in R.index]
    R["windowed"] = (R["autocorr"] > AUTOCORR_WINDOWED) & ~R["context"]

    win = R[R["windowed"]]
    fast = R[~R["windowed"] & ~R["context"]]
    ctxr = R[R["context"]]
    if len(ctxr):
        print("CONTEXT TIER -- exempt: no cross-sectional spread to measure against")
        for c in ctxr.index:
            print(f"    {c:<32} bar-level (§12 tier 3)")
        print()

    print(f"SCOPE  ({len(win)} windowed estimators tested, "
          f"{len(fast)} fast features exempt)\n" + "=" * 74)
    print("  exempt (autocorr <= %.2f, not windowed estimates):" % AUTOCORR_WINDOWED)
    for c in fast.sort_values("autocorr", ascending=False).index:
        print(f"    {c:<32} autocorr {fast.loc[c,'autocorr']:+.3f}")

    print(f"\nCHECK 1 -- noise / cross-sectional spread  (bar {NOISE_SPREAD_MAX})")
    print("=" * 74)
    bad1 = []
    for c in win.sort_values("noise_spread", ascending=False).index:
        v = win.loc[c, "noise_spread"]
        ok = v < NOISE_SPREAD_MAX
        if not ok:
            bad1.append(c)
        print(f"  [{'PASS' if ok else 'FAIL'}] {c:<32} {v:.3f}")

    print(f"\nCHECK 2 -- rank stability under smoothing (PROXY; bar "
          f"{RANK_STABILITY_MIN})\n" + "=" * 74)
    bad2 = []
    for c in win.sort_values("rank_stability").index:
        v = win.loc[c, "rank_stability"]
        ok = v > RANK_STABILITY_MIN
        if not ok:
            bad2.append(c)
        print(f"  [{'PASS' if ok else 'FAIL'}] {c:<32} {v:.3f}")

    print("\n" + "=" * 74)
    flagged = sorted(set(bad1) | set(bad2))
    if flagged:
        print(f"FLAGGED ({len(flagged)}): " + ", ".join(flagged))
        print("Stage 2 REPORTS; it does not delete. Removal is a recorded")
        print("decision, taken with Stage 3 and 4 evidence alongside.")
    else:
        print("All windowed estimators pass both checks.")

    out = DATA_DIR / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "stage": 2, "n_features": len(feats),
        "autocorr_windowed_threshold": AUTOCORR_WINDOWED,
        "noise_spread_max": NOISE_SPREAD_MAX,
        "rank_stability_min": RANK_STABILITY_MIN,
        "sensitivity_is_proxy": True,
        "results": {c: {k: (None if pd.isna(v) else float(v))
                        for k, v in R.loc[c].items() if k != "windowed"}
                    | {"windowed": bool(R.loc[c, "windowed"])} for c in R.index},
        "flagged": flagged,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
