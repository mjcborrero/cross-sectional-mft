"""Stage 8 -- selection by stability. Per docs/FEATURE_SELECTION.md §5.

THE CRITERION IS STABILITY, NOT MEAN IMPORTANCE
------------------------------------------------
At ~65 bars per feature a single fold's ranking is noise. A group that wins one
fold and loses four is a fluctuation; one that clears its shadow bar in most
folds is a finding. So selection is on the FRACTION of folds in which a group
beat the max-shadow bar, which targets false discoveries directly.

THE LEAK THIS STAGE HAD TO AVOID
---------------------------------
Stage 7 measured each group's importance ON each fold's TEST block. Selecting
with those numbers and then evaluating on the same blocks would be selection
leakage -- the exact failure §2 names as "the most common way feature selection
lies". It is avoided by making the selection SEQUENTIAL:

    fold k is scored using only the importances from folds 0 .. k-1

Fold 0 therefore has no evaluation (nothing precedes it) and the honest
comparison runs on folds 1-4. A group is selected for fold k when it beat the
bar in at least half of the folds before k.

A separate "final" set, using all five folds, is reported for use against the
holdouts. That is legitimate because the holdouts have never been read -- but
it is NOT the set the numbers below are earned on, and the two are kept
visibly apart.

WHAT IS COMPARED
----------------
Three models per fold, identical except for their feature set:

    ALL 72      everything that survived the freeze
    SELECTED    only the stably-important groups, chosen from earlier folds
    BENCHMARK   resid_reversal_8h alone (FEATURE_LIST_FROZEN.md §6 rule 1)

If SELECTED does not beat ALL, the selection did nothing and should be said so
plainly rather than presented as a refinement.

MULTIPLE TESTING IS OVER THE 9 MECHANISM FAMILIES
--------------------------------------------------
The project's standing rule. Each family is permuted as a block, and the count
of families clearing the bar is what the Bonferroni correction applies to --
not the 72 columns, which would both overcorrect and misattribute.
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
from scripts.stage7_importance import (LGB_PARAMS, MIN_COINS, N_REPEATS, SEED,
                                       FAMILIES, per_bar_ic, shuffle_within_bar)

PI = 0.5            # a group must clear the bar in at least half the prior folds
BENCHMARK = "resid_reversal_8h"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features/stage8_selection.json")
    args = ap.parse_args()

    import lightgbm as lgb

    fz = json.loads((DATA_DIR / "features/frozen_list.json").read_text())
    s7 = json.loads((DATA_DIR / "features/stage7_importance.json").read_text())
    cols = list(fz["features"])
    owner = {c: m["family"] for c, m in fz["features"].items()}

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
    X, Y = X.loc[idx].sort_index(), Y.loc[idx].sort_index()
    Rk = X.groupby(level="t_obs").rank(pct=True)
    Yr = Y.groupby(level="t_obs").rank(pct=True)
    bars = X.index.get_level_values("t_obs").to_numpy()
    F = foldmod.make_folds(np.sort(np.unique(bars)))

    # ---- per-fold selection record from Stage 7 --------------------------
    beat = {}                      # group -> {fold: bool}
    for r in s7["per_fold"]:
        beat.setdefault(r["group"], {})[r["fold"]] = bool(r["beats_shadow"])
    groups = [g.split("|") for g in beat]

    print("STAGE 8 -- selection by stability")
    print(f"  {len(groups)} groups, {len(F)} folds, pi = {PI:.0%} of prior folds")
    print(f"  selection is SEQUENTIAL: fold k uses only folds 0..k-1\n")

    # ---- sequential, leak-free evaluation --------------------------------
    print("SEQUENTIAL EVALUATION (fold 0 has no prior folds, so no evaluation)")
    print("=" * 74)
    print(f"  {'fold':>4} {'n_sel':>6} {'ALL 72':>9} {'SELECTED':>10} "
          f"{'BENCHMARK':>10}")
    rows = []
    for f in F:
        if f.k == 0:
            continue
        prior = range(f.k)
        sel_groups = [g for g in beat
                      if sum(beat[g].get(j, False) for j in prior) >= PI * f.k]
        sel = sorted({c for g in sel_groups for c in g.split("|")})
        if not sel:
            sel = [BENCHMARK]

        tr, te = np.isin(bars, f.train), np.isin(bars, f.test)
        bte, yte = bars[te], Y[te].to_numpy()
        out = {}
        for tag, feats in [("all", cols), ("sel", sel), ("bench", [BENCHMARK])]:
            m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=SEED)
            m.fit(Rk[tr][feats].to_numpy("float32"), Yr[tr].to_numpy())
            out[tag] = per_bar_ic(m.predict(Rk[te][feats].to_numpy("float32")),
                                  yte, bte)
        rows.append({"fold": f.k, "n_selected": len(sel), "selected": sel, **out})
        print(f"  {f.k:>4} {len(sel):>6} {out['all']:>+9.4f} {out['sel']:>+10.4f} "
              f"{out['bench']:>+10.4f}")

    D = pd.DataFrame(rows)
    print(f"  {'mean':>4} {'':>6} {D['all'].mean():>+9.4f} "
          f"{D['sel'].mean():>+10.4f} {D['bench'].mean():>+10.4f}")

    better_all = int((D["sel"] > D["all"]).sum())
    better_bench = int((D["sel"] > D["bench"]).sum())
    print(f"\n  SELECTED beats ALL 72 in {better_all}/{len(D)} folds, "
          f"beats BENCHMARK in {better_bench}/{len(D)}")

    # ---- family-level test, the unit multiple testing applies to ---------
    print(f"\nFAMILY-LEVEL PERMUTATION ({len(FAMILIES)} families, the unit the")
    print("project corrects over -- not the 72 columns)")
    print("=" * 74)
    rng = np.random.default_rng(SEED)
    fam_rows = []
    for f in F:
        tr, te = np.isin(bars, f.train), np.isin(bars, f.test)
        bte, yte = bars[te], Y[te].to_numpy()
        m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=SEED)
        m.fit(Rk[tr].to_numpy("float32"), Yr[tr].to_numpy())
        Rte = Rk[te]
        base = per_bar_ic(m.predict(Rte.to_numpy("float32")), yte, bte)

        # shadow bar, same construction as Stage 7
        shadows = []
        for c in rng.choice(cols, size=10, replace=False):
            P = Rte.copy()
            P[c] = shuffle_within_bar(P[c].to_numpy(), bte, rng)
            shadows.append(base - per_bar_ic(m.predict(P.to_numpy("float32")),
                                             yte, bte))
        bar_ = max(shadows)

        for code in FAMILIES:
            members = [c for c in cols if owner[c] == code]
            if not members:
                continue
            vals = []
            for _ in range(N_REPEATS):
                P = Rte.copy()
                for c in members:
                    P[c] = shuffle_within_bar(P[c].to_numpy(), bte, rng)
                vals.append(base - per_bar_ic(m.predict(P.to_numpy("float32")),
                                              yte, bte))
            fam_rows.append({"fold": f.k, "family": code, "n": len(members),
                             "drop": float(np.mean(vals)),
                             "beats": float(np.mean(vals)) > bar_})

    FF = pd.DataFrame(fam_rows).groupby("family").agg(
        n=("n", "first"), mean_drop=("drop", "mean"), folds=("beats", "sum"))
    FF = FF.sort_values(["folds", "mean_drop"], ascending=False)
    # The bar is the MAX of 10 shadow drops, so under the null that a family's
    # drop is exchangeable with a shadow's, P(clear the bar in one fold) =
    # 1/11. Folds are independent blocks, so the count is binomial(5, 1/11),
    # and Bonferroni multiplies by the 9 families tested. This is the
    # correction the project's rule specifies, applied at the unit it names.
    from math import comb
    p0 = 1.0 / 11.0
    nF = len(F)

    def fam_p(k: int) -> float:
        tail = sum(comb(nF, i) * p0 ** i * (1 - p0) ** (nF - i)
                   for i in range(k, nF + 1))
        return min(1.0, tail * len(FF))

    print(f"  null: bar is max of 10 shadows -> P(clear one fold) = 1/11;"
          f" Bonferroni x{len(FF)}")
    print(f"  {'fam':<4} {'cols':>5} {'mean drop':>11} {'folds>shadow':>13} "
          f"{'p (corr)':>10}")
    for c, r in FF.iterrows():
        p = fam_p(int(r["folds"]))
        star = "  *" if p < 0.05 else ""
        print(f"  {c:<4} {int(r['n']):>5} {r['mean_drop']:>+11.5f} "
              f"{int(r['folds']):>8}/{nF} {p:>10.4f}{star}")
    sig_fams = [c for c, r in FF.iterrows() if fam_p(int(r["folds"])) < 0.05]
    print(f"\n  families surviving correction: "
          + (", ".join(sig_fams) if sig_fams else "none"))

    # ---- the set to carry to the holdouts --------------------------------
    final_groups = [g for g in beat
                    if sum(beat[g].values()) >= PI * len(F) + 0.5]
    final = sorted({c for g in final_groups for c in g.split("|")})
    if BENCHMARK not in final:
        final.append(BENCHMARK)
        final = sorted(final)

    print("\n" + "=" * 74)
    print(f"FINAL SET FOR THE HOLDOUTS ({len(final)} features, >=3 of 5 folds)")
    print("  NOT the set the table above is earned on -- that one is sequential.")
    for c in final:
        print(f"    {c}  [{owner.get(c, '?')}]")

    print("\n" + "=" * 74)
    if better_all >= 3:
        print("Selection HELPS: the pruned set beats all 72 in most folds.")
    elif better_all <= 1 and len(D) >= 4:
        print("Selection DOES NOT HELP. The pruned set loses to all 72 in most")
        print("folds, so the honest conclusion is that stability selection did")
        print("not find a better subset than keeping everything -- and the")
        print("regularised model is already handling the noise columns.")
    else:
        print("Selection is a WASH: no clear gain over keeping all 72.")

    out = DATA_DIR / args.out
    out.write_text(json.dumps({
        "stage": 8, "pi": PI, "sequential": True,
        "frozen_hash": fz["content_hash"],
        "per_fold": D.to_dict("records"),
        "mean_ic": {"all": float(D["all"].mean()), "selected": float(D["sel"].mean()),
                    "benchmark": float(D["bench"].mean())},
        "selected_beats_all_folds": better_all,
        "selected_beats_benchmark_folds": better_bench,
        "family_test": {c: {"n": int(r["n"]), "mean_drop": float(r["mean_drop"]),
                            "folds": int(r["folds"]),
                            "p_bonferroni": fam_p(int(r["folds"]))}
                        for c, r in FF.iterrows()},
        "families_significant": sig_fams,
        "final_set": final,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
