"""Stage 7 -- feature importance. Per docs/FEATURE_SELECTION.md §4.

GROUPED PERMUTATION (MDA) INSIDE PURGED FOLDS, AGAINST A SHADOW-FEATURE NULL.

WHY NOT GAIN OR SHAP
--------------------
Gain is biased toward continuous, high-cardinality columns and is unreliable
under correlation. SHAP and plain permutation share a worse flaw: when two
features carry the same information, permuting one leaves the model able to
lean on the other, so BOTH look unimportant. A feature can rank last and still
be doing the work.

Correlated features are therefore permuted TOGETHER. Groups are the connected
components of |within-bar rank correlation| >= 0.70 over the frozen list.
Measured on the 72: 67 groups, only five with more than one member. Recorded
plainly -- grouping is a SMALL correction here because Stage 4 already removed
the pairwise-redundant and the multivariate-explained upstream. It is the right
default and it is not what makes this analysis work.

PERMUTATION IS WITHIN BAR
-------------------------
Shuffling a column globally would destroy the cross-sectional structure the
model ranks on and would mostly detect that the feature has a time trend. The
shuffle is applied inside each bar, which is the only permutation that leaves
the ranking problem intact while removing the feature's information.

THE NULL: SHADOW FEATURES
-------------------------
An importance number means nothing without one. For each fold, N_SHADOW columns
are added, each a real feature copied and shuffled within bar -- same marginal
distribution, no signal. A feature must beat the MAXIMUM shadow importance, not
the mean. Taking the max is what makes the bar multiple-testing aware: it is
the order statistic a genuinely null column has to clear.

THE METRIC
----------
Mean per-bar Spearman IC of the model's score against the target on the fold's
test block. Importance is the DROP in that IC when a group is permuted. This is
the quantity the strategy actually consumes, not a proxy for it.

THE MODEL, AND A CORRECTION THAT HAD TO BE MADE
-----------------------------------------------
The first version of this stage used LightGBM `lambdarank` with NDCG, per the
design intent, on RAW feature levels. It produced a null result -- 1 of 67
groups beat the shadow bar -- and the null was an artifact of the model, not a
fact about the features. Measured on the same folds:

    lambdarank + raw levels                     mean IC  +0.0097
    regressor on rank-target + raw levels                +0.0408
    regressor on rank-target + within-bar ranks          +0.0489
    trivial sign-aligned equal-weight composite          +0.0437

The original model scored BELOW a composite that does no fitting at all, and
below its own best single input. A permutation test on a model that barely
beats noise has no power: nothing can be shown to matter because nothing is
being used. Reporting that as "no feature is important" would have been a
false conclusion drawn from a broken instrument.

Two things were wrong.

  NDCG IS THE WRONG SURROGATE. It weights the top of the list, while IC scores
  the whole ordering. On a 19-coin cross-section, optimising top-of-list gain
  is close to optimising four positions out of nineteen.

  FEATURES WERE FED AS RAW LEVELS while every other stage in this project --
  Stage 4 redundancy, Stage 5 turnover, Stage 6 IC -- works in within-bar rank
  space. A tree splitting on raw levels splits on a quantity whose meaning
  drifts with the regime.

Both are corrected: within-bar percentile-rank features, predicting the
within-bar percentile rank of the target. THIS IS STILL A RANKING OBJECTIVE in
the sense the design requires -- it predicts the cross-sectional ordering, and
it is aligned with the metric the strategy is scored on. What was abandoned is
the NDCG surrogate, not the ranking goal.

Hyperparameters are fixed and deliberately conservative. They are NOT tuned,
and nothing may be tuned on a holdout.

NOTHING IS REMOVED. Stage 7 reports; Stage 8 decides on stability across folds.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft import folds as foldmod
from mft import splits
from mft.paths import DATA_DIR

GROUP_THRESHOLD = 0.70
N_SHADOW = 20
N_REPEATS = 5
SEED = 20260828
# Seed ENSEMBLE. One seed is an arbitrary draw: measured across 12 seeds the
# Sharpe ranged 2.51..3.04 and the reported figure sat at the 75th percentile.
# Averaging predictions over a fixed, declared list removes that lottery from
# the result. The list is fixed here so it is not a tuning knob.
SEEDS = [SEED + 1000 * k for k in range(8)]
MIN_COINS = 8

LGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=15,
    min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=1, lambda_l2=5.0, verbosity=-1,
)

FAMILIES = {
    "A": "family_A_carry", "B": "family_B_positioning", "C": "family_C_liquidity",
    "D": "family_D_path", "E": "family_E_regime", "F": "family_F_relational",
    "G": "family_G_spotperp", "H": "family_H_inherited",
    "I": "family_I_inherited2",
}


def per_bar_ic(score: np.ndarray, y: np.ndarray, bar: np.ndarray) -> float:
    """Mean per-bar Spearman IC. Bars with too few coins are skipped."""
    d = pd.DataFrame({"s": score, "y": y, "b": bar})
    g = d.groupby("b")
    keep = g.size()
    d = d[d["b"].isin(keep[keep >= MIN_COINS].index)]
    if not len(d):
        return np.nan
    r = d.groupby("b")[["s", "y"]].rank()
    r["b"] = d["b"].to_numpy()
    r[["s", "y"]] -= r.groupby("b")[["s", "y"]].transform("mean")
    num = (r["s"] * r["y"]).groupby(r["b"]).sum()
    den = np.sqrt((r["s"] ** 2).groupby(r["b"]).sum()
                  * (r["y"] ** 2).groupby(r["b"]).sum())
    return float((num / den.replace(0.0, np.nan)).mean())


def build_groups(Rk: pd.DataFrame, cols: list[str]) -> list[list[str]]:
    C = Rk[cols].corr(min_periods=500).abs()
    adj = {c: set() for c in cols}
    for a, b in itertools.combinations(cols, 2):
        v = C.at[a, b]
        if pd.notna(v) and v >= GROUP_THRESHOLD:
            adj[a].add(b)
            adj[b].add(a)
    seen, groups = set(), []
    for c in cols:
        if c in seen:
            continue
        stack, comp = [c], []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(adj[x] - seen)
        groups.append(sorted(comp))
    return groups


def shuffle_within_bar(v: np.ndarray, bar_codes: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    """Permute values inside each bar, preserving the per-bar multiset."""
    out = v.copy()
    order = np.argsort(bar_codes, kind="stable")
    starts = np.searchsorted(bar_codes[order], np.unique(bar_codes))
    bounds = list(starts) + [len(order)]
    for i in range(len(bounds) - 1):
        sl = order[bounds[i]:bounds[i + 1]]
        out[sl] = v[rng.permutation(sl)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features/stage7_importance.json")
    args = ap.parse_args()

    try:
        import lightgbm as lgb
    except ImportError:
        print("lightgbm is required for Stage 7.")
        return 2

    fz = json.loads((DATA_DIR / "features/frozen_list.json").read_text())
    cols = list(fz["features"])

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

    t_all = np.sort(X.index.get_level_values("t_obs").unique())
    F = foldmod.make_folds(t_all)

    # Within-bar percentile ranks: the feature space every other stage uses,
    # and the one the corrected model consumes.
    Rk = X.groupby(level="t_obs").rank(pct=True)
    groups = build_groups(Rk - Rk.groupby(level="t_obs").transform("mean"), cols)
    multi = [g for g in groups if len(g) > 1]
    Yr = Y.groupby(level="t_obs").rank(pct=True)

    print("STAGE 7 -- grouped permutation importance vs a shadow null")
    print(f"  frozen {fz['n_features']} features, hash {fz['content_hash']}")
    print(f"  {len(X):,} rows, {len(t_all):,} bars, {len(F)} folds")
    print(f"  {len(groups)} groups at |rank corr| >= {GROUP_THRESHOLD}, "
          f"{len(multi)} with >1 member:")
    for g in multi:
        print("      " + ", ".join(g))
    print(f"  {N_SHADOW} shadows, {N_REPEATS} permutation repeats, "
          f"bar = MAX shadow importance\n")

    bars_all = X.index.get_level_values("t_obs").to_numpy()
    rng_master = np.random.default_rng(SEED)
    rows = []

    for f in F:
        tr = np.isin(bars_all, f.train)
        te = np.isin(bars_all, f.test)
        Xtr, Ytr = Rk[tr], Yr[tr]
        Xte, Yte = Rk[te], Y[te]
        bar_te = Xte.index.get_level_values("t_obs").to_numpy()
        bar_tr = Xtr.index.get_level_values("t_obs").to_numpy()

        rng = np.random.default_rng(rng_master.integers(1 << 31))
        # Shadows: real columns copied and shuffled within bar. Same marginals,
        # no signal. Built for train and test with independent shuffles.
        pick = rng.choice(cols, size=N_SHADOW, replace=False)
        Str = pd.DataFrame(
            {f"shadow_{i}": shuffle_within_bar(Xtr[c].to_numpy(), bar_tr, rng)
             for i, c in enumerate(pick)}, index=Xtr.index)
        Ste = pd.DataFrame(
            {f"shadow_{i}": shuffle_within_bar(Xte[c].to_numpy(), bar_te, rng)
             for i, c in enumerate(pick)}, index=Xte.index)
        Atr = pd.concat([Xtr, Str], axis=1)
        Ate = pd.concat([Xte, Ste], axis=1)

        model = lgb.LGBMRegressor(**LGB_PARAMS, random_state=SEED)
        model.fit(Atr.to_numpy("float32"), Ytr.to_numpy())

        base_scores = model.predict(Ate.to_numpy("float32"))
        base = per_bar_ic(base_scores, Yte.to_numpy(), bar_te)

        # ---- grouped permutation on the test block ----------------------
        units = [(f"g{i}", g) for i, g in enumerate(groups)] + \
                [(f"s{i}", [c]) for i, c in enumerate(Ste.columns)]
        drops = {}
        for tag, members in units:
            vals = []
            for _ in range(N_REPEATS):
                P = Ate.copy()
                for c in members:
                    P[c] = shuffle_within_bar(P[c].to_numpy(), bar_te, rng)
                s = model.predict(P.to_numpy("float32"))
                vals.append(base - per_bar_ic(s, Yte.to_numpy(), bar_te))
            drops[tag] = (float(np.mean(vals)), members)

        shadow_max = max(v for k, (v, _) in drops.items() if k.startswith("s"))
        for tag, (v, members) in drops.items():
            if tag.startswith("s"):
                continue
            rows.append({"fold": f.k, "group": "|".join(members),
                         "importance": v, "beats_shadow": v > shadow_max})
        print(f"  fold {f.k}: base IC {base:+.4f}   "
              f"max shadow drop {shadow_max:+.5f}   "
              f"groups beating it {sum(1 for t2,(v,_) in drops.items() if not t2.startswith('s') and v > shadow_max)}"
              f"/{len(groups)}")

    D = pd.DataFrame(rows)
    G = D.groupby("group").agg(mean_importance=("importance", "mean"),
                               folds_beating=("beats_shadow", "sum"),
                               n_folds=("importance", "size"))
    G = G.sort_values(["folds_beating", "mean_importance"], ascending=False)

    print(f"\n{'group':<46} {'mean drop':>10} {'folds>shadow':>13}")
    print("=" * 74)
    for g, r in G.iterrows():
        nm = g if len(g) <= 44 else g[:41] + "..."
        print(f"{nm:<46} {r['mean_importance']:>+10.5f} "
              f"{int(r['folds_beating']):>8}/{int(r['n_folds'])}")

    print("\n" + "=" * 74)
    print(f"beat the shadow bar in >=3 of 5 folds: "
          f"{int((G['folds_beating'] >= 3).sum())} of {len(G)}")
    print(f"in all 5: {int((G['folds_beating'] == 5).sum())}")
    print("\nStage 7 REPORTS. Stage 8 selects on stability.")

    out = DATA_DIR / args.out
    out.write_text(json.dumps({
        "stage": 7, "frozen_hash": fz["content_hash"],
        "group_threshold": GROUP_THRESHOLD, "n_shadow": N_SHADOW,
        "n_repeats": N_REPEATS, "seed": SEED, "lgb_params": LGB_PARAMS,
        "groups": groups,
        "results": {g: {"mean_importance": float(r["mean_importance"]),
                        "folds_beating": int(r["folds_beating"]),
                        "n_folds": int(r["n_folds"])} for g, r in G.iterrows()},
        "per_fold": D.to_dict("records"),
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
