"""Small hyperparameter search. TRAIN only.

DELIBERATELY SMALL. Nine configurations, chosen to span the axes that matter
for a noisy panel -- capacity (`num_leaves`), the smallest leaf the model may
form (`min_child_samples`), and how far it is allowed to walk
(`learning_rate` x `n_estimators`). At ~65 bars per feature, a large grid does
not find a better model, it finds the fold noise; the grid is small so that the
selection has something left to generalise.

THE NUMBER A GRID SEARCH REPORTS IS NOT THE NUMBER YOU GET
-----------------------------------------------------------
Picking the best of nine configs on five folds and then quoting that config's
score on those same folds is selection on the evaluation set. The maximum of
nine noisy estimates is biased upward whether or not any config is genuinely
better.

So two numbers are produced and both are shown:

  IN-SAMPLE-SELECTED   best config by mean IC across all five folds, scored on
                       those folds. Optimistic. This is what a naive grid
                       search would report.
  SEQUENTIALLY SELECTED for fold k, the config is chosen using ONLY folds
                       0..k-1 and then scored on fold k. Honest, and the only
                       one that answers "does tuning help".

If the two differ materially, the difference IS the selection bias, measured
rather than argued about.

Seeds are ensembled (3 during the search, for cost) so a config is not
rewarded for a lucky draw. The winner is re-run with the full declared seed
list before anything is adopted.
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
from mft import splits, strategy
from mft.paths import DATA_DIR
from scripts.stage7_importance import (LGB_PARAMS, SEED, SEEDS, FAMILIES,
                                       per_bar_ic)

TUNE_SEEDS = SEEDS[:3]

BASE = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
            min_child_samples=200, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
            verbosity=-1)

GRID = [
    ("baseline",        {}),
    ("shallow",         {"num_leaves": 7}),
    ("deep",            {"num_leaves": 31}),
    ("big-leaf",        {"min_child_samples": 400}),
    ("small-leaf",      {"min_child_samples": 100}),
    ("slow-long",       {"learning_rate": 0.03, "n_estimators": 500}),
    ("fast-short",      {"learning_rate": 0.10, "n_estimators": 150}),
    ("strong-l2",       {"lambda_l2": 20.0}),
    ("shallow+strong",  {"num_leaves": 7, "lambda_l2": 20.0}),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features/tune.json")
    args = ap.parse_args()
    import lightgbm as lgb

    fz = json.loads((DATA_DIR / "features/frozen_list.json").read_text())
    cols = list(fz["features"])
    X = pd.concat([pd.read_parquet(DATA_DIR / f"features/{f}.parquet")
                   .set_index(["t_obs", "symbol"]) for f in FAMILIES.values()
                   if (DATA_DIR / f"features/{f}.parquet").exists()],
                  axis=1)[cols]
    T = pd.read_parquet(DATA_DIR / "grid/target_8h_lag60.parquet") \
        .set_index(["t_obs", "symbol"])
    i = X.index.intersection(T.index)
    X, T = X.loc[i].sort_index(), T.loc[i].sort_index()
    m = X.index.get_level_values("t_obs") < splits.HOLDOUT1_START_MS
    X, T = X[m], T[m]
    splits.assert_train_only(T.reset_index().assign(
        bar_close_time=T.reset_index()["t_obs"]))

    Rk = X.groupby(level="t_obs").rank(pct=True)
    Yr = T["target"].groupby(level="t_obs").rank(pct=True)
    Y = T["target"]
    bars = X.index.get_level_values("t_obs").to_numpy()
    F = foldmod.make_folds(np.sort(np.unique(bars)))

    print(f"TUNING -- {len(GRID)} configs x {len(F)} folds, "
          f"{len(TUNE_SEEDS)} seeds each")
    print("  metric: mean per-bar Spearman IC on the fold's test block\n")

    ic = {}          # (config, fold) -> IC
    for name, over in GRID:
        params = {**BASE, **over}
        row = []
        for f in F:
            tr, te = np.isin(bars, f.train), np.isin(bars, f.test)
            acc = np.zeros(int(te.sum()))
            for sd in TUNE_SEEDS:
                mdl = lgb.LGBMRegressor(**params, random_state=sd).fit(
                    Rk[tr].to_numpy("float32"), Yr[tr].to_numpy())
                p = mdl.predict(Rk[te].to_numpy("float32"))
                acc += (p - p.mean()) / (p.std() or 1.0)
            v = per_bar_ic(acc / len(TUNE_SEEDS), Y[te].to_numpy(), bars[te])
            ic[(name, f.k)] = float(v)
            row.append(v)
        print(f"  {name:<16} " + " ".join(f"{x:+.4f}" for x in row)
              + f"   mean {np.mean(row):+.4f}")

    names = [n for n, _ in GRID]
    mean_ic = {n: float(np.mean([ic[(n, f.k)] for f in F])) for n in names}
    best_all = max(mean_ic, key=mean_ic.get)

    print("\n" + "=" * 74)
    print("IN-SAMPLE-SELECTED (what a naive grid search would report)")
    print(f"  best config '{best_all}'  mean IC {mean_ic[best_all]:+.4f}")
    print(f"  baseline                 mean IC {mean_ic['baseline']:+.4f}")
    print(f"  apparent gain            {mean_ic[best_all]-mean_ic['baseline']:+.4f}")

    print("\nSEQUENTIALLY SELECTED (config for fold k chosen on folds 0..k-1)")
    print("=" * 74)
    seq, base_seq, picks = [], [], []
    for f in F:
        if f.k == 0:
            continue
        prior = [j.k for j in F if j.k < f.k]
        sel = max(names, key=lambda n: np.mean([ic[(n, j)] for j in prior]))
        picks.append((f.k, sel))
        seq.append(ic[(sel, f.k)])
        base_seq.append(ic[("baseline", f.k)])
        print(f"  fold {f.k}: picked '{sel:<16}' -> IC {ic[(sel, f.k)]:+.4f}"
              f"   (baseline {ic[('baseline', f.k)]:+.4f})")
    gain = float(np.mean(seq) - np.mean(base_seq))
    print(f"\n  tuned    mean IC {np.mean(seq):+.4f}")
    print(f"  baseline mean IC {np.mean(base_seq):+.4f}")
    print(f"  HONEST GAIN      {gain:+.4f}   "
          f"(tuned wins {sum(1 for a, b in zip(seq, base_seq) if a > b)}"
          f"/{len(seq)} folds)")

    bias = (mean_ic[best_all] - mean_ic["baseline"]) - gain
    print(f"\n  SELECTION BIAS   {bias:+.4f} -- the difference between what the")
    print("  grid search appears to buy and what it actually buys.")

    print("\n" + "=" * 74)
    if gain > 0.002 and sum(1 for a, b in zip(seq, base_seq) if a > b) >= 3:
        print(f"ADOPT '{best_all}': the gain survives sequential selection.")
        verdict = best_all
    else:
        print("KEEP THE BASELINE. The apparent gain does not survive being")
        print("chosen without seeing the fold it is scored on, which is the")
        print("only way it would ever be chosen in practice.")
        verdict = "baseline"

    out = DATA_DIR / args.out
    out.write_text(json.dumps({
        "grid": {n: o for n, o in GRID}, "base": BASE,
        "tune_seeds": TUNE_SEEDS,
        "ic": {f"{n}|{k}": v for (n, k), v in ic.items()},
        "mean_ic": mean_ic, "best_in_sample": best_all,
        "sequential_picks": picks,
        "sequential_tuned_ic": float(np.mean(seq)),
        "sequential_baseline_ic": float(np.mean(base_seq)),
        "honest_gain": gain, "selection_bias": float(bias),
        "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
