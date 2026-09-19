"""How much of the result is the seed? TRAIN, out of fold.

THE QUESTION
------------
`SEED = 20260828` is one arbitrary number, and LightGBM uses it for
`bagging_fraction` row sampling and `feature_fraction` column sampling. If the
Sharpe moves materially across seeds then the reported figure is one draw from
a distribution and quoting it alone overstates what is known.

TWO SEPARATE THINGS, OFTEN CONFLATED
-------------------------------------
DETERMINISM   Same seed, same machine, same code -> identical numbers? If not,
              nothing downstream can be trusted or re-checked. Tested first,
              by refitting the whole out-of-fold pipeline twice and requiring
              bit-identical scores.

STABILITY     Different seed -> how different a result? This is not a bug if it
              varies; it is sampling noise in the model fit, and the honest
              summary is a distribution rather than a point.

WHAT IS AND IS NOT RESAMPLED
-----------------------------
Only the model fit is reseeded. The folds, the frozen feature list, the target,
the weighting and the hedge are deterministic and identical across runs, so any
dispersion below is attributable to the fit alone.

That also makes this a LOWER BOUND on total uncertainty. It says nothing about
the variation that would come from a different fold layout, a different
universe, or a different sample period -- all of which are larger sources than
seed noise and none of which are measured here.
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
from scripts.stage7_importance import LGB_PARAMS, SEED, FAMILIES
from scripts.run_strategy import attach_funding

N_SEEDS = 12


def oof_scores(seed: int, Rk, Yr, bars, index) -> pd.Series:
    import lightgbm as lgb
    sc = pd.Series(np.nan, index=index)
    for f in foldmod.make_folds(np.sort(np.unique(bars))):
        tr, te = np.isin(bars, f.train), np.isin(bars, f.test)
        m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=seed).fit(
            Rk[tr].to_numpy("float32"), Yr[tr].to_numpy())
        sc[te] = m.predict(Rk[te].to_numpy("float32"))
    return sc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features/seed_stability.json")
    args = ap.parse_args()

    fz = json.loads((DATA_DIR / "features/frozen_list.json").read_text())
    cols = list(fz["features"])
    parts = [pd.read_parquet(DATA_DIR / f"features/{f}.parquet")
             .set_index(["t_obs", "symbol"]) for f in FAMILIES.values()
             if (DATA_DIR / f"features/{f}.parquet").exists()]
    X = pd.concat(parts, axis=1)[cols]
    T = pd.read_parquet(DATA_DIR / "grid/target_8h_lag60.parquet") \
        .set_index(["t_obs", "symbol"])
    i = X.index.intersection(T.index)
    X, T = X.loc[i].sort_index(), T.loc[i].sort_index()
    m = X.index.get_level_values("t_obs") < splits.HOLDOUT1_START_MS
    X, T = X[m], T[m]
    Rk = X.groupby(level="t_obs").rank(pct=True)
    Yr = T["target"].groupby(level="t_obs").rank(pct=True)
    bars = X.index.get_level_values("t_obs").to_numpy()

    print("SEED STABILITY -- book C, no smoothing, train out of fold")
    print(f"  spec: lam {strategy.LAMBDA}, band {strategy.BAND}\n")

    # ---- 1. determinism --------------------------------------------------
    print("DETERMINISM -- same seed twice, must be bit-identical")
    print("=" * 74)
    a = oof_scores(SEED, Rk, Yr, bars, X.index)
    b = oof_scores(SEED, Rk, Yr, bars, X.index)
    av, bv = a.to_numpy(), b.to_numpy()
    # NaN-aware. The first MIN_TRAIN_BARS bars have no out-of-fold score at
    # all, and `np.array_equal` returns False on NaN != NaN -- which is a
    # property of the comparison, not of the pipeline. Compare the NaN masks
    # and the finite values separately.
    mask_same = bool(np.array_equal(np.isnan(av), np.isnan(bv)))
    fin = ~np.isnan(av)
    vals_same = bool(np.array_equal(av[fin], bv[fin]))
    same = mask_same and vals_same
    print(f"  [{'PASS' if same else 'FAIL'}] identical scores across two runs")
    print(f"         {int(fin.sum()):,} scored rows, "
          f"{int((~fin).sum()):,} unscored (pre-first-fold, expected)")
    print(f"         NaN pattern identical: {mask_same}; "
          f"values bit-identical: {vals_same}; "
          f"max|diff| {np.nanmax(np.abs(av - bv)):.3e}")
    if not same:
        print("  Results are NOT reproducible run to run. Everything reported")
        print("  from this pipeline is a single unrepeatable draw.")

    # ---- 2. stability across seeds ---------------------------------------
    print(f"\nSTABILITY -- {N_SEEDS} seeds, only the model fit is reseeded")
    print("=" * 74)
    rows = []
    seeds = [SEED] + [SEED + 1000 * k for k in range(1, N_SEEDS)]
    for s in seeds:
        sc = a if s == SEED else oof_scores(s, Rk, Yr, bars, X.index)
        P = attach_funding(T.assign(score=sc).dropna(subset=["score"]))
        res = strategy.run(P)
        mm = strategy.metrics(res.ret, turnover=float(res.turnover.mean()))
        rows.append({"seed": s, "sharpe": mm["sharpe"],
                     "ann_return": mm["ann_return"],
                     "max_drawdown": mm["max_drawdown"],
                     "turnover": mm["turnover"],
                     "breakeven_bps": mm["breakeven_bps"],
                     })
        print(f"  seed {s:<10} Sharpe {mm['sharpe']:>+6.2f}  "
              f"ann {mm['ann_return']:>+7.2%}  maxDD {mm['max_drawdown']:>+7.2%}"
              f"  turn {mm['turnover']:>5.1%}  b/e {mm['breakeven_bps']:>5.2f}bp")

    D = pd.DataFrame(rows)
    print("\n" + "=" * 74)
    print(f"  {'metric':<16} {'mean':>9} {'sd':>8} {'min':>9} {'max':>9}")
    for k, fmt in [("sharpe", "{:+.2f}"), ("ann_return", "{:+.2%}"),
                   ("max_drawdown", "{:+.2%}"), ("breakeven_bps", "{:.2f}")]:
        v = D[k]
        print(f"  {k:<16} " + " ".join(
            f"{fmt.format(x):>9}" for x in [v.mean(), v.std(), v.min(), v.max()]))

    sh = D["sharpe"]
    rel = sh.std() / sh.mean()
    print(f"\n  Sharpe {sh.mean():+.2f} +/- {sh.std():.2f} "
          f"({rel:.1%} of the mean); reported seed {SEED} gave "
          f"{D.loc[D['seed'] == SEED, 'sharpe'].iloc[0]:+.2f}, which is the "
          f"{(sh < D.loc[D['seed']==SEED,'sharpe'].iloc[0]).mean():.0%} "
          f"percentile of this sample")
    print(f"  every seed is positive: {int((sh > 0).sum())}/{len(sh)}, "
          f"range {sh.min():+.2f} .. {sh.max():+.2f}")
    print("\n  This is a LOWER BOUND on uncertainty: only the model fit is")
    print("  reseeded. Fold layout, universe and sample period are held fixed")
    print("  and are larger sources of variation than seed noise.")

    out = DATA_DIR / args.out
    out.write_text(json.dumps({
        "deterministic": same, "n_seeds": len(seeds),
        "reported_seed": SEED, "per_seed": rows,
        "summary": {k: {"mean": float(D[k].mean()), "sd": float(D[k].std()),
                        "min": float(D[k].min()), "max": float(D[k].max())}
                    for k in ["sharpe", "ann_return", "max_drawdown",
                              "breakeven_bps"]},
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
