"""Combinatorial Purged Cross-Validation of the frozen book, on TRAIN only.

Lopez de Prado, Advances in Financial Machine Learning, ch. 12.

WHAT IT ADDS OVER THE WALK-FORWARD
-----------------------------------
The pipeline's validation is five purged expanding walk-forward folds
(mft/folds.py): one backtest path, one out-of-fold Sharpe (+2.86). A single
path is a single draw. CPCV splits the train bars into N contiguous groups,
takes every combination of k groups as a test set, fits on the rest, and then
STITCHES the test predictions into phi = k/N * C(N,k) complete backtest
paths -- each path covering every bar exactly once, each bar scored by a
model that never saw its group. With N=6, k=2 that is 15 fits and 5 paths.

The output is a DISTRIBUTION of out-of-fold Sharpes rather than one number.
A strategy whose walk-forward Sharpe sits inside the CPCV spread is
consistent; one whose walk-forward number is above every path was lucky in
its ordering.

WHAT IT DOES NOT ADD, STATED PLAINLY
-------------------------------------
CPCV trains on groups that come AFTER the test group. That is the point --
it asks whether the signal holds regardless of ordering -- but it means a
CPCV path is not a forward simulation. For a signal whose edge is
non-stationary (this one's is: it lives in low-dispersion bars and 2026's
violent tercile broke it) a CPCV Sharpe can be higher OR lower than
walk-forward, and the difference is itself information about stationarity.

CPCV touches TRAIN ONLY. The two holdouts were spend-once tests and are not
resampled here; this is additional evidence about the train period, not a
replacement for the holdout discipline.

PURGE AND EMBARGO
-----------------
For each test block, train bars within PURGE bars on either side are dropped
(their labels overlap the test window, or the test labels overlap them), and
a further EMBARGO bars after the block are dropped for serial correlation.
PURGE matches mft/folds.py (2 bars = ceil((8h+60min)/8h)); EMBARGO adds 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft import splits, strategy
from mft.paths import DATA_DIR
from scripts.run_strategy import attach_funding, load_panel
from scripts.stage7_importance import LGB_PARAMS, SEEDS, per_bar_ic
from mft.folds import MIN_TRAIN_BARS

N_GROUPS = 6
K_TEST = 2
PURGE = 2
EMBARGO = 2


def assemble_paths(splits_: list[tuple[int, ...]], n_groups: int) -> list[dict]:
    """Greedy fill: each split's test groups go to the lowest-numbered path
    that does not yet hold that group. Asserts every path is complete."""
    n_paths = K_TEST * len(splits_) // n_groups
    paths = [dict() for _ in range(n_paths)]
    for si, tg in enumerate(splits_):
        for g in tg:
            for p in paths:
                if g not in p:
                    p[g] = si
                    break
            else:
                raise RuntimeError(f"no path free for group {g} of split {si}")
    for i, p in enumerate(paths):
        assert sorted(p) == list(range(n_groups)), f"path {i} incomplete: {sorted(p)}"
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features/cpcv.json")
    a = ap.parse_args()
    import lightgbm as lgb

    X, T, cols, _ = load_panel(splits.TRUE_HOLDOUT1_MS, lag=60)
    bars = X.index.get_level_values("t_obs").to_numpy()
    ub = np.sort(np.unique(bars))
    R = X.groupby(level="t_obs").rank(pct=True)
    Y = T["target"].groupby(level="t_obs").rank(pct=True).to_numpy()
    bar_idx = np.searchsorted(ub, bars)                    # row -> bar index

    edges = np.linspace(0, len(ub), N_GROUPS + 1).astype(int)
    group_of_bar = np.zeros(len(ub), int)
    for g in range(N_GROUPS):
        group_of_bar[edges[g]:edges[g + 1]] = g
    grp = group_of_bar[bar_idx]                            # row -> group
    all_splits = list(combinations(range(N_GROUPS), K_TEST))
    paths = assemble_paths(all_splits, N_GROUPS)

    print(f"CPCV -- {N_GROUPS} groups, {K_TEST} test  ->  {len(all_splits)} fits, "
          f"{len(paths)} paths;  purge {PURGE}, embargo {EMBARGO} bars")
    print(f"  {len(ub):,} train bars, groups of ~{len(ub)//N_GROUPS} bars "
          f"({splits._fmt(int(ub[0]))} .. {splits._fmt(int(ub[-1]))})\n")

    # ---- one fit per split ------------------------------------------------
    scores = {}                                            # split -> Series
    split_ic = {}
    for si, tg in enumerate(all_splits):
        te = np.isin(grp, tg)
        keep = ~te
        for g in tg:                                       # purge + embargo
            lo, hi = edges[g], edges[g + 1] - 1
            keep &= ~((bar_idx >= lo - PURGE) & (bar_idx <= hi + PURGE + EMBARGO))
        acc = np.zeros(int(te.sum()))
        for sd in SEEDS:
            m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=sd).fit(
                R[keep].to_numpy("float32"), Y[keep])
            p = m.predict(R[te].to_numpy("float32"))
            acc += (p - p.mean()) / (p.std() or 1.0)
        sc = pd.Series(acc / len(SEEDS), index=X.index[te])
        scores[si] = sc
        split_ic[si] = per_bar_ic(sc.to_numpy(), T["target"].to_numpy()[te], bars[te])
        print(f"  split {si:>2} test {tg}  train {int(keep.sum()):>6,} rows  "
              f"IC {split_ic[si]:+.4f}")

    # ---- stitch paths and run the book on each ---------------------------
    wf_start = int(ub[MIN_TRAIN_BARS])
    print(f"\n  full span ({len(ub):,} bars)                        | same bars as "
          f"walk-forward ({len(ub)-MIN_TRAIN_BARS:,}, from {splits._fmt(wf_start)})")
    print(f"  {'path':<6}{'Sharpe':>8}{'ann':>9}{'maxDD':>9}{'hit':>7}{'turn':>7}   "
          f"| {'Sharpe':>6}{'maxDD':>8}   groups<-splits")
    res = []
    for pi, p in enumerate(paths):
        # Each split scored TWO groups; take only group g's rows from the
        # split assigned to (path, g).
        parts = [scores[p[g]][scores[p[g]].index.isin(X.index[grp == g])]
                 for g in range(N_GROUPS)]
        full = pd.concat(parts).sort_index()
        assert len(full) == len(X), (len(full), len(X))   # every bar once
        P = attach_funding(T.loc[full.index].assign(score=full))
        r = strategy.run(P)
        m = strategy.metrics(r.ret, turnover=float(r.turnover.mean()))
        # SAME BARS as the walk-forward OOF: its first MIN_TRAIN_BARS bars are
        # never scored out-of-fold, so the recorded +2.86 covers only the tail.
        # Comparing a full-span CPCV path to it conflates method with period.
        rs = r.ret[r.ret.index >= wf_start]
        ms = strategy.metrics(rs, turnover=float(r.turnover[r.turnover.index >= wf_start].mean()))
        res.append({"path": pi, **{k: m[k] for k in
                    ("sharpe", "ann_return", "max_drawdown", "hit_rate", "turnover", "bars")},
                    "same_bars_as_wf": {k: ms[k] for k in
                    ("sharpe", "ann_return", "max_drawdown", "bars")},
                    "assignment": {g: p[g] for g in range(N_GROUPS)}})
        print(f"  {pi:<6}{m['sharpe']:>+8.2f}{m['ann_return']:>+9.1%}"
              f"{m['max_drawdown']:>+9.1%}{m['hit_rate']:>7.1%}{m['turnover']:>7.1%}   "
              f"| {ms['sharpe']:>+6.2f}{ms['max_drawdown']:>+8.1%}   "
              + " ".join(f"{g}<-{p[g]}" for g in range(N_GROUPS)))

    sh = np.array([r_["sharpe"] for r_ in res])
    wf = json.loads((DATA_DIR / "features/strategy_train.json").read_text())["metrics"]["sharpe"]
    shs = np.array([r_["same_bars_as_wf"]["sharpe"] for r_ in res])
    print("\n" + "=" * 70)
    print(f"  CPCV Sharpe, full span : mean {sh.mean():+.2f}  sd {sh.std(ddof=1):.2f}  "
          f"min {sh.min():+.2f}  max {sh.max():+.2f}   ({len(sh)} paths)")
    print(f"  CPCV Sharpe, WF bars   : mean {shs.mean():+.2f}  sd {shs.std(ddof=1):.2f}  "
          f"min {shs.min():+.2f}  max {shs.max():+.2f}   <- like-for-like")
    print(f"  walk-forward (recorded): {wf:+.2f}")
    pos = (wf - shs.mean()) / shs.std(ddof=1) if shs.std(ddof=1) else np.nan
    print(f"  walk-forward sits {pos:+.2f} sd from the same-bars CPCV mean")
    sh = shs                                            # verdict on like-for-like
    print(f"  per-split test IC: mean {np.mean(list(split_ic.values())):+.4f}  "
          f"min {min(split_ic.values()):+.4f}  max {max(split_ic.values()):+.4f}")
    if wf > sh.max():
        print("  -> walk-forward is ABOVE every CPCV path: its ordering was favourable.")
    elif wf < sh.min():
        print("  -> walk-forward is BELOW every CPCV path: expanding-window training")
        print("     hurt it, or the edge is stronger when trained on later data.")
    else:
        print("  -> walk-forward is inside the CPCV spread: consistent.")

    out = {"n_groups": N_GROUPS, "k_test": K_TEST, "purge": PURGE, "embargo": EMBARGO,
           "n_splits": len(all_splits), "n_paths": len(paths),
           "split_ic": {str(k): v for k, v in split_ic.items()},
           "paths": res,
           "full_span": {"sharpe_mean": float(np.mean([r_["sharpe"] for r_ in res])),
                         "sharpe_sd": float(np.std([r_["sharpe"] for r_ in res], ddof=1))},
           "same_bars_as_wf": {"sharpe_mean": float(sh.mean()), "sharpe_sd": float(sh.std(ddof=1)),
                               "sharpe_min": float(sh.min()), "sharpe_max": float(sh.max()),
                               "wf_start": wf_start},
           "walk_forward_sharpe": wf}
    pth = DATA_DIR / a.out
    pth.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
