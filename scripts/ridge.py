"""Ridge instead of LightGBM, on the identical panel, book and evaluation.

WHY THIS IS WORTH TRYING
------------------------
The failure mode on record is train +2.86 / 2026 -0.35: a model that fits the
training period well and does not generalise. That is the signature of a
high-variance learner on a low signal-to-noise problem. A linear model with L2
shrinkage has far less capacity to memorise, and in cross-sectional equity-style
prediction a regularised linear combination of ranked features is the standard
baseline that GBMs frequently fail to beat out of sample.

WHAT IS HELD FIXED
------------------
Everything except the learner: the 72 frozen features, the within-bar pct-rank
transform, the rank target, the purged walk-forward folds, the book, the hedge
overlay, the funding attachment, and the 2026 evaluation with its
data-completeness cap. Only `fit` and `predict` change.

TWO DIFFERENCES THAT ARE NOT CHOICES
-------------------------------------
1. NaN. LightGBM routes missing values natively; Ridge cannot see one. A missing
   pct-rank is filled with 0.5 -- the NEUTRAL rank, i.e. "no opinion on this
   coin for this feature" -- which is the only fill that does not inject a
   directional view. The fraction filled is reported, because if it were large
   the two learners would not be seeing the same data.

2. Seeds. The LightGBM path averages 8 seeds because a single fit is an
   arbitrary draw (measured Sharpe range 2.51..3.04 across 12 seeds). Ridge has
   a closed-form solution: it is deterministic, so the ensemble is unnecessary
   rather than omitted. That removes a whole source of variance from the result.

ALPHA IS SWEPT, NOT PICKED
---------------------------
Reporting the best alpha would be selection on spent data. The sweep is printed
in full so the shape can be read: a PLATEAU across neighbouring alphas is
evidence of a real effect, a SPIKE at one value is noise. This is the same test
that failed the 30d momentum lookback.

Both holdouts are spent. The 2026 column is a diagnostic, not fresh evidence.
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
from scripts.run_strategy import attach_funding, cache_path, load_panel
from scripts.stage7_importance import per_bar_ic

ALPHAS = [1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0]
CAP = int(pd.Timestamp("2026-07-31", tz="UTC").timestamp() * 1000)
NEUTRAL = 0.5


def ranks(X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Within-bar pct ranks, neutral-filled. Returns the filled fraction."""
    R = X.groupby(level="t_obs").rank(pct=True)
    frac = float(R.isna().to_numpy().mean())
    return R.fillna(NEUTRAL), frac


def fit_predict(Rtr, ytr, Rte, alpha: float) -> np.ndarray:
    """Closed-form ridge on centred inputs. No intercept needed after centring,
    and centring per-fit avoids leaking the test mean into the fit."""
    Xtr = Rtr.to_numpy("float64")
    mu = Xtr.mean(axis=0)
    Xc = Xtr - mu
    yc = ytr - ytr.mean()
    p = Xc.shape[1]
    A = Xc.T @ Xc + alpha * np.eye(p)
    w = np.linalg.solve(A, Xc.T @ yc)
    return (Rte.to_numpy("float64") - mu) @ w


def book(P: pd.DataFrame, seed=None, init_w=None) -> dict:
    res = strategy.run(P, seed=seed, init_weights=init_w)
    m = strategy.metrics(res.ret, turnover=float(res.turnover.mean()))
    return m, res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", type=int, default=60)
    ap.add_argument("--out", default="features/ridge.json")
    args = ap.parse_args()

    lo = splits.TRUE_HOLDOUT2_MS
    Xtr, Ttr, cols, _ = load_panel(lo, lag=args.lag)
    Xte, Tte, _, _ = load_panel(CAP, lower_ms=lo, lag=args.lag)
    print(f"RIDGE vs LIGHTGBM -- {len(cols)} frozen features, lag {args.lag}")
    print(f"  train {len(Xtr):,} rows to {splits._fmt(lo)}")
    print(f"  2026  {len(Xte):,} rows, "
          f"{Xte.index.get_level_values('t_obs').nunique():,} bars "
          f"(capped {splits._fmt(CAP)})")

    Rtr, f1 = ranks(Xtr)
    Rte, f2 = ranks(Xte)
    print(f"  neutral-filled ranks: train {f1:.3%}, 2026 {f2:.3%}"
          f"   <- if large, the learners do not see the same data\n")

    Ytr = Ttr["target"].groupby(level="t_obs").rank(pct=True).to_numpy()
    bars_tr = Xtr.index.get_level_values("t_obs").to_numpy()

    # ---- TRAIN: purged walk-forward out-of-fold, identical folds ---------
    tr_only = bars_tr < splits.TRUE_HOLDOUT1_MS      # the true train block
    fold_bars = np.sort(np.unique(bars_tr[tr_only]))
    flds = list(foldmod.make_folds(fold_bars))
    print(f"  {len(flds)} purged walk-forward folds on "
          f"{len(fold_bars):,} train bars\n")

    print(f"  {'model':<16}{'trainIC':>9}{'trSharpe':>10}{'trRet':>9}"
          f"{'2026IC':>9}{'26Sharpe':>10}{'26Ret':>9}{'26mo+':>7}")
    print("  " + "-" * 79)
    out = {}

    for alpha in ALPHAS:
        sc = pd.Series(np.nan, index=Xtr.index)
        for f in flds:
            tr, te = np.isin(bars_tr, f.train), np.isin(bars_tr, f.test)
            p = fit_predict(Rtr[tr], Ytr[tr], Rtr[te], alpha)
            sc[te] = (p - p.mean()) / (p.std() or 1.0)
        Ptr = attach_funding(Ttr.assign(score=sc).dropna(subset=["score"]))
        ic_tr = per_bar_ic(Ptr["score"].to_numpy(), Ptr["target"].to_numpy(),
                           Ptr.index.get_level_values("t_obs").to_numpy())
        m_tr, res_tr = book(Ptr)

        # ---- 2026: one fit on ALL pre-2026, applied forward, state carried
        p = fit_predict(Rtr, Ytr, Rte, alpha)
        Pte = attach_funding(Tte.assign(score=(p - p.mean()) / (p.std() or 1.0)))
        ic_te = per_bar_ic(Pte["score"].to_numpy(), Pte["target"].to_numpy(),
                           Pte.index.get_level_values("t_obs").to_numpy())
        seed_hist = (res_tr.ret.to_numpy(),
                     Ptr.groupby(level="t_obs")["fwd_rm"].first()
                     .reindex(res_tr.ret.index).to_numpy())
        iw = res_tr.weights.iloc[-1].reindex(
            sorted(Pte.index.get_level_values("symbol").unique())).fillna(0.0)
        m_te, res_te = book(Pte, seed=seed_hist, init_w=iw.to_numpy())

        per = pd.Series(res_te.ret.to_numpy(),
                        index=pd.to_datetime(res_te.ret.index, unit="ms"))
        mo = per.groupby(per.index.to_period("M")).apply(lambda g: (1+g).prod()-1)
        pos = int((mo > 0).sum())

        print(f"  ridge a={alpha:<9.0f}{ic_tr:>+9.4f}{m_tr['sharpe']:>+10.2f}"
              f"{m_tr['ann_return']:>+9.1%}{ic_te:>+9.4f}"
              f"{m_te['sharpe']:>+10.2f}{m_te['ann_return']:>+9.1%}"
              f"{pos:>4}/{len(mo)}")
        out[f"ridge_{alpha:g}"] = {"train": {"ic": ic_tr, **m_tr},
                                   "h2026": {"ic": ic_te, **m_te},
                                   "months_positive_2026": pos}

    print("  " + "-" * 79)
    print(f"  {'LightGBM (rec.)':<16}{'':>9}{2.86:>+10.2f}{0.4871:>+9.1%}"
          f"{'':>9}{-0.35:>+10.2f}{-0.0417:>+9.1%}{3:>4}/7")

    best = max(out, key=lambda k: out[k]["train"]["sharpe"])
    print(f"\n  best on TRAIN: {best} at Sharpe "
          f"{out[best]['train']['sharpe']:+.2f} -> its 2026 is "
          f"{out[best]['h2026']['sharpe']:+.2f}")
    print("  Read the COLUMN SHAPE, not the best cell. A plateau across "
          "neighbouring\n  alphas is signal; a single spike is noise.")

    p = DATA_DIR / args.out
    p.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
