"""Is the score->return relationship monotonic, and does the BOOK sit on the
part of it that works?

scripts/ic_vs_pnl.py found, in 2026, positive IC in every space (+0.031 target,
+0.027 dollar residual) while the top-5 leg earned -1.57 bps and the bottom-5
leg earned +3.75 bps. Positive average rank correlation with INVERTED extremes
is only possible if the relationship is strongly non-monotonic -- so the shape
has to be looked at directly rather than summarised by one number.

This bins every bar's cross-section into quintiles by score and reports the
mean forward dollar residual of each bin, for TRAIN and 2026 separately.

WHAT THE TWO OUTCOMES MEAN
--------------------------
  * Monotone in train, inverted tails in 2026 -> a regime flip. The book was
    correctly specified and 2026 broke it.
  * Inverted tails in BOTH -> the book has always been reading the wrong part
    of its own signal, and train's +2.86 came from somewhere other than the
    ranking edge. That would be far more serious, and would mean the
    top-k/bottom-k construction is the defect rather than the signal.

Q1 is the LOWEST score (the book's short leg), Q5 the highest (long leg).
A well-specified long/short book needs Q5 > Q1. IC only needs the average
slope to be positive, which is a much weaker condition.
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
from scripts.ridge import CAP, fit_predict, ranks
from scripts.run_strategy import cache_path, load_panel
from scripts.stage7_importance import per_bar_ic

NQ = 5


def profile(score: np.ndarray, T: pd.DataFrame, label: str) -> dict:
    d = pd.DataFrame({
        "s": score,
        "b": T.index.get_level_values("t_obs").to_numpy(),
        "res": (T["fwd_ret"] - T["beta"] * T["fwd_rm"]).to_numpy(),
        "tgt": T["target"].to_numpy()}).dropna()
    d["q"] = d.groupby("b")["s"].transform(
        lambda s: pd.qcut(s.rank(method="first"), NQ, labels=False)
        if len(s) >= NQ else np.nan)
    d = d.dropna(subset=["q"])
    g = d.groupby("q")
    res_bps = (g["res"].mean() * 1e4)
    tgt = g["tgt"].mean()
    n = g.size()
    print(f"\n  {label}   ({d['b'].nunique():,} bars)")
    print(f"    {'bin':<6}{'n':>9}{'residual bps':>15}{'target':>11}")
    for q in range(NQ):
        tag = "Q1 short" if q == 0 else ("Q5 long" if q == NQ - 1 else f"Q{q+1}")
        print(f"    {tag:<6}{n.get(q,0):>9,}{res_bps.get(q,np.nan):>+15.2f}"
              f"{tgt.get(q,np.nan):>+11.4f}")
    spread = float(res_bps.get(NQ - 1, np.nan) - res_bps.get(0, np.nan))
    mono = bool(np.all(np.diff(res_bps.to_numpy()) > 0))
    print(f"    Q5-Q1 spread {spread:+.2f} bps   "
          f"{'MONOTONE' if mono else 'NOT monotone'}")
    return {"res_bps": res_bps.to_dict(), "target": tgt.to_dict(),
            "spread_bps": spread, "monotone": mono}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=1000.0)
    ap.add_argument("--out", default="features/monotonicity.json")
    a = ap.parse_args()

    print("QUINTILE PROFILE -- does the book sit on the working part "
          "of its own signal?")

    lo = splits.TRUE_HOLDOUT2_MS
    Xtr, Ttr, cols, _ = load_panel(lo, lag=a.lag)
    Xte, Tte, _, _ = load_panel(CAP, lower_ms=lo, lag=a.lag)
    Rtr, _ = ranks(Xtr)
    Rte, _ = ranks(Xte)
    Ytr = Ttr["target"].groupby(level="t_obs").rank(pct=True).to_numpy()
    bars_tr = Xtr.index.get_level_values("t_obs").to_numpy()

    out = {}

    # ---- TRAIN, honestly out-of-fold ------------------------------------
    tr_only = bars_tr < splits.TRUE_HOLDOUT1_MS
    sc = pd.Series(np.nan, index=Xtr.index)
    for f in foldmod.make_folds(np.sort(np.unique(bars_tr[tr_only]))):
        tr, te = np.isin(bars_tr, f.train), np.isin(bars_tr, f.test)
        p = fit_predict(Rtr[tr], Ytr[tr], Rtr[te], a.alpha)
        sc[te] = (p - p.mean()) / (p.std() or 1.0)
    m = sc.notna().to_numpy()
    out["train_oof"] = profile(sc[m].to_numpy(), Ttr[m],
                               f"TRAIN (out-of-fold, ridge a={a.alpha:g})")

    # ---- 2026 ------------------------------------------------------------
    p = fit_predict(Rtr, Ytr, Rte, a.alpha)
    out["h2026"] = profile((p - p.mean()) / (p.std() or 1.0), Tte,
                           f"2026 (ridge a={a.alpha:g})")

    # ---- LightGBM on 2026, for the same picture --------------------------
    import lightgbm as lgb
    from scripts.stage7_importance import LGB_PARAMS, SEEDS
    acc = np.zeros(len(Xte))
    for sd in SEEDS:
        mdl = lgb.LGBMRegressor(**LGB_PARAMS, random_state=sd).fit(
            Rtr.to_numpy("float32"), Ytr)
        q = mdl.predict(Rte.to_numpy("float32"))
        acc += (q - q.mean()) / (q.std() or 1.0)
    out["h2026_lgbm"] = profile(acc / len(SEEDS), Tte, "2026 (lightgbm)")

    print("\n" + "=" * 72)
    t, h = out["train_oof"]["spread_bps"], out["h2026"]["spread_bps"]
    if t > 0 and h < 0:
        print("  Q5-Q1 flipped sign between train and 2026 -> REGIME FLIP.")
        print("  The book was specified correctly; the tail relationship")
        print("  reversed. Nothing in the construction is broken.")
    elif t < 0:
        print("  Q5-Q1 is NEGATIVE on TRAIN TOO. The long/short book has been")
        print("  reading the inverted end of its own signal all along, and the")
        print("  train Sharpe came from something other than this spread.")
        print("  That points at the CONSTRUCTION, not the signal.")
    else:
        print(f"  train Q5-Q1 {t:+.2f} bps, 2026 {h:+.2f} bps.")

    pth = DATA_DIR / a.out
    pth.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
