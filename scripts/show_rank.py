"""Print an actual cross-section as the ridge model ranks it.

Everything else in this project reports aggregates. This prints the object those
aggregates are made of: one bar, twenty coins, in the order the model puts them,
with the weight each one receives and what it went on to do.

BARS ARE CHOSEN MECHANICALLY, NOT PICKED
-----------------------------------------
Two bars are shown: the MEDIAN bar of the quiet dispersion tercile and the
MEDIAN bar of the violent tercile. Median-by-construction, so neither is a
flattering example -- and the pair is the point, because the quiet/violent split
is where the edge lives and dies (+9.27 bps vs -35.52 bps in 2026).

COLUMNS
-------
  score     ridge output, standardised across the bar
  rank      the model's ordering, 1 = most negative view
  weight    what the BOOK actually holds: demeaned rank / sigma_eps,
            renormalised to gross 1. Negative = short.
  sig_eps   idiosyncratic vol, 8h units. The divisor in the weight.
  resid     what happened: fwd_ret - beta*fwd_rm, in bps
  contrib   weight * resid, in bps. These sum to the bar's book return.

A bar where the model is right has positive contributions concentrated at both
ENDS -- big shorts on coins that fell, big longs on coins that rose.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft import splits, strategy
from mft.paths import DATA_DIR
from scripts.ridge import CAP, fit_predict, ranks
from scripts.run_strategy import attach_funding, load_panel


def show(t: int, P: pd.DataFrame, tag: str) -> None:
    g = P.xs(t, level="t_obs").copy()
    g["rank"] = g["score"].rank().astype(int)
    g["resid_bps"] = (g["fwd_ret"] - g["beta"] * g["fwd_rm"]) * 1e4
    g["contrib_bps"] = g["weight"] * g["resid_bps"]
    g = g.sort_values("score")
    ts = pd.to_datetime(t, unit="ms", utc=True)

    disp = g["resid_bps"].std()
    print(f"\n{'='*78}")
    print(f"{tag}   {ts:%Y-%m-%d %H:%M} UTC")
    print(f"  cross-sectional dispersion {disp:.0f} bps   "
          f"market return {g['fwd_rm'].iloc[0]*1e4:+.0f} bps")
    print(f"{'='*78}")
    print(f"  {'rank':>4} {'symbol':<12}{'score':>8}{'weight':>9}"
          f"{'sig_eps':>9}{'resid':>9}{'contrib':>10}")
    print("  " + "-" * 74)
    for _, r in g.iterrows():
        side = "SHORT" if r["weight"] < 0 else "LONG "
        print(f"  {r['rank']:>4} {r.name:<12}{r['score']:>+8.2f}"
              f"{r['weight']:>+9.3f}{r['sigma_eps']:>9.4f}"
              f"{r['resid_bps']:>+9.0f}{r['contrib_bps']:>+10.2f}   {side}")
    print("  " + "-" * 74)
    tot = g["contrib_bps"].sum()
    lo5, hi5 = g.head(5)["contrib_bps"].sum(), g.tail(5)["contrib_bps"].sum()
    print(f"  {'':>4} {'BOOK RETURN':<12}{'':>8}{g['weight'].abs().sum():>+9.3f}"
          f"{'':>9}{'':>9}{tot:>+10.2f} bps")
    print(f"       short end (5 most negative)  {lo5:>+8.2f} bps")
    print(f"       long  end (5 most positive)  {hi5:>+8.2f} bps")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=1000.0)
    args = ap.parse_args()

    lo = splits.TRUE_HOLDOUT2_MS
    Xtr, Ttr, cols, _ = load_panel(lo, lag=args.lag)
    Xte, Tte, _, _ = load_panel(CAP, lower_ms=lo, lag=args.lag)
    Rtr, _ = ranks(Xtr)
    Rte, _ = ranks(Xte)
    Ytr = Ttr["target"].groupby(level="t_obs").rank(pct=True).to_numpy()

    p = fit_predict(Rtr, Ytr, Rte, args.alpha)
    P = Tte.assign(score=(p - p.mean()) / (p.std() or 1.0))
    P = P.assign(weight=strategy.target_weights(P))

    print(f"RIDGE (alpha={args.alpha:g}) -- 2026 cross-sections")
    print("  fitted on all data before 2026, applied forward.")

    resid = (P["fwd_ret"] - P["beta"] * P["fwd_rm"])
    disp = resid.groupby(level="t_obs").std().sort_index()
    terc = pd.qcut(disp, 3, labels=["quiet", "mid", "violent"])

    for name in ("quiet", "violent"):
        d = disp[terc == name].sort_values()
        t = int(d.index[len(d) // 2])          # the MEDIAN bar of the tercile
        show(t, P, f"MEDIAN {name.upper()} BAR")

    print(f"\n{'='*78}")
    print("  Same model, same weights, 8 hours apart in the same year.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
