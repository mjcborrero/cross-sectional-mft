"""Is the dispersion split TRADABLE, or only visible after the fact?

WHY THIS DECIDES WHETHER THE FINDING IS USABLE
-----------------------------------------------
scripts/ic_by_dispersion.py split bars by the cross-sectional dispersion of the
FORWARD residual and found the edge concentrated in quiet bars:

                    train            2026
    quiet    Q5-Q1  +9.11 bps       +9.27 bps
    mid             +9.17           +10.27
    violent        -14.57          -35.52

The quiet/mid spread is essentially IDENTICAL across train and 2026. The whole
failure is the violent tercile. That is a mechanism, not a curve fit -- but the
split uses a quantity dated AFTER the decision, so as written it is a
post-mortem, not a strategy.

It becomes tradable if and only if dispersion is PREDICTABLE from information
available at t_obs. Volatility is famously persistent, so the prior is good,
but the prior is not evidence.

WHAT IS MEASURED
----------------
Trailing cross-sectional dispersion of realised residuals, computed strictly
before t_obs, against the forward dispersion the split used. Reported as rank
correlation and, more usefully, as a CONFUSION RATE: how often a bar that
trailing dispersion calls "quiet" actually turns out violent. That error rate,
not the correlation, is what determines whether a gate helps or hurts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft import splits
from mft.paths import DATA_DIR
from scripts.ridge import CAP
from scripts.run_strategy import load_panel

WINS = [3, 9, 30, 90]        # bars of trailing history (1d, 3d, 10d, 30d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", type=int, default=60)
    ap.add_argument("--out", default="features/dispersion_predictable.json")
    a = ap.parse_args()

    lo = splits.TRUE_HOLDOUT2_MS
    _, Ttr, _, _ = load_panel(lo, lag=a.lag)
    _, Tte, _, _ = load_panel(CAP, lower_ms=lo, lag=a.lag)

    out = {}
    for label, T in (("TRAIN", Ttr), ("2026", Tte)):
        res = (T["fwd_ret"] - T["beta"] * T["fwd_rm"])
        disp = res.groupby(level="t_obs").std().sort_index()
        print(f"\n{label}  ({len(disp):,} bars, mean dispersion "
              f"{disp.mean()*1e4:.1f} bps)")
        print(f"  {'trailing win':<14}{'rank corr':>11}{'P(violent|called quiet)':>26}")
        row = {}
        # Forward tercile: the label the P&L split used.
        fwd_t = pd.qcut(disp, 3, labels=[0, 1, 2]).astype(float)
        for w in WINS:
            # STRICTLY before t_obs: rolling mean of PAST dispersions, shifted.
            trail = disp.rolling(w, min_periods=max(2, w // 2)).mean().shift(1)
            m = trail.notna() & fwd_t.notna()
            rc = float(trail[m].rank().corr(disp[m].rank()))
            pred_t = pd.qcut(trail[m], 3, labels=[0, 1, 2]).astype(float)
            called_quiet = pred_t == 0
            p_bad = float((fwd_t[m][called_quiet] == 2).mean())
            print(f"  {w:>3} bars      {rc:>11.4f}{p_bad:>26.1%}")
            row[f"win{w}"] = {"rank_corr": rc, "p_violent_given_quiet": p_bad}
        out[label] = row
        print(f"  (a coin-flip gate would show 33.3%; lower is better)")

    print("\n" + "=" * 68)
    best = max(WINS, key=lambda w: out["2026"][f"win{w}"]["rank_corr"])
    r26 = out["2026"][f"win{best}"]
    print(f"  best on 2026: {best}-bar trailing, rank corr "
          f"{r26['rank_corr']:.3f}, "
          f"P(violent|called quiet) {r26['p_violent_given_quiet']:.1%}")
    if r26["p_violent_given_quiet"] < 0.25:
        print("  Dispersion IS forecastable well enough to gate on. The")
        print("  quiet/mid edge is reachable with information available at")
        print("  decision time -- worth building and FORWARD testing.")
    else:
        print("  The gate misclassifies too often to be worth it: a bar called")
        print("  quiet is violent nearly as often as chance, so gating would")
        print("  keep much of the losing tercile while discarding good bars.")

    p = DATA_DIR / a.out
    p.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
