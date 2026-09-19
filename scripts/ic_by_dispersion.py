"""Is the IC positive only on bars where nothing is at stake?

THE CONTRADICTION THIS RESOLVES
--------------------------------
In 2026 the mean per-bar Spearman IC is POSITIVE (+0.031 on target, +0.027 on
the dollar residual) while the pooled quintile means are INVERTED (Q1 highest,
Q5 near the bottom) and the book loses 15%.

Both can be true at once, because they weight bars differently:

  * per-bar IC gives EVERY BAR EQUAL WEIGHT. A bar where the cross-section is
    nearly flat counts as much as one where it is wide.
  * P&L weights each bar by HOW MUCH MOVED. A bar with wide dispersion
    dominates the pooled mean and the book's return.

So a model that ranks correctly on quiet bars and incorrectly on violent ones
posts a positive IC and loses money. The IC is not wrong; it is answering a
question nobody trades.

THE TEST
--------
Split bars into terciles by cross-sectional dispersion (std of the dollar
residual within the bar, known only afterwards -- this is a DIAGNOSTIC, not a
tradable filter) and report IC and the Q5-Q1 dollar spread in each.

If IC is positive in the quiet tercile and negative in the violent one, the
"positive IC" is an artifact of equal-weighting bars, and IR ~ IC*sqrt(breadth)
never applied to this book in the first place.

Also reports DISPERSION-WEIGHTED IC -- each bar weighted by its dispersion,
which is the version that corresponds to what the book actually earns.
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
from scripts.run_strategy import load_panel

K = 5


def per_bar_table(score: np.ndarray, T: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({
        "s": score,
        "b": T.index.get_level_values("t_obs").to_numpy(),
        "res": (T["fwd_ret"] - T["beta"] * T["fwd_rm"]).to_numpy()}).dropna()
    rows = []
    for b, g in d.groupby("b"):
        if len(g) < 2 * K:
            continue
        r = g[["s", "res"]].rank()
        ic = float(r["s"].corr(r["res"]))
        o = g.sort_values("s")
        rows.append({"b": b, "ic": ic, "disp": float(g["res"].std()),
                     "spread": float(o["res"].iloc[-K:].mean()
                                     - o["res"].iloc[:K].mean())})
    return pd.DataFrame(rows).dropna()


def report(tab: pd.DataFrame, label: str) -> dict:
    tab = tab.copy()
    tab["t"] = pd.qcut(tab["disp"], 3, labels=["quiet", "mid", "violent"])
    print(f"\n  {label}   ({len(tab):,} bars)")
    print(f"    {'tercile':<9}{'bars':>7}{'disp bps':>11}{'mean IC':>10}"
          f"{'Q5-Q1 bps':>12}{'share of P&L':>14}")
    tot = float(tab["spread"].sum())
    out = {}
    for t, g in tab.groupby("t", observed=True):
        sh = float(g["spread"].sum()) / tot if tot else np.nan
        print(f"    {str(t):<9}{len(g):>7,}{g['disp'].mean()*1e4:>11.1f}"
              f"{g['ic'].mean():>+10.4f}{g['spread'].mean()*1e4:>+12.2f}"
              f"{sh:>+14.1%}")
        out[str(t)] = {"bars": len(g), "ic": float(g["ic"].mean()),
                       "spread_bps": float(g["spread"].mean() * 1e4)}
    plain = float(tab["ic"].mean())
    w = float(np.average(tab["ic"], weights=tab["disp"]))
    print(f"    {'':9}{'':7}{'':11}{'':10}")
    print(f"    plain IC (every bar equal)        {plain:>+8.4f}")
    print(f"    dispersion-weighted IC            {w:>+8.4f}   "
          f"<- what the book earns")
    out["plain_ic"] = plain
    out["disp_weighted_ic"] = w
    out["overall_spread_bps"] = float(tab["spread"].mean() * 1e4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=1000.0)
    ap.add_argument("--out", default="features/ic_by_dispersion.json")
    a = ap.parse_args()

    lo = splits.TRUE_HOLDOUT2_MS
    Xtr, Ttr, cols, _ = load_panel(lo, lag=a.lag)
    Xte, Tte, _, _ = load_panel(CAP, lower_ms=lo, lag=a.lag)
    Rtr, _ = ranks(Xtr)
    Rte, _ = ranks(Xte)
    Ytr = Ttr["target"].groupby(level="t_obs").rank(pct=True).to_numpy()
    bars_tr = Xtr.index.get_level_values("t_obs").to_numpy()

    print("IC BY CROSS-SECTIONAL DISPERSION")
    print("  per-bar IC weights every bar equally; P&L does not.")

    out = {}
    tr_only = bars_tr < splits.TRUE_HOLDOUT1_MS
    sc = pd.Series(np.nan, index=Xtr.index)
    for f in foldmod.make_folds(np.sort(np.unique(bars_tr[tr_only]))):
        tr, te = np.isin(bars_tr, f.train), np.isin(bars_tr, f.test)
        p = fit_predict(Rtr[tr], Ytr[tr], Rtr[te], a.alpha)
        sc[te] = (p - p.mean()) / (p.std() or 1.0)
    m = sc.notna().to_numpy()
    out["train_oof"] = report(per_bar_table(sc[m].to_numpy(), Ttr[m]),
                              "TRAIN (out-of-fold, ridge)")

    p = fit_predict(Rtr, Ytr, Rte, a.alpha)
    out["h2026"] = report(
        per_bar_table((p - p.mean()) / (p.std() or 1.0), Tte), "2026 (ridge)")

    print("\n" + "=" * 72)
    t, h = out["train_oof"], out["h2026"]
    print(f"  train: plain {t['plain_ic']:+.4f} -> disp-weighted "
          f"{t['disp_weighted_ic']:+.4f}")
    print(f"  2026 : plain {h['plain_ic']:+.4f} -> disp-weighted "
          f"{h['disp_weighted_ic']:+.4f}")
    if h["plain_ic"] > 0 > h["disp_weighted_ic"]:
        print("\n  CONFIRMED: the 2026 IC is positive only because quiet bars")
        print("  count as much as violent ones. Weighted the way P&L weights")
        print("  them, the edge is NEGATIVE. The positive IC was never")
        print("  tradable, and IR ~ IC*sqrt(breadth) did not apply.")

    pth = DATA_DIR / a.out
    pth.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
