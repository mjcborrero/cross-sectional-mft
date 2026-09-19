"""Stage 5 -- implied cost. Per docs/FEATURE_EVALUATION.md §7.

Target-free, and NOT A KILL CRITERION. §7: "Priced information, obtained free."
Nothing is removed here and nothing may be. The output is a price tag attached
to each survivor, which becomes decisive only later, when two features have
similar value and different cost.

WHAT IS MEASURED
----------------
The pre-registered quantity is rank autocorrelation at lag 1:

    near 1.0   a static tilt -- ranks the same coins the same way every bar.
               Cheap to trade, but carries no timing.
    near 0     churn, paid for in slippage even under the zero-fee assumption.

Measured between CONSECUTIVE decision instants, over the coins present in both,
in within-bar percentile-rank space -- the same space Stage 4 used, because
ranks are all a ranking objective consumes.

Alongside it, the cost that autocorrelation implies is reported directly:
`quintile_turnover` is the fraction of a top-4/bottom-4 book that has to be
replaced at each 8h rebalance if the book is formed on this feature alone.
Autocorrelation is the pre-registered number; turnover is the same fact in the
units the P&L will actually charge, and the two are reported together so the
translation is visible rather than assumed.

k = 4 is roughly a quintile of a ~19-coin cross-section, and a bar needs 2k
coins for a top-k and a bottom-k book to be disjoint.

WHO IS EXEMPT
-------------
Context (bar-level) columns, for the same reason Stage 3 exempts them: they
produce no cross-sectional ordering, so there is no book to turn over and the
question is undefined rather than badly answered.

A HIGH SCORE IS NOT A GOOD SCORE
--------------------------------
Both ends of this scale are expensive in different currencies. A feature that
persists at +0.9 across 90-day-disjoint snapshots will price as nearly free to
trade -- and that is exactly what makes it a static tilt with no timing in it. Reading low turnover as quality would invert the stage's purpose.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft.paths import DATA_DIR
from mft.featdefs.carry import Carry
from mft.featdefs.positioning import Positioning
from mft.featdefs.liquidity import Liquidity
from mft.featdefs.path import PathAttention
from mft.featdefs.regime import ReturnRegime
from mft.featdefs.relational import Relational
from mft.featdefs.spotperp import SpotPerp
from mft.featdefs.inherited import Inherited
from mft.featdefs.inherited2 import Inherited2

_FAMS = (Carry, Positioning, Liquidity, PathAttention, ReturnRegime,
         Relational, SpotPerp, Inherited, Inherited2)
CONTEXT = set()
for _f in _FAMS:
    CONTEXT |= set(getattr(_f, "context_columns", ()))

FAMILIES = {
    "A": "family_A_carry", "B": "family_B_positioning", "C": "family_C_liquidity",
    "D": "family_D_path", "E": "family_E_regime", "F": "family_F_relational",
    "G": "family_G_spotperp", "H": "family_H_inherited",
    "I": "family_I_inherited2",
}

K = 4              # top-k / bottom-k book; ~a quintile of ~19 coins
MIN_COINS = 2 * K  # a bar needs 2k coins for the two sides to be disjoint


def rank_autocorr(W: pd.DataFrame) -> tuple[float, int]:
    """Mean lag-1 rank correlation between consecutive bars.

    Computed on coins available in BOTH bars, centred within each bar so a
    row-wise Pearson on ranks equals that bar-pair's Spearman.
    """
    R = W.rank(axis=1, pct=True)
    A, B = R, R.shift(1)
    m = A.notna() & B.notna()
    n = m.sum(axis=1)
    A, B = A.where(m), B.where(m)
    Ac = A.sub(A.mean(axis=1), axis=0)
    Bc = B.sub(B.mean(axis=1), axis=0)
    den = np.sqrt((Ac ** 2).sum(axis=1) * (Bc ** 2).sum(axis=1))
    rho = ((Ac * Bc).sum(axis=1) / den.replace(0.0, np.nan)).where(n >= MIN_COINS)
    rho = rho.replace([np.inf, -np.inf], np.nan).dropna()
    return (float(rho.mean()) if len(rho) else np.nan), len(rho)


def quintile_turnover(W: pd.DataFrame) -> tuple[float, float]:
    """Fraction of a top-K/bottom-K book replaced per rebalance, and the median
    realised book size.

    NORMALISED BY THE REALISED BOOK, NOT BY K. A tie at the cutoff puts every
    tied name in the book, so `rank <= K` can select more than K. That is not
    an edge case here: the sector tier takes one value per sector, six per bar,
    so `sector_cohesion` selects a median of 5 names for a "top-4" book and
    dividing by K produced a turnover of -12.8%. Negative turnover is
    impossible, which is the only reason the error was visible at all -- the
    same mis-normalisation was quietly deflating every other coarse feature's
    cost by a few points without ever going out of range.

    Dividing by max(|S_t|, |S_t-1|) is bounded in [0, 1] and reduces to the
    plain formula whenever the two books are the same size, which is every
    per-coin feature.

    Done in numpy: `DataFrame.where(series, ...)` aligns the series to the
    frame's COLUMNS, not its rows, so masking bars with a per-bar eligibility
    series silently scrambles it.
    """
    ok = (W.notna().sum(axis=1) >= MIN_COINS).to_numpy()
    top = (W.rank(axis=1, ascending=False).to_numpy() <= K) & ok[:, None]
    bot = (W.rank(axis=1, ascending=True).to_numpy() <= K) & ok[:, None]
    both = ok[1:] & ok[:-1]
    if not both.any():
        return np.nan, np.nan

    def side(S):
        keep = (S[1:] & S[:-1]).sum(axis=1)[both]
        size = np.maximum(S[1:].sum(axis=1), S[:-1].sum(axis=1))[both]
        return 1.0 - keep / np.where(size > 0, size, np.nan)

    t = np.nanmean(np.concatenate([side(top), side(bot)]))
    return float(t), float(np.median(top.sum(axis=1)[ok]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features/stage5_turnover.json")
    args = ap.parse_args()

    owner, parts = {}, []
    for code, fname in FAMILIES.items():
        p = DATA_DIR / f"features/{fname}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p).set_index(["t_obs", "symbol"])
        for c in d.columns:
            owner[c] = code
        parts.append(d)
    X = pd.concat(parts, axis=1)

    s4 = json.loads((DATA_DIR / "features/stage4_redundancy.json").read_text())
    survivors = [c for c in X.columns if c not in s4["absorbed"]]

    print(f"Stage 5 -- implied cost   {len(survivors)} survivors of "
          f"{len(X.columns)}, {len(X):,} rows")
    print(f"top-{K}/bottom-{K} book, 8h rebalance; NOT a kill criterion\n")

    rows = []
    for c in survivors:
        if c in CONTEXT:
            rows.append({"feature": c, "family": owner[c], "autocorr": np.nan,
                         "turnover": np.nan, "book": np.nan, "n_pairs": 0,
                         "context": True})
            continue
        W = X[c].unstack("symbol")
        rho, n = rank_autocorr(W)
        tno, book = quintile_turnover(W)
        rows.append({"feature": c, "family": owner[c], "autocorr": rho,
                     "turnover": tno, "book": book, "n_pairs": n,
                     "context": False})

    R = pd.DataFrame(rows).set_index("feature")
    ctx = R[R["context"]]
    M = R[~R["context"]].sort_values("autocorr", ascending=False)

    if len(ctx):
        print(f"CONTEXT TIER -- exempt ({len(ctx)}): no cross-sectional "
              f"ordering, so no book to turn over")
        for c in ctx.index:
            print(f"    {c}")
        print()

    print(f"{'feature':<34} {'fam':<4} {'rank autocorr':>13} {'turnover/8h':>12}"
          f" {'book':>5}")
    print("=" * 80)
    for c in M.index:
        b = M.loc[c, "book"]
        tie = "" if b == K else f"  <- {b:.0f} names, ties at the cutoff"
        print(f"{c:<34} {M.loc[c,'family']:<4} "
              f"{M.loc[c,'autocorr']:>13.3f} {M.loc[c,'turnover']:>11.1%} "
              f"{b:>5.0f}{tie}")

    print("\n" + "=" * 74)
    q = M["autocorr"]
    print(f"static tilts   (autocorr >= 0.95): {int((q >= 0.95).sum()):>3}")
    print(f"slow           (0.80 .. 0.95):     {int(((q >= 0.80) & (q < 0.95)).sum()):>3}")
    print(f"medium         (0.50 .. 0.80):     {int(((q >= 0.50) & (q < 0.80)).sum()):>3}")
    print(f"fast, expensive(autocorr < 0.50):  {int((q < 0.50).sum()):>3}")
    print("\nNothing is removed by this stage. §7: priced information, "
          "obtained free.")

    out = DATA_DIR / args.out
    out.write_text(json.dumps({
        "stage": 5, "kill_criterion": False, "k": K, "min_coins": MIN_COINS,
        "rebalance": "8h",
        "results": {c: {"family": R.loc[c, "family"],
                        "autocorr": None if pd.isna(R.loc[c, "autocorr"])
                        else float(R.loc[c, "autocorr"]),
                        "turnover": None if pd.isna(R.loc[c, "turnover"])
                        else float(R.loc[c, "turnover"]),
                        "book_size": None if pd.isna(R.loc[c, "book"])
                        else float(R.loc[c, "book"]),
                        "n_pairs": int(R.loc[c, "n_pairs"]),
                        "context": bool(R.loc[c, "context"])} for c in R.index},
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
