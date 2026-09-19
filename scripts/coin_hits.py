"""Per-coin hits from the saved rank tables: does the ranking help for THIS coin?

WHY THE BASE RATE IS PER COIN
------------------------------
Every coin has its own drift against the volume-weighted index over train.
BTC beat it (BTC is half the index and outperformed ETH); most alts trailed
it. So "BTC hits 52% when long" is not evidence of ranking skill unless BTC
hits LESS than 52% when the ranking is ignored. The base for each coin is its
own unconditional P(resid > 0) across every bar it appears in, and the lift is
measured against that -- for shorts, against 1 - base.

    lift_long  = hit_long  - P(resid > 0 | coin)
    lift_short = hit_short - P(resid < 0 | coin)

A coin whose long-lift and short-lift are BOTH positive is one the ranking
genuinely times. A coin with a high long hit and a negative short hit is one
the ranking is merely riding.

z is the lift in standard errors of a proportion. With ~900 observations per
cell, |z| > 2 is roughly the 5% line; with 20 coins x 2 sides = 40 cells, one
or two past it are expected by chance.

Both models are shown side by side because agreement between them says the
pattern is the COIN's; disagreement says it is the model's.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft.paths import DATA_DIR


def per_coin(path: Path) -> pd.DataFrame:
    D = pd.read_parquet(path)
    sgn = np.where(D["side"] == "long", 1.0, -1.0)
    D["resid"] = D["ret_resid"] * sgn                  # unsigned again
    base = D.groupby("symbol")["resid"].apply(lambda r: (r > 0).mean())
    rows = []
    for sym, g in D.groupby("symbol"):
        b = base[sym]
        r = {"coin": sym.replace("USDT", ""), "base": b}
        for side, bs in (("long", b), ("short", 1 - b)):
            s = g[g["side"] == side]
            n = len(s)
            h = s["hit_resid"].mean() if n else np.nan
            se = np.sqrt(bs * (1 - bs) / n) if n else np.nan
            r[f"n_{side}"] = n
            r[f"hit_{side}"] = h
            r[f"lift_{side}"] = h - bs
            r[f"z_{side}"] = (h - bs) / se if n else np.nan
            r[f"bps_{side}"] = s["ret_resid"].mean() * 1e4 if n else np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index("coin")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="8h", help="which rank_hits tables: 8h "
                    "(rank_hits_8h_{ridge,lgbm}) or days (rank_hits{,_lgbm})")
    a = ap.parse_args()
    if a.tag == "8h":
        files = {"lgbm": "rank_hits_8h_lgbm", "ridge": "rank_hits_8h_ridge"}
    else:
        files = {"lgbm": "rank_hits_lgbm", "ridge": "rank_hits"}

    T = {m: per_coin(DATA_DIR / f"features/{f}.parquet") for m, f in files.items()}
    L = T["lgbm"].sort_values("lift_long", ascending=False)
    R = T["ridge"].reindex(L.index)

    print(f"PER-COIN HITS -- {a.tag}, train, out-of-fold, resid space")
    print("  base = the coin's OWN P(resid>0); lift is against that, not the pool\n")
    hdr = (f"  {'coin':<6}{'base':>6} | {'LGBM long':^24} | {'LGBM short':^24} | "
           f"{'ridge long':^12} | {'ridge short':^12}")
    sub = (f"  {'':<6}{'':>6} | {'n':>5}{'hit':>7}{'lift':>7}{'z':>5} | "
           f"{'n':>5}{'hit':>7}{'lift':>7}{'z':>5} | {'lift':>6}{'z':>5} | {'lift':>6}{'z':>5}")
    print(hdr); print(sub); print("  " + "-" * 104)
    for c, g in L.iterrows():
        r = R.loc[c]
        print(f"  {c:<6}{g['base']:>6.1%} | {g['n_long']:>5}{g['hit_long']:>7.1%}"
              f"{g['lift_long']:>+7.1%}{g['z_long']:>+5.1f} | {g['n_short']:>5}"
              f"{g['hit_short']:>7.1%}{g['lift_short']:>+7.1%}{g['z_short']:>+5.1f} | "
              f"{r['lift_long']:>+6.1%}{r['z_long']:>+5.1f} | "
              f"{r['lift_short']:>+6.1%}{r['z_short']:>+5.1f}")

    print("\n  SUMMARY (LightGBM)")
    both = L[(L["lift_long"] > 0) & (L["lift_short"] > 0)]
    sig = L[(L["z_long"].abs() > 2) | (L["z_short"].abs() > 2)]
    print(f"    coins with POSITIVE lift on both sides : {len(both)}/20  "
          + (", ".join(both.index) if len(both) else "none"))
    print(f"    cells with |z| > 2 (of 40)             : "
          f"{int((L['z_long'].abs() > 2).sum() + (L['z_short'].abs() > 2).sum())}"
          f"   (expect ~2 by chance)")
    print(f"    mean lift long  {L['lift_long'].mean():+.2%}   "
          f"mean lift short {L['lift_short'].mean():+.2%}")
    print(f"    spread of coin base rates: {L['base'].min():.1%} .. {L['base'].max():.1%}"
          f"   <- the drift the pooled base rate hides")

    agree = np.corrcoef(L["lift_long"], R["lift_long"])[0, 1]
    print(f"\n  ridge/LGBM agreement on per-coin long lift: corr {agree:+.2f}"
          f"   ({'coin-driven' if agree > 0.5 else 'model-driven or noise'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
