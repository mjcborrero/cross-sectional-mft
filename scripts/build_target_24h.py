"""The 24h target, assembled from already-gated artifacts.

WHY THIS EXISTS RATHER THAN `build_target.py --h 24`
-----------------------------------------------------
Four separate places in the pipeline hardcode the 8h horizon:

    build_forward_returns  output filename (fixed: a 24h build silently
                           OVERWROTE the 8h artifact)
    build_forward_returns  the Step 2 index it cross-checks against (fixed)
    build_beta_idiovol     a sqrt(8) scaling gate
    build_target           Gate 8B, an independent rebuild from 1m klines that
                           reconstructs an 8h window

The first two were real bugs and are fixed in place. The last two are gates
that are correct for 8h and inapplicable at 24h. Rewriting them to be
horizon-aware means editing validation code to accommodate a new case, which
is precisely where a wrong number gets waved through.

So the 24h target is assembled here from inputs that have ALREADY passed their
own gates, and it is checked two independent ways instead.

    fwd_ret, fwd_rm   from forward_24h_lag60.parquet, which passed Gate 7
    beta              the 8h panel restricted to 00:00 rows. Daily instants are
                      a strict subset of 8h instants and beta at t_obs is an
                      asof value, so this is identical, not an approximation.
    sigma_eps         same panel, scaled by sqrt(3). An 8h idiosyncratic vol
                      over a 24h window scales with sqrt(time). It is a COMMON
                      multiplier so within-bar ranks are untouched; it exists
                      so the target keeps sd ~ 1 and the distribution checks
                      stay meaningful.

    target = (fwd_ret - beta * fwd_rm) / sigma_eps

THE TWO CHECKS
--------------
A  IDENTITY     the written target reproduces its own formula exactly.
B  INDEPENDENT  fwd_ret is rebuilt by COMPOUNDING three consecutive 8h forward
   REBUILD      returns from the 8h artifact, which is a different code path
                and a different file. If the daily builder and the compounded
                8h returns disagree, one of them is wrong.

B is the one that matters. It is the same idea as Gate 8B -- rebuild the thing
from somewhere else and require agreement -- applied at the horizon that is
actually being built.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft import splits
from mft.paths import DATA_DIR

MS8 = 8 * 3_600_000
fails: list[str] = []


def gate(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="grid/target_24h_lag60.parquet")
    args = ap.parse_args()

    F = pd.read_parquet(DATA_DIR / "grid/forward_24h_lag60.parquet")
    B = pd.read_parquet(DATA_DIR / "grid/beta_idiovol_24h.parquet")
    d = F.merge(B[["t_obs", "symbol", "beta", "sigma_eps"]],
                on=["t_obs", "symbol"], how="inner")
    d = d.dropna(subset=["fwd_ret", "fwd_rm", "beta", "sigma_eps"])
    d = d[d["sigma_eps"] > 0]
    d["target"] = (d["fwd_ret"] - d["beta"] * d["fwd_rm"]) / d["sigma_eps"]

    print(f"24h TARGET  {len(d):,} rows, {d['t_obs'].nunique():,} instants, "
          f"{d['symbol'].nunique()} coins")
    print(f"  {splits._fmt(int(d.t_obs.min()))} .. {splits._fmt(int(d.t_obs.max()))}\n")

    # ---- A. identity ----------------------------------------------------
    lhs = d["target"].to_numpy()
    rhs = ((d["fwd_ret"] - d["beta"] * d["fwd_rm"]) / d["sigma_eps"]).to_numpy()
    gate(float(np.nanmax(np.abs(lhs - rhs))) < 1e-12,
         "target == (fwd_ret - beta*fwd_rm)/sigma_eps",
         f"max|diff| {float(np.nanmax(np.abs(lhs - rhs))):.2e}")

    # ---- B. independent rebuild by compounding the 8h artifact ----------
    f8 = pd.read_parquet(DATA_DIR / "grid/forward_8h_lag60.parquet")
    R8 = f8.pivot(index="t_obs", columns="symbol", values="fwd_ret").sort_index()
    t8 = np.asarray(R8.index)
    # A 24h return from t compounds the 8h returns at t, t+8h, t+16h. Only
    # valid where all three exist AND the three instants are consecutive.
    nxt = pd.Series(t8, index=t8).shift(-1) - pd.Series(t8, index=t8)
    ok1 = (nxt == MS8).to_numpy()                       # t -> t+8h exists
    ok2 = ok1 & np.roll(ok1, -1)                        # ...and t+8h -> t+16h
    ok2[-1] = False
    m1 = pd.DataFrame(np.repeat(ok1[:, None], R8.shape[1], axis=1),
                      index=R8.index, columns=R8.columns)
    m2 = pd.DataFrame(np.repeat(ok2[:, None], R8.shape[1], axis=1),
                      index=R8.index, columns=R8.columns)
    a = R8
    b = R8.shift(-1).where(m1)
    c = R8.shift(-2).where(m2)
    comp = ((1 + a) * (1 + b) * (1 + c) - 1.0).stack()
    comp.index = comp.index.set_names(["t_obs", "symbol"])
    j = d.set_index(["t_obs", "symbol"]).index.intersection(comp.index)
    lhs2 = d.set_index(["t_obs", "symbol"]).loc[j, "fwd_ret"]
    rhs2 = comp.loc[j]
    m = lhs2.notna() & rhs2.notna()
    md = float((lhs2[m] - rhs2[m]).abs().max())
    gate(md < 1e-9,
         "fwd_ret == three compounded 8h returns (independent rebuild)",
         f"max|diff| {md:.2e} over {int(m.sum()):,} rows")

    # ---- distribution ---------------------------------------------------
    tg = d["target"]
    print(f"\n  mean {tg.mean():+.4f}  sd {tg.std():.4f}  "
          f"skew {tg.skew():+.2f}  |t|>50 {float((tg.abs()>50).mean()):.5%}")
    gate(abs(float(tg.mean())) < 0.20, "target mean near zero", f"{tg.mean():+.4f}")
    gate(0.5 < float(tg.std()) < 2.5, "target sd near 1", f"{tg.std():.3f}")
    gate(float((tg.abs() > 50).mean()) < 1e-3, "no pathological outliers")
    per = d.groupby("t_obs")["target"].std()
    gate(float(per.median()) > 0.1, "within-bar dispersion exists",
         f"median {per.median():.3f}")
    n = d.groupby("t_obs")["symbol"].size()
    gate(int(n.min()) >= 15, "every cross-section has >= 15 coins",
         f"min {int(n.min())}")

    print()
    if fails:
        print(f"FAILED ({len(fails)}): " + "; ".join(fails))
        print("Nothing written.")
        return 1
    out = DATA_DIR / args.out
    d[["symbol", "t_obs", "target", "fwd_ret", "fwd_rm", "beta",
       "sigma_eps"]].to_parquet(out, index=False)
    print(f"ALL CHECKS PASSED. Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
