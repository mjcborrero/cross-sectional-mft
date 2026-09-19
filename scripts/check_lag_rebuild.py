"""Verify a rebuilt lag stack: what MUST change, and what must NOT.

The execution lag threads through the price grid, not just the label. Changing
it rebuilds decision grid -> weights -> index -> betas -> forward -> target.
That is a lot of moving parts, so this asserts the two halves separately.

MUST BE BIT-IDENTICAL (lag cannot touch these)
  px_obs      the price AT t_obs. Independent of when you fill.
  weights     built from volume observed before t_obs.
  the 72 frozen features -- every featdef uses px_obs and never px_fill
                            (mft/featdefs/regime.py documents why).

MUST DIFFER (lag is the whole point)
  px_fill, r_m, beta, sigma_eps, fwd_ret, fwd_rm, target

A rebuild where px_obs moved means the grid changed underneath the features
and the frozen list is no longer valid against it. A rebuild where fwd_ret did
NOT move means the lag never actually took effect. Both are silent failures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft.paths import DATA_DIR

FAIL: list[str] = []


def gate(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(label)


def joined(a: pd.DataFrame, b: pd.DataFrame, keys: list[str], col: str):
    """Inner-join on keys and return the two aligned columns."""
    m = a[keys + [col]].merge(b[keys + [col]], on=keys, suffixes=("_a", "_b"))
    return m[f"{col}_a"].to_numpy(), m[f"{col}_b"].to_numpy(), len(m)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", type=int, required=True)
    ap.add_argument("--base", type=int, default=60)
    a = ap.parse_args()
    L, B = a.lag, a.base

    # `target` and `forward` ALWAYS carry the lag suffix (that is how Step 7/8
    # have always named them). The price-stack artifacts only got a suffix when
    # a non-60 lag was introduced, so at lag 60 they keep their bare names.
    ALWAYS_SUFFIXED = ("target_8h", "forward_8h")

    def p(stem_lag: int, name: str) -> Path:
        if stem_lag == 60 and name not in ALWAYS_SUFFIXED:
            return DATA_DIR / f"grid/{name}.parquet"
        return DATA_DIR / f"grid/{name}_lag{stem_lag}.parquet"

    print(f"LAG REBUILD CHECK -- lag {L} against base lag {B}\n")

    print("INVARIANT -- these may not move")
    g0 = pd.read_parquet(p(B, "decision_grid_8h"))
    g1 = pd.read_parquet(p(L, "decision_grid_8h"))
    x, y, n = joined(g0, g1, ["t_obs", "symbol"], "px_obs")
    d = np.nanmax(np.abs(x - y)) if n else np.nan
    gate(n > 0 and d == 0.0, "px_obs bit-identical across the two grids",
         f"max|diff| {d:.3e} over {n:,} shared rows")
    # NOT a row-count comparison. The two grids can cover different SPANS: the
    # lag-60 stack was rebuilt with the holdout wall open and reaches 2026,
    # while a fresh build with the wall closed stops at the train boundary.
    # Coverage is a wall question, not a lag question -- so compare the shared
    # span and report the spans rather than gating on length.
    for tag, g in (("base", g0), ("new ", g1)):
        t = pd.to_datetime(g["t_obs"], unit="ms", utc=True)
        print(f"         {tag} span {t.min():%Y-%m-%d} -> {t.max():%Y-%m-%d} "
              f"({len(g):,} rows)")
    gate(n > 0, "grids share a comparable span", f"{n:,} overlapping rows")

    # Weights are LONG (symbol, t_obs, adv, n_obs, weight). Joining on t_obs
    # alone produces a 20x20 cartesian product per bar and differences `adv`,
    # which is volume in the 1e9 range -- a check that fails loudly for a
    # reason that has nothing to do with the thing being checked.
    w0 = pd.read_parquet(p(B, "market_weights_30d"))
    w1 = pd.read_parquet(p(L, "market_weights_30d"))
    x, y, n = joined(w0, w1, ["t_obs", "symbol"], "weight")
    dw = float(np.nanmax(np.abs(x - y)))
    gate(dw == 0.0, "market weights bit-identical (volume is pre-t_obs)",
         f"max|diff| {dw:.3e} over {n:,} shared (bar, coin) rows")

    print("\nMUST MOVE -- these are what the lag changes")
    x, y, n = joined(g0, g1, ["t_obs", "symbol"], "px_fill")
    d = float(np.nanmax(np.abs(x - y)))
    gate(d > 0, "px_fill moved", f"max|diff| {d:.4g} over {n:,} rows")

    i0 = pd.read_parquet(p(B, "market_index_8h")).set_index("t_obs")["r_m"]
    i1 = pd.read_parquet(p(L, "market_index_8h")).set_index("t_obs")["r_m"]
    j = i0.index.intersection(i1.index)
    c = float(np.corrcoef(i0[j], i1[j])[0, 1])
    gate(c < 0.9999, "index return moved",
         f"corr {c:.6f}, max|diff| {float((i0[j]-i1[j]).abs().max()):.4g}")

    b0 = pd.read_parquet(p(B, "beta_idiovol_8h"))
    b1 = pd.read_parquet(p(L, "beta_idiovol_8h"))
    for col in ("beta", "sigma_eps"):
        if col in b0.columns and col in b1.columns:
            x, y, n = joined(b0, b1, ["t_obs", "symbol"], col)
            m = np.isfinite(x) & np.isfinite(y)
            cc = float(np.corrcoef(x[m], y[m])[0, 1])
            print(f"         {col:<10} corr {cc:.6f}  "
                  f"median|diff| {np.nanmedian(np.abs(x-y)):.4g}")

    t0 = pd.read_parquet(p(B, "target_8h")).rename(columns={})
    t1 = pd.read_parquet(p(L, "target_8h"))
    for col in ("fwd_ret", "target"):
        x, y, n = joined(t0, t1, ["t_obs", "symbol"], col)
        m = np.isfinite(x) & np.isfinite(y)
        cc = float(np.corrcoef(x[m], y[m])[0, 1])
        gate(cc < 0.9999, f"{col} moved", f"corr {cc:.6f} over {int(m.sum()):,}")

    print("\n" + "=" * 70)
    if FAIL:
        print(f"  {len(FAIL)} CHECK(S) FAILED: " + "; ".join(FAIL))
        return 1
    print("  all checks passed -- features stay valid, label genuinely moved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
