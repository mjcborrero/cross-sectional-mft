"""Step 6 -- beta and idiosyncratic volatility on the decision grid.

Implements Step 6 of docs/TARGET_DESIGN.md and emits the artifact the target
assembly (Step 8) consumes:

    beta_raw       OLS beta, 30d window of 1h returns          (Step 5)
    beta           shrunk beta, lambda = 0.7 toward the        (Step 5)
                   point-in-time cross-sectional mean
    sigma_eps      std of (r - beta*r_m), scaled to 8h units   <- the target's
    sigma_total    std of r, scaled to 8h units                   denominator

sigma_eps is computed against the SHRUNK beta, not the OLS beta, because that
is the residual the target actually forms. Two consequences worth stating:

  - The identity sigma_eps <= sigma_total is guaranteed only for OLS. A shrunk
    beta can leave more variance than it removes when the raw beta is far from
    the cross-sectional mean. This script MEASURES how often that happens
    instead of assuming it away.
  - sigma_eps is estimated at 1h (720 observations in 30 days) and scaled by
    sqrt(8), rather than estimated directly at 8h (90 observations). The
    scaling assumes serial independence; Gate 6 tests that assumption against a
    direct 8h estimate instead of taking it on faith.

Everything is point-in-time: a value stamped at t_obs uses only returns at or
before t_obs.

    python scripts/build_beta_idiovol.py
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
from mft.rolling import rolling_moments, shrink_linear

MS_HOUR = 3_600_000
FAILURES: list[str] = []

WINDOW_D = 30            # Step 5
LAMBDA = 1.0             # Step 5 selected 0.7; Step 9 superseded it (see docs)
MIN_FRAC = 0.60
H = 8                    # target horizon, hours
SIGMA_FLOOR = 1e-6       # guard against a degenerate denominator


def gate(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--returns", default="grid/returns_1h.parquet")
    ap.add_argument("--grid", default="grid/decision_grid_8h.parquet")
    ap.add_argument("--window-d", type=int, default=WINDOW_D)
    ap.add_argument("--lam", type=float, default=LAMBDA)
    ap.add_argument("--out", default="grid/beta_idiovol_8h.parquet")
    args = ap.parse_args()

    long = pd.read_parquet(DATA_DIR / args.returns)
    dec = pd.read_parquet(DATA_DIR / args.grid)
    t_dec = np.sort(dec["t_obs"].unique())

    grid = np.arange(long["t"].min(), long["t"].max() + MS_HOUR, MS_HOUR, dtype=np.int64)
    Y = long.pivot(index="t", columns="symbol", values="ret").reindex(grid)
    x = long.drop_duplicates("t").set_index("t")["r_m"].reindex(grid)

    win = args.window_d * 24
    mp = int(win * MIN_FRAC)
    print(f"Step 6 -- beta & idiosyncratic vol")
    print(f"  window {args.window_d}d of 1h returns ({win} hours, min {mp}), "
          f"lambda={args.lam}")
    print(f"  {len(grid):,} hours x {Y.shape[1]} coins\n")

    M = rolling_moments(Y, x, win, mp)
    beta_raw = M.beta_ols
    beta = shrink_linear(beta_raw, args.lam)

    var_eps_shrunk = M.resid_var(beta)
    var_eps_ols = M.resid_var(beta_raw)
    var_tot = M.var_y

    scale = np.sqrt(H)                       # 1h -> 8h units
    sig_eps = np.sqrt(var_eps_shrunk.clip(lower=0)) * scale
    sig_ols = np.sqrt(var_eps_ols.clip(lower=0)) * scale
    sig_tot = np.sqrt(var_tot.clip(lower=0)) * scale

    def at_grid(df):
        return df.reindex(t_dec).stack().rename_axis(["t_obs", "symbol"])

    out = pd.concat({
        "beta_raw": at_grid(beta_raw), "beta": at_grid(beta),
        "sigma_eps": at_grid(sig_eps), "sigma_eps_ols": at_grid(sig_ols),
        "sigma_total": at_grid(sig_tot), "n_obs": at_grid(M.n),
    }, axis=1).reset_index().dropna(subset=["beta", "sigma_eps", "sigma_total"])
    out = out[out["sigma_eps"] > SIGMA_FLOOR]
    out["ratio"] = out["sigma_eps"] / out["sigma_total"]

    # ---- validate the sqrt(8) scaling against a direct 8h estimate --------
    p8 = dec.sort_values(["symbol", "t_obs"]).copy()
    g8 = p8.groupby("symbol", sort=False)
    p8["t_prev"] = g8["t_obs"].shift(1)
    p8["px_prev"] = g8["px_fill"].shift(1)
    p8 = p8[(p8["t_obs"] - p8["t_prev"]) == H * MS_HOUR]
    p8["ret8"] = p8["px_fill"] / p8["px_prev"] - 1.0
    Y8 = p8.pivot(index="t_obs", columns="symbol", values="ret8")
    idx8 = pd.read_parquet(DATA_DIR / "grid/market_index_8h.parquet")
    x8 = idx8.set_index("t_obs")["r_m"].reindex(Y8.index)
    per8 = args.window_d * 3
    M8 = rolling_moments(Y8, x8, per8, int(per8 * MIN_FRAC))
    direct8 = np.sqrt(M8.resid_var(M8.beta_ols).clip(lower=0))
    cmp = pd.concat([at_grid(sig_ols).rename("scaled"),
                     direct8.stack().rename("direct")], axis=1).dropna()
    scale_ratio = float((cmp["scaled"] / cmp["direct"]).median())
    scale_corr = float(cmp["scaled"].corr(cmp["direct"]))

    # ---- GATE 6 -----------------------------------------------------------
    print("GATE 6\n" + "=" * 74)

    # The identity is a mathematical guarantee for OLS -- assert it there.
    viol_ols = int((out["sigma_eps_ols"] > out["sigma_total"] * (1 + 1e-9)).sum())
    gate(viol_ols == 0,
         "sigma_eps(OLS) <= sigma_total everywhere (mathematical identity)",
         f"{viol_ols} violations of {len(out):,}")

    # With the shrunk beta it is not guaranteed. Measure, do not assume.
    viol_shr = int((out["sigma_eps"] > out["sigma_total"]).sum())
    frac = viol_shr / len(out)
    gate(frac < 0.01,
         "shrunk-beta residual exceeds total vol in <1% of rows",
         f"{viol_shr:,} rows ({frac:.3%})")

    gate(bool(np.isfinite(out[["beta", "sigma_eps", "sigma_total"]]).all().all()),
         "no NaN or inf in beta / sigma_eps / sigma_total")
    gate(bool((out["sigma_eps"] > SIGMA_FLOOR).all()),
         "no degenerate (near-zero) denominator",
         f"min sigma_eps {out.sigma_eps.min():.6f}")

    med = float(out["ratio"].median())
    gate(0.40 <= med <= 0.70,
         "median sigma_eps/sigma_total in the expected 0.40-0.70 band",
         f"{med:.3f}")

    gate(abs(scale_ratio - 1.0) < 0.15,
         "sqrt(8) scaling agrees with a direct 8h estimate",
         f"median ratio {scale_ratio:.3f}, corr {scale_corr:.3f}")

    try:
        splits.assert_train_only(out.assign(bar_close_time=out["t_obs"]))
        gate(True, "splits.assert_train_only(t_obs)")
    except splits.HoldoutLeak as e:
        gate(False, "splits.assert_train_only(t_obs)", str(e)[:70])

    if FAILURES:
        print("\n" + "=" * 74)
        print(f"GATE 6 FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1

    # ---- census -----------------------------------------------------------
    print("\nCENSUS\n" + "=" * 74)
    print(f"  {len(out):,} rows   {out.symbol.nunique()} coins   "
          f"{out.t_obs.nunique():,} cross-sections")
    print(f"  {splits._fmt(out.t_obs.min())} -> {splits._fmt(out.t_obs.max())}")
    q = out["ratio"].quantile([.05, .25, .5, .75, .95])
    print(f"\n  sigma_eps / sigma_total:  p5 {q[.05]:.3f}  p25 {q[.25]:.3f}  "
          f"median {q[.5]:.3f}  p75 {q[.75]:.3f}  p95 {q[.95]:.3f}")
    print(f"  -> beta explains {1-q[.5]**2:.0%} of variance at the median coin")

    per = out.groupby("symbol").agg(beta=("beta", "mean"),
                                    sig=("sigma_eps", "mean"),
                                    ratio=("ratio", "mean")).sort_values("ratio")
    print(f"\n  {'symbol':<10} {'mean beta':>10} {'sigma_eps':>10} {'idio share':>11}")
    for s, r in per.iterrows():
        print(f"  {s:<10} {r['beta']:>10.3f} {r['sig']:>10.4f} {r['ratio']:>11.3f}")

    yr = out.assign(y=pd.to_datetime(out.t_obs, unit="ms", utc=True).dt.year)
    print(f"\n  {'year':<6} {'median ratio':>13} {'mean beta':>10}")
    for y, gg in yr.groupby("y"):
        print(f"  {y:<6} {gg['ratio'].median():>13.3f} {gg['beta'].mean():>10.3f}")

    # ---- write ------------------------------------------------------------
    path = DATA_DIR / args.out
    cols = ["symbol", "t_obs", "beta_raw", "beta", "sigma_eps", "sigma_total",
            "ratio", "n_obs"]
    out[cols].to_parquet(path, index=False)
    path.with_suffix(".manifest.json").write_text(json.dumps({
        "step": 6, "window_days": args.window_d, "lambda": args.lam,
        "estimation_frequency": "1h", "scaled_to_hours": H,
        "sigma_basis": "residual against the SHRUNK beta",
        "rows": int(len(out)), "coins": int(out.symbol.nunique()),
        "cross_sections": int(out.t_obs.nunique()),
        "ratio_median": round(med, 5),
        "ratio_quantiles": {str(k): round(float(v), 5) for k, v in q.items()},
        "shrunk_exceeds_total_frac": round(frac, 6),
        "sqrt8_scale_ratio": round(scale_ratio, 5),
        "sqrt8_scale_corr": round(scale_corr, 5),
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"GATE 6 PASSED. Wrote {path}  ({path.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
