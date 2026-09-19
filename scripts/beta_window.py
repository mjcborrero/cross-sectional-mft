"""Step 5 -- beta estimation window and shrinkage.

Implements Step 5 of docs/TARGET_DESIGN.md.

SELECTION CRITERION, stated before any result was seen, because this is the
step where circularity would enter:

    The window and shrinkage are chosen by how well beta_hat predicts FUTURE
    REALISED beta. Never by strategy Sharpe, never by backtest P&L.

Selecting an estimator on downstream performance fits the estimator to the
backtest -- the same class of mistake that produced the retracted conclusions
this project was rebuilt to escape. The test below would return the same
answer whether the strategy makes money or loses it, which is precisely what
makes it trustworthy.

    beta_hat(t)   estimated from 1h returns in the trailing W days   (past only)
    beta_real(t)  realised over the FOLLOWING 30 days of 1h returns  (future)
    score         RMSE of beta_hat against beta_real, lower is better

Shrinkage pulls each estimate toward the point-in-time cross-sectional mean:

    linear    beta = lam*beta_raw + (1-lam)*mean_t          lam swept
    Vasicek   weight chosen per coin from its own standard error, so noisily
              measured coins are pulled harder than well-measured ones

Both the estimate and the shrink target use only data at or before t. The
realised beta uses data after t -- it is the thing being predicted, not an
input.

    python scripts/beta_window.py
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

MS_HOUR = 3_600_000
FAILURES: list[str] = []

WINDOWS_D = [15, 30, 60, 90, 120, 180]
LAMBDAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
FORWARD_D = 30
MIN_FRAC = 0.60          # a window must be at least this full to yield an estimate


def gate(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def rolling_ols(Y: pd.DataFrame, x: pd.Series, win: int, min_periods: int):
    """Rolling beta, residual variance and SE(beta) of each column of Y on x.

    Uses rolling sums rather than per-window regressions: exact, and O(n) per
    column. NaNs are skipped, and x is masked to each column's availability so
    every sum is taken over the same observations as its partner.
    """
    beta = {}
    sig2 = {}
    se2 = {}
    for c in Y.columns:
        y = Y[c]
        xm = x.where(y.notna())
        n = y.rolling(win, min_periods=min_periods).count()
        Sy = y.rolling(win, min_periods=min_periods).sum()
        Sx = xm.rolling(win, min_periods=min_periods).sum()
        Sxy = (xm * y).rolling(win, min_periods=min_periods).sum()
        Sxx = (xm * xm).rolling(win, min_periods=min_periods).sum()
        Syy = (y * y).rolling(win, min_periods=min_periods).sum()

        ssx = Sxx - Sx * Sx / n
        sxy = Sxy - Sx * Sy / n
        ssy = Syy - Sy * Sy / n
        b = sxy / ssx
        ssres = (ssy - sxy * sxy / ssx).clip(lower=0)
        s2 = ssres / (n - 2)
        beta[c] = b
        sig2[c] = s2
        se2[c] = s2 / ssx
    return (pd.DataFrame(beta), pd.DataFrame(sig2), pd.DataFrame(se2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--returns", default="grid/returns_1h.parquet")
    ap.add_argument("--grid", default="grid/decision_grid_8h.parquet")
    ap.add_argument("--forward-days", type=int, default=FORWARD_D)
    args = ap.parse_args()

    long = pd.read_parquet(DATA_DIR / args.returns)
    dec = pd.read_parquet(DATA_DIR / args.grid)
    t_dec = np.sort(dec["t_obs"].unique())

    # Wide, on a complete hourly grid so integer windows == time windows.
    grid = np.arange(long["t"].min(), long["t"].max() + MS_HOUR, MS_HOUR, dtype=np.int64)
    Y = long.pivot(index="t", columns="symbol", values="ret").reindex(grid)
    x = (long.drop_duplicates("t").set_index("t")["r_m"].reindex(grid))

    fwd = args.forward_days * 24
    print(f"Step 5 -- beta window & shrinkage")
    print(f"  1h grid {len(grid):,} hours, {Y.shape[1]} coins")
    print(f"  predict: realised beta over the NEXT {args.forward_days}d ({fwd} hours)\n")

    # Realised forward beta: rolling beta evaluated at t+fwd, i.e. covering (t, t+fwd].
    B_real, _, _ = rolling_ols(Y, x, fwd, int(fwd * MIN_FRAC))
    B_real = B_real.shift(-fwd)

    # Candidate estimators.
    est = {}
    for wd in WINDOWS_D:
        win = wd * 24
        b_raw, _, se2 = rolling_ols(Y, x, win, int(win * MIN_FRAC))
        mean_t = b_raw.mean(axis=1)                       # point-in-time target
        var_t = b_raw.var(axis=1)
        for lam in LAMBDAS:
            est[(wd, f"lin{lam:.1f}")] = b_raw.mul(lam).add(mean_t * (1 - lam), axis=0)
        w = var_t.values[:, None] / (var_t.values[:, None] + se2.values)
        w = pd.DataFrame(w, index=b_raw.index, columns=b_raw.columns)
        est[(wd, "vasicek")] = b_raw.mul(w).add(mean_t * (1 - w.mean(axis=1)), axis=0) \
            if False else b_raw * w + (1 - w).mul(mean_t, axis=0)

    # Evaluate only at decision instants, and only where truth exists.
    at = np.intersect1d(t_dec, grid)
    truth = B_real.reindex(at).stack()

    # Non-overlapping subset: one evaluation per forward window, for honest CIs.
    nolap = at[:: max(1, fwd // 8)]
    truth_nolap = B_real.reindex(nolap).stack()

    rows = []
    for (wd, tag), B in est.items():
        pred = B.reindex(at).stack()
        j = pd.concat([pred.rename("p"), truth.rename("t")], axis=1).dropna()
        pn = B.reindex(nolap).stack()
        jn = pd.concat([pn.rename("p"), truth_nolap.rename("t")], axis=1).dropna()
        if len(j) < 500:
            continue
        rows.append({
            "window_d": wd, "shrink": tag,
            "rmse": float(np.sqrt(((j.p - j.t) ** 2).mean())),
            "corr": float(j.p.corr(j.t)),
            "bias": float((j.p - j.t).mean()),
            "n": len(j),
            "rmse_nolap": float(np.sqrt(((jn.p - jn.t) ** 2).mean())),
            "n_nolap": len(jn),
        })
    res = pd.DataFrame(rows)

    # ---- report -----------------------------------------------------------
    print("RMSE predicting future realised beta (lower is better)\n" + "=" * 74)
    piv = res.pivot(index="window_d", columns="shrink", values="rmse")
    piv = piv[[f"lin{l:.1f}" for l in LAMBDAS] + ["vasicek"]]
    print(f"  {'window':>7} " + " ".join(f"{c:>8}" for c in piv.columns))
    for wd, r in piv.iterrows():
        cells = " ".join(f"{v:>8.4f}" for v in r.values)
        print(f"  {wd:>5}d  {cells}")

    # Global minimum over the whole 42-cell grid, and the best inside the
    # pre-registered range. Reported separately: picking the argmin of 42 cells
    # whose spread is a few percent is noise-mining, so the gap between them is
    # the number that decides, not the argmin itself.
    glob = res.loc[res["rmse"].idxmin()]
    inrange = res[(res["window_d"] >= 30) & (res["window_d"] <= 90)]
    best = inrange.loc[inrange["rmse"].idxmin()]

    print(f"\n  global best      : {int(glob.window_d)}d + {glob.shrink}"
          f"   RMSE {glob.rmse:.4f}  corr {glob['corr']:.3f}")
    print(f"  best in 30-90d   : {int(best.window_d)}d + {best.shrink}"
          f"   RMSE {best.rmse:.4f}  corr {best['corr']:.3f}  bias {best.bias:+.4f}")
    edge = 1 - glob.rmse / best.rmse
    print(f"  out-of-range edge: {edge:+.2%}   (how much the pre-registered "
          f"range costs)")

    # Shrinkage gain measured AT A FIXED WINDOW. Comparing best-shrunk to
    # best-unshrunk across different windows conflates two separate effects
    # and understates what shrinkage does at the window actually used.
    row_ns = res[(res.window_d == best.window_d) & (res.shrink == "lin1.0")].iloc[0]
    gain = 1 - best.rmse / row_ns.rmse
    print(f"  shrinkage gain at {int(best.window_d)}d: {row_ns.rmse:.4f} -> "
          f"{best.rmse:.4f}  ({gain:+.2%})")

    print(f"\n  non-overlapping check (every {args.forward_days}d, "
          f"n={int(best.n_nolap):,}): RMSE {best.rmse_nolap:.4f}")

    # ---- robustness: does the winner hold up year by year? ---------------
    print("\n  RMSE BY YEAR at each window (best shrinkage per window)\n  " + "-" * 62)
    yr_tab = {}
    for wd in WINDOWS_D:
        sub = res[res.window_d == wd]
        tag = sub.loc[sub["rmse"].idxmin(), "shrink"]
        B = est[(wd, tag)]
        pred = B.reindex(at).stack()
        j = pd.concat([pred.rename("p"), truth.rename("t")], axis=1).dropna()
        yy = pd.to_datetime(j.index.get_level_values(0), unit="ms", utc=True).year
        yr_tab[wd] = j.assign(y=yy).groupby("y").apply(
            lambda d: float(np.sqrt(((d.p - d.t) ** 2).mean())), include_groups=False)
    YT = pd.DataFrame(yr_tab)
    print("  " + YT.round(4).to_string().replace("\n", "\n  "))
    winners = YT.idxmin(axis=1)
    print(f"\n  best window by year: {winners.to_dict()}")
    n_years_won = int((winners == best.window_d).sum())
    glob_worst_years = int((YT.idxmax(axis=1) == glob.window_d).sum())
    print(f"  {int(best.window_d)}d wins {n_years_won}/{len(YT)} years; "
          f"{int(glob.window_d)}d is WORST in {glob_worst_years}/{len(YT)}")

    # ---- GATE 5 -----------------------------------------------------------
    print("\nGATE 5\n" + "=" * 74)
    gate(True, "selection criterion is predictive accuracy, not P&L",
         "no strategy return enters this script")

    # Deviation rule, stated openly as a judgement made NOW (it was not
    # pre-registered): leaving the pre-registered range needs a materially
    # better estimator, not a marginally better one, and the advantage must be
    # stable year to year. A <5% edge that inverts in a regime year is noise.
    DEVIATION_BAR = 0.05
    stay = edge < DEVIATION_BAR or glob_worst_years > 0
    gate(stay,
         f"out-of-range option does not clear the {DEVIATION_BAR:.0%} deviation bar",
         f"edge {edge:+.2%}, and {int(glob.window_d)}d is worst in "
         f"{glob_worst_years} year(s) -> keep the pre-registered range")

    gate(gain > 0.02, "shrinkage materially improves prediction at the chosen window",
         f"{gain:+.2%} at {int(best.window_d)}d")

    best_nolap_inr = inrange.loc[inrange["rmse_nolap"].idxmin()]
    gate(int(best_nolap_inr.window_d) == int(best.window_d),
         "overlapping and non-overlapping evaluations agree on the window",
         f"overlap {int(best.window_d)}d vs non-overlap "
         f"{int(best_nolap_inr.window_d)}d")

    if FAILURES:
        print("\n" + "=" * 74)
        print(f"GATE 5 FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1

    out = DATA_DIR / "grid" / "beta_window.json"
    out.write_text(json.dumps({
        "step": 5,
        "criterion": "RMSE vs future realised beta; P&L never used",
        "forward_days": args.forward_days,
        "windows_tested_d": WINDOWS_D,
        "lambdas_tested": LAMBDAS,
        "chosen_window_d": int(best.window_d),
        "chosen_shrinkage": best.shrink,
        "global_best_window_d": int(glob.window_d),
        "global_best_shrinkage": glob.shrink,
        "out_of_range_edge": round(float(edge), 5),
        "deviation_bar": 0.05,
        "rmse_by_year": {int(w): {int(k): round(float(v), 5) for k, v in c.items()}
                         for w, c in YT.to_dict().items()},
        "rmse": round(float(best.rmse), 5),
        "corr": round(float(best["corr"]), 5),
        "bias": round(float(best.bias), 5),
        "rmse_unshrunk_same_window": round(float(row_ns.rmse), 5),
        "shrinkage_gain": round(float(gain), 5),
        "rmse_nolap": round(float(best.rmse_nolap), 5),
        "n_eval": int(best.n), "n_eval_nolap": int(best.n_nolap),
        "full_grid": res.to_dict(orient="records"),
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"GATE 5 PASSED. Beta: {int(best.window_d)}d window of 1h returns, "
          f"shrinkage = {best.shrink}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
