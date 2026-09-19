"""Step 8 -- assemble the target.

Implements Step 8 of docs/TARGET_DESIGN.md.

    t_i,t_obs = ( fwd_ret_i - beta_i * fwd_rm ) / sigma_eps_i

combining Step 7's forward legs with Step 6's beta and idiosyncratic vol.

GATE 8 ABORTS; IT DOES NOT PRINT. Two reconstructions, at different depths:

  8A  ALGEBRAIC, every row. Recompute the target from the stored components.
      Catches assembly and join errors -- a beta joined to the wrong
      (symbol, t_obs) is the classic silent killer here.

  8B  FROM RAW, sampled timestamps. Rebuild EVERYTHING from the 1m klines --
      prices, weights, the 1h index, rolling betas, shrinkage, idiosyncratic
      vol -- bypassing every intermediate parquet, and compare. This is the
      check that catches a stale artifact, a convention that drifted between
      steps, or a file rebuilt with different parameters. 8A cannot see any of
      those, because it trusts the same inputs that produced the error.

The distinction matters: re-running the same code path and getting the same
answer proves nothing. Only an independent derivation is evidence.

    python scripts/build_target.py
    python scripts/build_target.py --sample-times 40
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
from mft.data.store import ParquetStore
from mft.paths import DATA_DIR, NORMALIZED_DIR
from mft.rolling import rolling_moments, shrink_linear

MS_MIN = 60_000
MS_HOUR = 3_600_000
TOL_MS = 5 * MS_MIN
H = 8
LAG_MIN = 60
WINDOW_D = 30
LAMBDA = 1.0
MIN_FRAC = 0.60
MIN_COINS_IDX = 10
LAM_RUN = LAMBDA

CORR_MIN = 0.999          # pre-registered in TARGET_DESIGN.md Gate 8
ABS_TOL = 1e-9            # algebraic path must be exact to floating point
RAW_RTOL = 1e-6           # independent path: same maths, so near-exact

FAILURES: list[str] = []


def gate(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def asof_px(ct, px, instants):
    i = np.searchsorted(ct, instants, side="left") - 1
    out = np.full(len(instants), np.nan)
    ok = i >= 0
    if ok.any():
        v = i[ok]
        fresh = (instants[ok] - ct[v]) <= TOL_MS
        sel = np.where(ok)[0][fresh]
        out[sel] = px[v][fresh]
    return out


def rebuild_from_raw(t_targets: np.ndarray, klines: dict, dec: pd.DataFrame,
                     universe: list, lag_min: int = LAG_MIN) -> pd.DataFrame:
    """Independent re-derivation of the target at selected instants.

    Touches no intermediate artifact except the decision grid's own timestamps
    and trailing volumes, which are themselves re-derived checks from Step 0.

    `lag_min` MUST match the lag the forward legs were built at. It used to be
    the module constant, which meant that rebuilding the stack at any other lag
    compared a lag-60 reconstruction against a lag-L target and failed Gate 8B
    on fwd_rm at corr ~0.82 -- the verifier being stale, not the build. Same
    bug class as the Step 7 reference index, one layer down.
    """
    lag_ms = lag_min * MS_MIN
    step_ms = H * MS_HOUR

    # --- forward legs, straight from klines -------------------------------
    fills = np.concatenate([t_targets + lag_ms, t_targets + lag_ms + step_ms])
    pxf = {s: asof_px(*klines[s], fills) for s in universe}
    n = len(t_targets)
    fwd = pd.DataFrame({s: pxf[s][n:] / pxf[s][:n] - 1.0 for s in universe},
                       index=t_targets)

    # --- weights, re-derived from the grid's trailing volumes -------------
    g = dec.sort_values(["symbol", "t_obs"]).copy()
    g["dt"] = pd.to_datetime(g["t_obs"], unit="ms", utc=True)
    adv = {}
    for sym, gg in g.groupby("symbol", sort=False):
        s = gg.set_index("dt")["qv_window"]
        a = s.rolling(f"{WINDOW_D}D", min_periods=WINDOW_D).mean()
        adv[sym] = pd.Series(a.to_numpy(), index=gg["t_obs"].to_numpy())
    ADV = pd.DataFrame(adv)
    Wt = ADV.div(ADV.sum(axis=1), axis=0)

    # --- forward index leg: weights stamped at t_obs ----------------------
    w_at = Wt.reindex(t_targets).reindex(columns=fwd.columns)
    M = w_at.where(fwd.notna())
    M = M.div(M.sum(axis=1), axis=0)
    fwd_rm = (M * fwd).sum(axis=1, min_count=1)

    # --- 1h grid, index, rolling beta and idio vol ------------------------
    lo = int(min(t_targets)) - (WINDOW_D + 20) * 24 * MS_HOUR
    hi = int(max(t_targets)) + MS_HOUR
    hgrid = np.arange(-(-lo // MS_HOUR) * MS_HOUR, hi, MS_HOUR, dtype=np.int64)
    P1 = pd.DataFrame({s: asof_px(*klines[s], hgrid) for s in universe}, index=hgrid)
    R1 = P1.pct_change().replace([np.inf, -np.inf], np.nan)

    Wg = Wt.reindex(Wt.index.union(hgrid)).ffill().reindex(hgrid).shift(1)
    Wg = Wg.reindex(columns=R1.columns)
    M1 = Wg.where(R1.notna())
    M1 = M1.div(M1.sum(axis=1), axis=0)
    rm1 = (M1 * R1).sum(axis=1, min_count=1)
    rm1 = rm1.where(M1.notna().sum(axis=1) >= MIN_COINS_IDX)

    win = WINDOW_D * 24
    mom = rolling_moments(R1, rm1, win, int(win * MIN_FRAC))
    b_raw = mom.beta_ols
    beta = shrink_linear(b_raw, LAM_RUN)
    sig = np.sqrt(mom.resid_var(beta).clip(lower=0)) * np.sqrt(H)

    out = pd.concat({
        "fwd_ret": fwd.stack(), "beta_r": beta.reindex(t_targets).stack(),
        "sigma_r": sig.reindex(t_targets).stack(),
    }, axis=1).rename_axis(["t_obs", "symbol"]).reset_index()
    out["fwd_rm_r"] = out["t_obs"].map(fwd_rm)
    out["target_raw"] = (out["fwd_ret"] - out["beta_r"] * out["fwd_rm_r"]) / out["sigma_r"]
    return out.dropna(subset=["target_raw"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward", default=f"grid/forward_8h_lag{LAG_MIN}.parquet")
    ap.add_argument("--betas", default="grid/beta_idiovol_8h.parquet")
    ap.add_argument("--grid", default="grid/decision_grid_8h.parquet")
    ap.add_argument("--lam", type=float, default=LAMBDA)
    ap.add_argument("--sample-times", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"grid/target_8h_lag{LAG_MIN}.parquet")
    # The execution lag is a REAL parameter, not a constant. It must match the
    # lag the --forward legs were built at; Gate 8B rebuilds from raw klines
    # using it, so a mismatch is caught rather than silently verified away.
    ap.add_argument("--lag", type=int, default=LAG_MIN,
                    help="execution lag in minutes; MUST match --forward")
    ap.add_argument("--weights", default=None,
                    help="weights parquet for the warm-up boundary "
                         "(default: the lag-matched file)")
    args = ap.parse_args()
    lag = args.lag

    global LAM_RUN
    LAM_RUN = args.lam
    fwd = pd.read_parquet(DATA_DIR / args.forward)
    bi = pd.read_parquet(DATA_DIR / args.betas)
    dec = pd.read_parquet(DATA_DIR / args.grid)

    print(f"Step 8 -- assemble target   h={H}h  lag={lag}min\n")

    df = fwd.merge(bi[["symbol", "t_obs", "beta", "beta_raw", "sigma_eps",
                       "sigma_total"]],
                   on=["symbol", "t_obs"], how="inner", validate="one_to_one")
    df["target"] = (df["fwd_ret"] - df["beta"] * df["fwd_rm"]) / df["sigma_eps"]
    df = df.dropna(subset=["target"])

    # ---- warm-up trim -----------------------------------------------------
    # Added 2026-08-05 in response to Gate 8B, NOT pre-registered.
    #
    # Beta is regressed on the market index, but the index only exists once
    # volume weights do (2020-09-28, after their own 30d warm-up). A target
    # dated before weights_start + beta_window therefore has a beta fitted
    # against an index that did not exist for part of its window: median n_obs
    # 572 of 720 there, versus 720 after. Those betas are not wrong so much as
    # resting on a shorter, differently-composed sample -- which is also why an
    # independent rebuild could not reproduce them, since the reproduction
    # depends on exactly when the index blinked into existence.
    #
    # Dropping them costs 0.65% of rows and removes a genuine soft spot rather
    # than merely silencing a check.
    wpath = args.weights or ("grid/market_weights_30d.parquet" if lag == 60
                             else f"grid/market_weights_30d_lag{lag}.parquet")
    wts = pd.read_parquet(DATA_DIR / wpath)
    warmup_end = int(wts["t_obs"].min()) + WINDOW_D * 24 * MS_HOUR
    n_before = len(df)
    df = df[df["t_obs"] >= warmup_end].reset_index(drop=True)
    print(f"  warm-up trim: dropped {n_before - len(df):,} rows before "
          f"{splits._fmt(warmup_end)} "
          f"(beta window not fully supported by a defined index)")

    print(f"  {len(df):,} rows   {df.symbol.nunique()} coins   "
          f"{df.t_obs.nunique():,} cross-sections")

    print("\nGATE 8 -- reconstruction (aborts on failure)\n" + "=" * 74)

    # ---- 8A: algebraic, every row ----------------------------------------
    recomp = (df["fwd_ret"] - df["beta"] * df["fwd_rm"]) / df["sigma_eps"]
    d = (recomp - df["target"]).abs()
    c = float(np.corrcoef(recomp, df["target"])[0, 1])
    gate(c >= CORR_MIN and float(d.max()) < ABS_TOL,
         f"8A algebraic reconstruction (all {len(df):,} rows)",
         f"corr {c:.8f}, max|diff| {d.max():.2e}")

    # ---- 8B: independent, from raw klines --------------------------------
    rng = np.random.default_rng(args.seed)
    times = np.sort(rng.choice(np.sort(df["t_obs"].unique()),
                               size=min(args.sample_times, df.t_obs.nunique()),
                               replace=False))
    universe = sorted(dec["symbol"].unique())
    store = ParquetStore(NORMALIZED_DIR)
    wall = splits.HOLDOUT1_START_MS
    klines = {}
    for s in universe:
        kl = store.load("perp_klines", s, columns=["close_time", "close"])
        kl = kl[kl["close_time"] < wall].sort_values("close_time")
        klines[s] = (kl["close_time"].to_numpy(), kl["close"].to_numpy())

    raw = rebuild_from_raw(times, klines, dec, universe, lag_min=lag)
    chk = df.merge(raw[["symbol", "t_obs", "target_raw", "beta_r", "sigma_r",
                        "fwd_rm_r"]],
                   on=["symbol", "t_obs"], how="inner")
    rel = ((chk["target_raw"] - chk["target"]).abs()
           / chk["target"].abs().clip(lower=1e-6))
    c2 = float(chk["target_raw"].corr(chk["target"]))
    gate(c2 >= CORR_MIN and float(rel.max()) < RAW_RTOL,
         f"8B independent rebuild from 1m klines ({len(chk):,} rows, "
         f"{len(times)} instants)",
         f"corr {c2:.8f}, max rel diff {rel.max():.2e}")

    # Component-level agreement, so a failure above localises immediately.
    for name, a, b in [("beta", "beta", "beta_r"),
                       ("sigma_eps", "sigma_eps", "sigma_r"),
                       ("fwd_rm", "fwd_rm", "fwd_rm_r")]:
        r = ((chk[b] - chk[a]).abs() / chk[a].abs().clip(lower=1e-12)).max()
        gate(float(r) < RAW_RTOL, f"     component {name} agrees",
             f"max rel diff {r:.2e}")

    # ---- structural gates -------------------------------------------------
    gate(bool(np.isfinite(df["target"]).all()), "no NaN or inf in target")
    gate(bool(df.groupby(["t_obs", "symbol"]).size().max() == 1),
         "one target per (t_obs, symbol)")
    try:
        splits.assert_train_only(df.assign(bar_close_time=df["t_obs"]))
        gate(True, "splits.assert_train_only(t_obs)")
    except splits.HoldoutLeak as e:
        gate(False, "splits.assert_train_only(t_obs)", str(e)[:70])
    sizes = df.groupby("t_obs").size()
    gate(int(sizes.min()) >= 15, "every cross-section has >= 15 coins",
         f"min {int(sizes.min())}, mean {sizes.mean():.1f}")

    if FAILURES:
        print("\n" + "=" * 74)
        print(f"GATE 8 FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        print("ABORTING. Nothing written -- a mis-assembled target produces")
        print("plausible-looking results that are wrong by multiples.")
        return 1

    # ---- census -----------------------------------------------------------
    print("\nCENSUS\n" + "=" * 74)
    q = df["target"].quantile([.01, .05, .25, .5, .75, .95, .99])
    print(f"  {splits._fmt(df.t_obs.min())} -> {splits._fmt(df.t_obs.max())}")
    print(f"  mean {df.target.mean():+.4f}   std {df.target.std():.4f}   "
          f"skew {df.target.skew():+.3f}   kurt {df.target.kurtosis():+.2f}")
    print(f"  p1 {q[.01]:+.3f}  p5 {q[.05]:+.3f}  p25 {q[.25]:+.3f}  "
          f"med {q[.5]:+.3f}  p75 {q[.75]:+.3f}  p95 {q[.95]:+.3f}  p99 {q[.99]:+.3f}")
    print(f"  |target| > 5: {(df.target.abs() > 5).mean():.3%}   "
          f"> 10: {(df.target.abs() > 10).mean():.4%}")

    yr = df.assign(y=pd.to_datetime(df.t_obs, unit="ms", utc=True).dt.year)
    print(f"\n  {'year':<6} {'rows':>8} {'mean':>9} {'std':>8} {'xs disp':>9}")
    for y, gg in yr.groupby("y"):
        xs = gg.groupby("t_obs")["target"].std().mean()
        print(f"  {y:<6} {len(gg):>8,} {gg.target.mean():>+9.4f} "
              f"{gg.target.std():>8.4f} {xs:>9.4f}")

    out = DATA_DIR / args.out
    cols = ["symbol", "t_obs", "target", "fwd_ret", "fwd_rm", "beta",
            "sigma_eps", "sigma_total"]
    df[cols].to_parquet(out, index=False)
    out.with_suffix(".manifest.json").write_text(json.dumps({
        "step": 8, "h_hours": H, "lag_min": lag,
        "formula": "(fwd_ret - beta*fwd_rm) / sigma_eps",
        "beta_window_d": WINDOW_D, "beta_lambda": args.lam,
        "beta_frequency": "1h", "sigma_basis": "residual vs shrunk beta, x sqrt(8)",
        "rows": int(len(df)), "coins": int(df.symbol.nunique()),
        "cross_sections": int(df.t_obs.nunique()),
        "gate8a_corr": c, "gate8a_max_abs_diff": float(d.max()),
        "gate8b_corr": c2, "gate8b_max_rel_diff": float(rel.max()),
        "gate8b_instants": int(len(times)), "gate8b_rows": int(len(chk)),
        "warmup_trim_ms": warmup_end, "warmup_rows_dropped": int(n_before - len(df)),
        "target_mean": round(float(df.target.mean()), 6),
        "target_std": round(float(df.target.std()), 6),
        "target_quantiles": {str(k): round(float(v), 5) for k, v in q.items()},
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"GATE 8 PASSED. Wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
